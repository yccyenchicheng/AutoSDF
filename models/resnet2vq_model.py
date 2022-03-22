import os
from collections import OrderedDict

import numpy as np
import einops
import marching_cubes as mcubes
from omegaconf import OmegaConf
from termcolor import colored, cprint
from einops import rearrange, repeat
from tqdm import tqdm

import torch
import torch.nn.functional as F
from torch import nn, optim

import torchvision.utils as vutils
import torchvision.transforms as transforms

from models.base_model import BaseModel
from models.networks.resnet2vq_net import ResNet2VQ
from models.networks.pvqvae_networks.auto_encoder import PVQVAE

import utils
from utils.util_3d import init_mesh_renderer, render_mesh, render_sdf

class ResNet2VQModel(BaseModel):
    def name(self):
        return 'ResNet2VQ-Model'

    def initialize(self, opt):
        BaseModel.initialize(self, opt)
        self.isTrain = opt.isTrain
        self.model_name = self.name()

        # -------------------------------
        # Define Networks
        # -------------------------------

        assert opt.vq_cfg is not None

        resnet_conf = OmegaConf.load(opt.resnet_cfg)
        vq_conf = OmegaConf.load(opt.vq_cfg)
        
        # init resnet2vq network
        self.net = ResNet2VQ(opt)
        self.net.to(opt.device)
        
        # init vqvae for decoding shapes
        mparam = vq_conf.model.params
        n_embed = mparam.n_embed
        embed_dim = mparam.embed_dim
        ddconfig = mparam.ddconfig
        self.vqvae = PVQVAE(ddconfig, n_embed, embed_dim)
        self.load_vqvae(opt.vq_ckpt)
        self.vqvae.to(opt.device)
        self.vqvae.eval()

        if self.isTrain:
            # ----------------------------------
            # define loss functions
            # ----------------------------------
            self.criterion_nll = nn.NLLLoss()
            self.criterion_nce = nn.CrossEntropyLoss()
            self.criterion_nce.to(opt.device)

            # ---------------------------------
            # initialize optimizers
            # ---------------------------------
            self.optimizer = optim.Adam([p for p in self.net.parameters() if p.requires_grad == True], lr=opt.lr)

            if opt.debug == '1':
                self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, 100 if opt.dataset_mode == 'imagenet' else 30, 1.)
            else:
                self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, 100 if opt.dataset_mode == 'imagenet' else 50, 0.9)

            self.optimizers = [self.optimizer]
            self.schedulers = [self.scheduler]

            self.print_networks(verbose=False)

        # transforms
        self.to_tensor = transforms.ToTensor()
        self.normalize = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))

        resolution = resnet_conf.data.resolution
        self.resolution = resolution

        # hyper-parameters for SDF
        if opt.dataset_mode in ['sdf', 'sdf_code', 'sdf_code_v0']:
            nC = resolution
            assert nC == 64, 'right now, only trained with sdf resolution = 64'
            self.down_size = 8   # x: res, x_cube: res//8
            self.cube_size = nC // self.down_size    # size per cube. nC=64, down_size=8 -> size=8 for each smaller cube
            self.stride = nC // self.down_size
            self.ncubes_per_dim = nC // self.cube_size

        # grid size
        if opt.vq_note == 'default':
            self.grid_size = 8
        elif opt.vq_note == '4x4x4':
            self.grid_size = 4

        # setup renderer
        dist, elev, azim = 1.7, 20, 20   
        self.renderer = init_mesh_renderer(image_size=256, dist=dist, elev=elev, azim=azim, device=self.opt.device)

        # for saving best ckpt
        self.best_iou = -1e12

    def load_vqvae(self, vq_ckpt):
        assert type(vq_ckpt) == str         
        state_dict = torch.load(vq_ckpt)

        self.vqvae.load_state_dict(state_dict)
        print(colored('[*] VQVAE: weight successfully load from: %s' % vq_ckpt, 'blue'))

    def set_input(self, input, gen_order=None):
        # x, y = input
        self.x = input['sdf']
        self.x_idx = input['idx']
        self.z_q = input['z_q']
        bs, dz, hz, wz = self.x_idx.shape
        self.z_shape = self.z_q.shape

        self.img = input['img']

        # only use in test_iou
        self.gt_vox = input['gt_vox']

        self.x_idx_seq = rearrange(self.x_idx, 'bs dz hz wz -> (dz hz wz) bs').contiguous() # to (T, B)
        self.x_idx = self.x_idx_seq.clone()

        vars_list = ['x_idx', 'z_q', 'x', 'img']

        self.tocuda(var_names=vars_list)

    def forward(self):        
        self.outp = self.net(self.img)

    def inference(self, data, should_render=False, verbose=False):
        self.net.eval()
        
        self.set_input(data)

        with torch.no_grad():
            self.x = self.x
            self.x_recon = self.vqvae.decode(self.z_q) # could extract this as well

            outp = self.net(self.img)
            outp = F.softmax(outp, dim=1) # compute the prob. of next ele
            outp = outp.argmax(dim=1)
            outp = rearrange(outp, 'bs d h w -> (d h w) bs')
            
            self.x_recon_resnet = self.vqvae.decode_enc_idices(outp, z_spatial_dim=self.grid_size)

            self.img = self.img

        self.net.train()

    def eval_metrics(self, dataloader, thres=0.0):
        self.eval()
        
        ret = OrderedDict([
            ('iou', 0.0),
            ('iou_std', 0.0),
        ])
        self.train()
        return ret

    def backward(self):
        target = self.x_idx
        # target = rearrange(target, '(d h w) bs -> bs d h w', d=8, h=8, w=8)
        target = rearrange(target, '(d h w) bs -> bs d h w', d=self.grid_size, h=self.grid_size, w=self.grid_size)
        outp = self.outp
        
        loss_nll = self.criterion_nce(outp, target)

        self.loss = loss_nll

        self.loss_nll = loss_nll
        # self.loss_rec = loss_rec
        self.loss.backward()

    def optimize_parameters(self, total_steps):
        # self.vqvae.train()

        self.set_requires_grad([self.net], requires_grad=True)

        self.forward()
        self.optimizer.zero_grad()
        self.backward()
        self.optimizer.step()

    def get_current_errors(self):
        
        ret = OrderedDict([
            ('nll', self.loss_nll.data),
        ])

        return ret

    def get_current_visuals(self):

        with torch.no_grad():
            # self.image = self.render(self.x)
            self.image_recon = render_sdf(self.renderer, self.x_recon)
            self.image_recon_resnet = render_sdf(self.renderer, self.x_recon_resnet)
            # self.image_recon_gt = self.render(self.x_recon_gt)
            self.image = self.img
            pass

        vis_tensor_names = [
            'image',
            'image_recon',
            'image_recon_resnet',
        ]

        vis_ims = self.tnsrs2ims(vis_tensor_names)
        visuals = zip(vis_tensor_names, vis_ims)
                            
        return OrderedDict(visuals)


    def save(self, label):

        state_dict = {
            'vqvae': self.vqvae.cpu().state_dict(),
            'resnet2vq': self.net.cpu().state_dict(),
        }

        save_filename = 'resnet2vq_%s.pth' % (label)
        save_path = os.path.join(self.save_dir, save_filename)

        torch.save(state_dict, save_path)
        self.vqvae.to(self.opt.device)
        self.net.to(self.opt.device)

    def load_ckpt(self, ckpt):
        if type(ckpt) == str:
            state_dict = torch.load(ckpt)
        else:
            state_dict = ckpt

        self.vqvae.load_state_dict(state_dict['vqvae'])
        self.net.load_state_dict(state_dict['resnet2vq'])
        print(colored('[*] weight successfully load from: %s' % ckpt, 'blue'))


