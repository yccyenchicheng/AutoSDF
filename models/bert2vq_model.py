
import os
from collections import OrderedDict
from turtle import forward

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

# renderer
import pytorch3d

import torchvision.utils as vutils
import torchvision.transforms as transforms

from torch.hub import load_state_dict_from_url

from models.base_model import BaseModel
from models.networks.bert2vq_net import BERT2VQ
from models.networks.pvqvae_networks.auto_encoder import PVQVAE
import utils
from utils.util import NoamLR
# from utils.util_3d import init_mesh_renderer, render_mesh, init_snet_to_pix3dvox_params, render_sdf, snet_to_pix3dvox

class BERT2VQModel(BaseModel):
    def name(self):
        return 'BERT2VQ-Model'

    def initialize(self, opt):
        BaseModel.initialize(self, opt)
        self.isTrain = opt.isTrain
        self.model_name = self.name()

        # -------------------------------
        # Define Networks
        # -------------------------------

        assert opt.vq_cfg is not None
        
        bert_conf = OmegaConf.load(opt.bert_cfg)
        vq_conf = OmegaConf.load(opt.vq_cfg)
        
        # init resnet2vq network
        self.net = BERT2VQ(opt)
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
            self.criterion_nce = nn.CrossEntropyLoss()
            self.criterion_nce.to(opt.device)

            # ---------------------------------
            # initialize optimizers
            # ---------------------------------
            self.optimizer = optim.AdamW([p for p in self.net.parameters() if p.requires_grad == True], lr=opt.lr)

            self.scheduler = NoamLR(self.optimizer, warmup_steps=bert_conf.hyper_params.warmup_steps)
            
            self.optimizers = [self.optimizer]
            self.schedulers = [self.scheduler]

            self.print_networks(verbose=False)

        
        # resolution = resnet_conf.data.resolution
        # self.resolution = resolution

        # hyper-parameters for SDF
        # if opt.dataset_mode in ['sdf', 'sdf_code', 'sdf_code_v0']:
        #     nC = resolution
        #     assert nC == 64, 'right now, only trained with sdf resolution = 64'
        #     self.down_size = 8   # x: res, x_cube: res//8
        #     self.cube_size = nC // self.down_size    # size per cube. nC=64, down_size=8 -> size=8 for each smaller cube
        #     self.stride = nC // self.down_size
        #     self.ncubes_per_dim = nC // self.cube_size

        # setup renderer
        # dist, elev, azim = 1.7, 20, 20   
        # self.renderer = init_mesh_renderer(image_size=256, dist=dist, elev=elev, azim=azim, device=self.opt.device)

        #
        # self.Rt, self.S, self.vox_thres = init_snet_to_pix3dvox_params()

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

        self.text = input['text']

        self.x_idx_seq = rearrange(self.x_idx, 'bs dz hz wz -> (dz hz wz) bs').contiguous() # to (T, B)
        self.x_idx = self.x_idx_seq.clone()

        vars_list = ['x_idx', 'z_q', 'x']

        self.tocuda(var_names=vars_list)
    
    def inference(self, data, should_render=True, verbose=False):
        pass

    def forward(self):
        self.outp = self.net(self.text)
    
    def backward(self):
        '''backward pass for the Lang to (P)VQ-VAE code model'''
        target = self.x_idx
        target = rearrange(target, '(d h w) bs -> bs d h w', d=8, h=8, w=8)
        outp = self.outp
        
        loss_nll = self.criterion_nce(outp, target)

        self.loss = loss_nll

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
            ('nll', self.loss.data),
        ])

        return ret
    
    def get_current_visuals(self):
                            
        return OrderedDict()
    
    def eval_metrics(self, dataloader, thres=0.0):
                            
        return OrderedDict()

    def save(self, label):

        state_dict = {
            # 'vqvae': self.vqvae.cpu().state_dict(),
            'bert2vq': self.net.cpu().state_dict(),
        }

        save_filename = 'bert2vq_%s.pth' % (label)
        save_path = os.path.join(self.save_dir, save_filename)

        torch.save(state_dict, save_path)
        # self.vqvae.to(self.opt.device)
        self.net.to(self.opt.device)

    def load_ckpt(self, ckpt):
        if type(ckpt) == str:
            state_dict = torch.load(ckpt)
        else:
            state_dict = ckpt

        # self.vqvae.load_state_dict(state_dict['vqvae'])
        self.net.load_state_dict(state_dict['bert2vq'])
        print(colored('[*] weight successfully load from: %s' % ckpt, 'blue'))