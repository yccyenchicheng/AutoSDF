import os
from collections import OrderedDict

import numpy as np
import einops
import marching_cubes as mcubes
import omegaconf
from termcolor import colored
from einops import rearrange
from tqdm import tqdm

import torch
from torch import nn, optim
from torch.profiler import record_function

import torchvision.utils as vutils
import torchvision.transforms as transforms

from models.base_model import BaseModel
from models.networks.pvqvae_networks.auto_encoder import PVQVAE
from models.networks.losses import VQLoss

import utils.util
from utils.util_3d import init_mesh_renderer, render_sdf

class PVQVAEModel(BaseModel):
    def name(self):
        return 'PVQVAE-Model'

    def initialize(self, opt):
        BaseModel.initialize(self, opt)
        self.isTrain = opt.isTrain
        self.model_name = self.name()

        # -------------------------------
        # Define Networks
        # -------------------------------

        # model
        assert opt.vq_cfg is not None
        configs = omegaconf.OmegaConf.load(opt.vq_cfg)
        mparam = configs.model.params
        n_embed = mparam.n_embed
        embed_dim = mparam.embed_dim
        ddconfig = mparam.ddconfig

        n_down = len(ddconfig.ch_mult) - 1

        self.vqvae = PVQVAE(ddconfig, n_embed, embed_dim)
        self.vqvae.to(opt.device)

        if self.isTrain:
            # ----------------------------------
            # define loss functions
            # ----------------------------------
            lossconfig = configs.lossconfig
            lossparams = lossconfig.params
            self.loss_vq = VQLoss(**lossparams).to(opt.device)

            # ---------------------------------
            # initialize optimizers
            # ---------------------------------
            self.optimizer = optim.Adam(self.vqvae.parameters(), lr=opt.lr, betas=(0.5, 0.9))
            self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, 10 if opt.dataset_mode == 'imagenet' else 30, 0.5,)
            # self.scheduler = optim.lr_scheduler.StepLR(self.optimizer)

            self.optimizers = [self.optimizer]
            self.schedulers = [self.scheduler]

            self.print_networks(verbose=False)

        resolution = configs.model.params.ddconfig['resolution']
        self.resolution = resolution

        # setup hyper-params 
        nC = resolution
        self.cube_size = 2 ** n_down # patch_size
        self.stride = self.cube_size
        self.ncubes_per_dim = nC // self.cube_size
        assert nC == 64, 'right now, only trained with sdf resolution = 64'
        assert (nC % self.cube_size) == 0, 'nC should be divisable by cube_size'

        # setup renderer
        dist, elev, azim = 1.7, 20, 20   
        self.renderer = init_mesh_renderer(image_size=256, dist=dist, elev=elev, azim=azim, device=self.opt.device)

        # for saving best ckpt
        self.best_iou = -1e12
        
    @staticmethod
    # def unfold_to_cubes(self, x, cube_size=8, stride=8):
    def unfold_to_cubes(x, cube_size=8, stride=8):
        """ 
            assume x.shape: b, c, d, h, w 
            return: x_cubes: (b cubes)
        """
        x_cubes = x.unfold(2, cube_size, stride).unfold(3, cube_size, stride).unfold(4, cube_size, stride)
        x_cubes = rearrange(x_cubes, 'b c p1 p2 p3 d h w -> b c (p1 p2 p3) d h w')
        x_cubes = rearrange(x_cubes, 'b c p d h w -> (b p) c d h w')

        return x_cubes

    @staticmethod
    # def fold_to_voxels(self, x_cubes, batch_size, ncubes_per_dim):
    def fold_to_voxels(x_cubes, batch_size, ncubes_per_dim):
        x = rearrange(x_cubes, '(b p) c d h w -> b p c d h w', b=batch_size) 
        x = rearrange(x, 'b (p1 p2 p3) c d h w -> b c (p1 d) (p2 h) (p3 w)',
                        p1=ncubes_per_dim, p2=ncubes_per_dim, p3=ncubes_per_dim)
        return x

    def set_input(self, input):
        '''Samples at training time'''
        # import pdb; pdb.set_trace()
        x = input['sdf']
        self.x = x
        self.cur_bs = x.shape[0] # to handle last batch

        self.x_cubes = self.unfold_to_cubes(x, self.cube_size, self.stride)
        vars_list = ['x', 'x_cubes']

        self.tocuda(var_names=vars_list)

    def forward(self):
        # qloss: codebook loss
        self.zq_cubes, self.qloss, _ = self.vqvae.encode(self.x_cubes) # zq_cubes: ncubes X zdim X 1 X 1 X 1
        self.zq_voxels = self.fold_to_voxels(self.zq_cubes, batch_size=self.cur_bs, ncubes_per_dim=self.ncubes_per_dim) # zq_voxels: bs X zdim X ncubes_per_dim X ncubes_per_dim X ncubes_per_dim
        self.x_recon = self.vqvae.decode(self.zq_voxels)

    def inference(self, data, should_render=False, verbose=False):
        self.vqvae.eval()
        self.set_input(data)

        # make sure it has the same name as forward
        with torch.no_grad():
            self.zq_cubes, _, self.info = self.vqvae.encode(self.x_cubes)
            self.zq_voxels = self.fold_to_voxels(self.zq_cubes, batch_size=self.cur_bs, ncubes_per_dim=self.ncubes_per_dim)
            self.x_recon = self.vqvae.decode(self.zq_voxels)
            # _, _, quant_ix = info
            # 

            if should_render:
                self.image = render_sdf(self.renderer, self.x)
                self.image_recon = render_sdf(self.renderer, self.x_recon)

        self.vqvae.train()

    def test_iou(self, data, thres=0.0):
        """
            thres: threshold to consider a voxel to be free space or occupied space.
        """
        # self.set_input(data)

        self.vqvae.eval()
        self.inference(data, should_render=False)
        self.vqvae.train()

        x = self.x
        x_recon = self.x_recon

        iou = utils.util.iou(x, x_recon, thres)

        return iou

    def eval_metrics(self, dataloader, thres=0.0):
        self.eval()

        iou_list = []
        with torch.no_grad():
            for ix, test_data in tqdm(enumerate(dataloader), total=len(dataloader)):

                iou = self.test_iou(test_data, thres=thres)
                iou_list.append(iou.detach())

        iou = torch.cat(iou_list)
        iou_mean, iou_std = iou.mean(), iou.std()
        
        ret = OrderedDict([
            ('iou', iou_mean.data),
            ('iou_std', iou_std.data),
        ])

        # check whether to save best epoch
        if ret['iou'] > self.best_iou:
            self.best_iou = ret['iou']
            save_name = f'epoch-best'
            self.save(save_name)

        self.train()
        return ret


    def backward(self):
        '''backward pass for the generator in training the unsupervised model'''
        aeloss, log_dict_ae = self.loss_vq(self.qloss, self.x, self.x_recon)

        self.loss = aeloss

        self.loss_codebook = log_dict_ae['loss_codebook']
        self.loss_nll = log_dict_ae['loss_nll']
        self.loss_rec = log_dict_ae['loss_rec']
        self.loss_p = log_dict_ae['loss_p']

        self.loss.backward()

    def optimize_parameters(self, total_steps):

        self.forward()
        self.optimizer.zero_grad(set_to_none=True)
        self.backward()
        self.optimizer.step()

    def get_logs_data(self):
        """ return a dictionary with
            key: graph name
            value: an OrderedDict with the data to plot
        
        """
        raise NotImplementedError
        return ret
    
    def get_current_errors(self):
        
        ret = OrderedDict([
            ('codebook', self.loss_codebook.data),
            ('nll', self.loss_nll.data),
            ('rec', self.loss_rec.data),
            ('p', self.loss_p.data),
        ])

        return ret

    def get_current_visuals(self):

        with torch.no_grad():
            self.image = render_sdf(self.renderer, self.x)
            self.image_recon = render_sdf(self.renderer, self.x_recon)

        vis_tensor_names = [
            'image',
            'image_recon',
        ]

        vis_ims = self.tnsrs2ims(vis_tensor_names)
        # vis_tensor_names = ['%s/%s' % (phase, n) for n in vis_tensor_names]
        visuals = zip(vis_tensor_names, vis_ims)
                            
        return OrderedDict(visuals)

    def save(self, label):
        save_filename = 'vqvae_%s.pth' % (label)
        save_path = os.path.join(self.save_dir, save_filename)
        if len(self.gpu_ids) and torch.cuda.is_available():
            torch.save(self.vqvae.cpu().state_dict(), save_path)
            self.vqvae.cuda(self.gpu_ids[0])
        else:
            torch.save(self.vqvae.cpu().state_dict(), save_path)

    def get_codebook_weight(self):
        ret = self.vqvae.quantize.embedding.cpu().state_dict()
        self.vqvae.quantize.embedding.cuda()
        return ret

    def load_ckpt(self, ckpt):
        if type(ckpt) == str:
            state_dict = torch.load(ckpt)
        else:
            state_dict = ckpt

        self.vqvae.load_state_dict(state_dict)
        print(colored('[*] weight successfully load from: %s' % ckpt, 'blue'))


