import os
import argparse

from termcolor import colored
from omegaconf import OmegaConf

import torch
from torch.utils.tensorboard import SummaryWriter

import utils

class BaseOptions():
    def __init__(self):
        self.parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        self.initialized = False

    def initialize(self):
        # hyper parameters
        self.parser.add_argument('--batch_size', type=int, default=1, help='input batch size')
        self.parser.add_argument('--gpu_ids', type=str, default='1', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')

        # log stuff
        self.parser.add_argument('--logs_dir', type=str, default='./logs', help='models are saved here')
        self.parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints', help='models are saved here')
        self.parser.add_argument('--name', type=str, default='experiment_name', help='name of the experiment. It decides where to store samples and models')

        # model stuff
        self.parser.add_argument('--model', type=str, default='pvqvae', choices=['pvqvae', 'rand_tf', 'resnet2vq', 'bert2vq'], help='chooses which model to use.')
        self.parser.add_argument('--ckpt', type=str, default=None, help='ckpt to load.')
        self.parser.add_argument('--input_nc', type=int, default=3, help='# of input image channels')
        self.parser.add_argument('--output_nc', type=int, default=3, help='# of output image channels')
        self.parser.add_argument('--ngf', type=int, default=64, help='# of gen filters in first conv layer')
        self.parser.add_argument('--ndf', type=int, default=64, help='# of discrim filters in first conv layer')
        self.parser.add_argument('--use_bin_sdf', type=str, default='0', help='use binarized sdf for training vox model or not')
        
        # vqvae stuff; need to load the corresponding vqvae weight for training transformer
        self.parser.add_argument('--vq_model', type=str, default='vqvae', choices=['vqvae', 'pvqvae'], help='vqvae model to use.')
        self.parser.add_argument('--vq_cfg', type=str, default='configs/pvqvae_snet.yaml', help='vqvae model config file')
        self.parser.add_argument('--vq_dset', type=str, default=None, help='dataset vqvae originally trained on')
        self.parser.add_argument('--vq_cat', type=str, default=None, help='for setting code dir in XXXCodeDataset.')
        self.parser.add_argument('--vq_ckpt', type=str, default=None, help='vq ckpt to load.')
        self.parser.add_argument('--vq_note', type=str, default='default', help='for different setting of p-vqvae. used in extract_code.py')
        
        # transformer stuff
        self.parser.add_argument('--tf_cfg', type=str, default='configs/rand_tf_snet_code.yaml', help='tf model config file')

        # dataset stuff
        self.parser.add_argument('--dataset_mode', type=str, default='snet', help='chooses how datasets are loaded. [mnist, snet, abc, snet-abc]')
        self.parser.add_argument('--trunc_thres', type=float, default=0.2, help='threshold for truncated sdf. value will be: sdf=torch.clamp(sdf, -trunc_thres, trunc_thres)')
        self.parser.add_argument('--iou_thres', type=float, default=0.0, help='threshold for computing 3d iou.')
        self.parser.add_argument('--ratio', type=float, default=1., help='ratio of the dataset to use')
        
        self.parser.add_argument('--cat', type=str, default='chair', help='category for shapenet')
        self.parser.add_argument('--max_dataset_size', default=2147483648, type=int, help='chooses how datasets are loaded. [mnist, sdf, abc, snet-abc]')
        self.parser.add_argument('--nThreads', default=9, type=int, help='# threads for loading data')        
        self.parser.add_argument('--serial_batches', action='store_true', help='if true, takes images in order to make batches, otherwise takes them randomly')
        
        # pix3d stuff
        self.parser.add_argument('--pix3d_mode', type=str, default='noBG')
        self.parser.add_argument('--snet_mode', type=str, default='noBG')

        # single-view recon. stuff.
        self.parser.add_argument('--use_marginal', type=str, default='0')
        self.parser.add_argument('--resnet2vq_ckpt', type=str, default=None)

        # single-view recon.: DEPRECATED
        self.parser.add_argument('--resnet_model', type=str, default=None)
        self.parser.add_argument('--resnet_cfg', type=str, default='configs/resnet2vq_pix3d.yaml', help='resnet2XX model config file')
        self.parser.add_argument('--resnet_ckpt', type=str, default=None)
        self.parser.add_argument('--resnet_arch', type=str, default='resnet18')
        self.parser.add_argument('--resnet_norm', type=str, default='gn', choices=['bn', 'gn'])
        self.parser.add_argument('--resnet_dset', type=str, help='resnet is trained on which dset')

        # shapeglot lang specific arguments
        self.parser.add_argument('--bert_cfg', type=str, default='configs/bert2vq_shapeglot.yaml', help='bert2VQ model config file')

        # for qual
        self.parser.add_argument('--n_less', type=int, default=0, help='for less context')
        self.parser.add_argument('--alpha', type=float, default=0.75, help='for less context')
        self.parser.add_argument('--topk', type=int, default=30, help='for less context')

        # misc
        self.parser.add_argument('--debug', default='0', type=str, choices=['0', '1'], help='if true, debug mode')
        self.parser.add_argument('--seed', default=111, type=int, help='seed')
        self.parser.add_argument('--profiler', default='0', type=str, choices=['0', '1'], help='use profiler or not')

        self.initialized = True

    def parse(self):
        if not self.initialized:
            self.initialize()
        self.opt = self.parser.parse_args()
        self.opt.isTrain = self.isTrain   # train or test

        # added by yc
        self.opt.device = 'cuda'
        
        if self.opt.model in ['vqvae', 'pvqvae']:
            configs = OmegaConf.load(self.opt.vq_cfg)
            mparam = configs.model.params
            lparam = configs.lossconfig.params
            ddconfig = mparam.ddconfig
            
            zdim = ddconfig.z_channels
            # update name with model confige file
            name = '%s-k%s-d%s-ch%s-z%s-codeW%s-lpipsW%s' % (
                self.opt.name,
                mparam.n_embed, mparam.embed_dim,
                ddconfig.ch, ddconfig.z_channels,
                lparam.codebook_weight, lparam.perceptual_weight,
            )
        elif self.opt.model in ['rand_tf']:
            configs = OmegaConf.load(self.opt.tf_cfg)
        
        # if is in training phase, make the logs_dir more verbose about hyper-parameters
        if self.opt.isTrain:
            if 'transformer' in self.opt.model:
                tf_arch = configs.model.arch
                pe_conf = configs.pe
                
                ntokens_tf = configs.model.params.ntokens
                d_tf = configs.model.params.embed_dim
                d_hid = configs.model.params.d_hid
                
                nhead = configs.model.params.nhead
                nlayers_enc = configs.model.params.nlayers_enc
                nlayers_dec = configs.model.params.nlayers_dec
                
                name = '%s-arch-%s-k%s-tfDim%s-hidDim%s-nH%s-nEnc%s-nDec%s-posD%s-posInit%s' % (
                    self.opt.name, tf_arch, ntokens_tf, d_tf, d_hid,
                    nhead,  nlayers_enc, nlayers_dec,
                    pe_conf.pos_embed_dim, pe_conf.init_factor)
            else:
                name = self.opt.name
        else:
            name = self.opt.name

        self.opt.name = name
        
        self.opt.gpu_ids_str = self.opt.gpu_ids

        str_ids = self.opt.gpu_ids.split(',')
        self.opt.gpu_ids = []
        for str_id in str_ids:
            id = int(str_id)
            if id >= 0:
                self.opt.gpu_ids.append(id)

        # set gpu ids
        if len(self.opt.gpu_ids) > 0:
            torch.cuda.set_device(self.opt.gpu_ids[0])
            
        args = vars(self.opt)

        print('------------ Options -------------')
        for k, v in sorted(args.items()):
            print('%s: %s' % (str(k), str(v)))
        print('-------------- End ----------------')

        # save to the disk
        if self.opt.isTrain:
            expr_dir = os.path.join(self.opt.logs_dir, self.opt.name)

            utils.util.mkdirs(expr_dir)
            file_name = os.path.join(expr_dir, 'opt.txt')
            with open(file_name, 'wt') as opt_file:
                opt_file.write('------------ Options -------------\n')
                for k, v in sorted(args.items()):
                    opt_file.write('%s: %s\n' % (str(k), str(v)))
                opt_file.write('-------------- End ----------------\n')
            
            # tensorboard writer
            tb_dir = '%s/tboard' % expr_dir
            if not os.path.exists(tb_dir):
                os.makedirs(tb_dir)
            self.opt.tb_dir = tb_dir
            writer = SummaryWriter(log_dir=tb_dir)
            self.opt.writer = writer

        return self.opt
