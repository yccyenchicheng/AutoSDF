
import numpy as np
from imageio import imread
from PIL import Image

from termcolor import colored, cprint

import torch
import torch.nn as nn
import torch.utils.data as data
import torchvision.transforms as transforms

from torchvision import datasets

from configs.paths import dataroot

class BaseDataset(data.Dataset):
    def __init__(self):
        super(BaseDataset, self).__init__()

    def name(self):
        return 'BaseDataset'

    def initialize(self, opt):
        pass

def CreateDataset(opt):
    dataset = None

    # decide resolution later at model

    if opt.dataset_mode == 'mnist':
        trfs = transforms.Compose([transforms.ToTensor(),
                                   transforms.Normalize((0.5), (0.5))])
        train_dataset = datasets.MNIST(dataroot, download=True, transform=trfs, train=True)
        test_dataset = datasets.MNIST(dataroot, download=True, transform=trfs, train=False)

        train_dataset.name = lambda: 'MNIST'
        test_dataset.name = lambda: 'MNIST'
        
    elif opt.dataset_mode == 'snet':
        from datasets.snet_dataset import ShapeNetDataset
        train_dataset = ShapeNetDataset()
        test_dataset = ShapeNetDataset()
        train_dataset.initialize(opt, 'train', cat=opt.cat)
        test_dataset.initialize(opt, 'test', cat=opt.cat)
        
    elif opt.dataset_mode == 'snet_code':
        from datasets.snet_dataset import ShapeNetCodeDataset
        train_dataset = ShapeNetCodeDataset()
        test_dataset = ShapeNetCodeDataset()
        train_dataset.initialize(opt, 'train', cat=opt.cat)
        test_dataset.initialize(opt, 'test', cat=opt.cat)
        
    elif opt.dataset_mode == 'snet_img':
        from datasets.snet_dataset import ShapeNetImgDataset
        train_dataset = ShapeNetImgDataset()
        test_dataset = ShapeNetImgDataset()
        train_dataset.initialize(opt, 'train', cat=opt.cat)
        test_dataset.initialize(opt, 'test', cat=opt.cat)

    elif opt.dataset_mode == 'pix3d':
        """ pix3d dataset. """
        from datasets.pix3d_dataset import Pix3DDataset
        train_dataset = Pix3DDataset()
        test_dataset = Pix3DDataset()
        train_dataset.initialize(opt, 'train', cat=opt.cat)
        test_dataset.initialize(opt, 'test', cat=opt.cat)

    elif opt.dataset_mode == 'pix3d_code':
        from datasets.pix3d_dataset import Pix3DCodeDataset
        train_dataset = Pix3DCodeDataset()
        test_dataset = Pix3DCodeDataset()
        train_dataset.initialize(opt, 'train', cat=opt.cat)
        test_dataset.initialize(opt, 'test', cat=opt.cat)

    elif opt.dataset_mode == 'pix3d_img':
        from datasets.pix3d_dataset import Pix3DImgDataset
        train_dataset = Pix3DImgDataset()
        test_dataset = Pix3DImgDataset()
        train_dataset.initialize(opt, 'train', cat=opt.cat)
        test_dataset.initialize(opt, 'test', cat=opt.cat)
    
    elif opt.dataset_mode == 'shapenet_lang':
        """ ShapeNet Image dataset. Need to load the code as well """
        from datasets.shapeglot_dataset import ShapeGlotConditionalDataset
        train_dataset = ShapeGlotConditionalDataset()
        test_dataset = ShapeGlotConditionalDataset()
        train_dataset.initialize(opt, 'train', cat=opt.cat)
        test_dataset.initialize(opt, 'test', cat=opt.cat)


    else:
        raise ValueError("Dataset [%s] not recognized." % opt.dataset_mode)

    cprint("[*] Dataset has been created: %s" % (train_dataset.name()), 'blue')
    return train_dataset, test_dataset
