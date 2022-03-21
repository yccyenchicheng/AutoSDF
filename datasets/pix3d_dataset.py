"""
    adopted from: https://github.com/shubhtuls/PixelTransformer/blob/03b65b8612fe583b3e35fc82b446b5503dd7b6bd/data/shapenet.py
"""

import os
import glob
import json
import socket

import scipy.io
import h5py
import numpy as np
from PIL import Image
from termcolor import colored, cprint

import torch
import torch.nn.functional as F
import torchvision.utils as vutils
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from torchvision.transforms.functional import InterpolationMode

from datasets.base_dataset import BaseDataset

from configs.paths import dataroot
from preprocess.create_sdf_file_lst import json_f_dict
from utils.pix3d_util import downsample_voxel

hostname = socket.gethostname()


def get_code_setting(opt):
    code_setting = f'{opt.vq_model}-{opt.vq_dset}-{opt.vq_cat}-T{opt.trunc_thres}'
    if opt.vq_note != 'default':
        code_setting = f'{code_setting}-{opt.vq_note}'
    return code_setting

# self.img_list, self.model_list, self.cats_list, self.gt_voxel_list = self.load_custom_splits_by_imgs(cat, phase)
class Pix3DDataset(BaseDataset):

    # def initialize(self, opt, phase='train', cat='chair'):
    def initialize(self, opt, phase='train', cat='chair'):
        self.opt = opt
        self.max_dataset_size = opt.max_dataset_size

        self.sdf_list, self.cats_list = load_custom_splits(self, cat, phase, opt)

        np.random.default_rng(seed=0).shuffle(self.sdf_list)
        np.random.default_rng(seed=0).shuffle(self.cats_list)

        # need to check the seed for reproducibility
        self.sdf_list = self.sdf_list[:self.max_dataset_size]
        cprint('[*] %d samples loaded.' % (len(self.sdf_list)), 'yellow')
        
        self.N = len(self.sdf_list)

        self.to_tensor = transforms.ToTensor()
        self.normalize = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        self.resize = transforms.Resize((256, 256))

    def __getitem__(self, index):

        cat_name = self.cats_list[index]
        sdf_h5_file = self.sdf_list[index]
        
        h5_f = h5py.File(sdf_h5_file, 'r')
        sdf = h5_f['pc_sdf_sample'][:].astype(np.float32)
        sdf = torch.Tensor(sdf).view(1, 64, 64, 64)

        thres = self.opt.trunc_thres
        if thres != 0.0:
            sdf = torch.clamp(sdf, min=-thres, max=thres)

        ret = {
            'sdf': sdf,
            # 'cat_id': synset,
            'cat_str': cat_name,
            'path': sdf_h5_file,
            # 'tsdf': tsdf,
        }

        return ret

    def __len__(self):
        return self.N

    def name(self):
        return 'Pix3DDataset'

class Pix3DCodeDataset(BaseDataset):

    # def initialize(self, opt, phase='train', cat='chair'):
    def initialize(self, opt, phase='train', cat='all'):
        self.opt = opt
        self.max_dataset_size = opt.max_dataset_size

        info_file = json_f_dict[hostname]['pix3d']
        info_path = f'preprocess/info_files/{info_file}'

        with open(info_path) as f:
            self.info = json.load(f)

        if cat == 'all':
            cats = self.info['all_cats']
        else:
            cats = [cat]

        code_setting = f'{opt.vq_model}-{opt.vq_dset}-{opt.vq_cat}-T{opt.trunc_thres}'
        self.code_dir = f'{dataroot}/extracted_code/{code_setting}'
        assert os.path.exists(self.code_dir), f'{self.code_dir} should exist.'
        
        if cat == 'all':
            cats = self.info['all_cats']
        else:
            cats = [cat]

        self.model_list = []
        self.cats_list = []
        for c in cats:
            # if c == 'chair':
            #     continue
            with open(f'{dataroot}/pix3d/filelists/{c}_{phase}.lst') as f:
                model_list_s = []
                for l in f.readlines():
                    model_id = l.rstrip('\n')

                    # path = f'{dataroot}/pix3d/SDF_v1_64/{c}/{model_id}/ori_sample_grid.h5'
                    path = f'{self.code_root}/{c}/{model_id}'
                    model_list_s.append(path)

                self.model_list += model_list_s
                self.cats_list += [c] * len(model_list_s)
                print('[*] %d samples for %s.' % (len(model_list_s), c))

        np.random.default_rng(seed=0).shuffle(self.model_list)
        np.random.default_rng(seed=0).shuffle(self.cats_list)

        cprint('[*] (Pix3DDataset) there are %d categories.' % (len(cats)), 'yellow')

        # need to check the seed for reproducibility
        self.model_list = self.model_list[:self.max_dataset_size]
        cprint('[*] %d code loaded.' % (len(self.model_list)), 'yellow')

        self.N = len(self.model_list)

        self.to_tensor = transforms.ToTensor()
        self.normalize = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))

    def __getitem__(self, index):

        catname = self.cats_list[index]
        model = self.model_list[index]

        sdf_p = f'{model}/x.npy'
        code_p = f'{model}/code.npy'
        codeix_p = f'{model}/codeix.npy'

        sdf = torch.from_numpy(np.load(sdf_p))
        code = torch.from_numpy(np.load(code_p))
        codeix = torch.from_numpy(np.load(codeix_p))

        ret = {
            'sdf': sdf,
            'z_q': code,
            'idx': codeix,
            'cat_str': catname,
            'path': model,
        }

        return ret

    def __len__(self):
        return self.N

    def name(self):
        return 'Pix3DCodeDataset'

####################################################################################################

# from https://github.com/laughtervv/DISN/blob/master/preprocessing/info.json

class Pix3DImgDataset(BaseDataset):
    # def initialize(self, opt, phase='train', cat='chair'):
    def initialize(self, opt, phase='train', cat='chair', input_txt=None, by_imgs=True):
        self.opt = opt
        self.max_dataset_size = opt.max_dataset_size
        self.phase = phase
        self.ratio = opt.ratio

        self.pix3d_mode = opt.pix3d_mode

        # with open(f'{dataroot}/ShapeNet/info.json') as f:
        #     self.info = json.load(f)
        info_file = json_f_dict[hostname]['pix3d']
        info_path = f'preprocess/info_files/{info_file}'

        with open(info_path) as f:
            self.info = json.load(f)

        code_setting = f'{opt.vq_model}-{opt.vq_dset}-{opt.vq_cat}-T{opt.trunc_thres}'
        self.code_dir = f'{dataroot}/extracted_code/{code_setting}'
        assert os.path.exists(self.code_dir), f'{self.code_dir} should exist.'
        
        self.img_list, self.model_list, self.cats_list, self.gt_voxel_list = \
            load_custom_splits_for_img_dset(self, cat, phase, opt)

        np.random.default_rng(seed=0).shuffle(self.img_list)
        np.random.default_rng(seed=0).shuffle(self.model_list)
        np.random.default_rng(seed=0).shuffle(self.cats_list)
        np.random.default_rng(seed=0).shuffle(self.gt_voxel_list)

        # need to check the seed for reproducibility
        self.img_list = self.img_list[:self.max_dataset_size]
        self.model_list = self.model_list[:self.max_dataset_size]
        self.cats_list = self.cats_list[:self.max_dataset_size]
        self.gt_voxel_list = self.gt_voxel_list[:self.max_dataset_size]
        cprint('[*] %d img_list loaded.' % (len(self.img_list)), 'yellow')
        cprint('[*] %d code loaded.' % (len(self.model_list)), 'yellow')
        cprint(f'[*] ratio: {self.ratio}')
        
        self.N = len(self.img_list)
        self.to_tensor = transforms.ToTensor()

        mean, std = [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]
        # mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
        if phase == 'train':
            self.transforms = transforms.Compose([
                transforms.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.25),
                transforms.RandomAffine(0, scale=(0.7, 1.25), interpolation=InterpolationMode.BILINEAR),
                transforms.Normalize(mean, std),
                transforms.RandomHorizontalFlip(),
                transforms.Resize((256, 256)),
            ])
        else:
            self.transforms = transforms.Compose([
                transforms.Normalize(mean, std),
                transforms.Resize((256, 256)),
            ])

        self.n_view = 1
    
    def process_img(self, img):
        img_t = self.to_tensor(img)

        # noBG is trained with this augmentation
        if self.opt.pix3d_mode == 'noBG':
            _, oh, ow = img_t.shape

            ls = max(oh, ow)

            pad_h1, pad_h2 = (ls - oh) // 2, (ls - oh) - (ls - oh) // 2
            pad_w1, pad_w2 = (ls - ow) // 2, (ls - ow) - (ls - ow) // 2
            img_t = F.pad(img_t[None, ...], (pad_w1, pad_w2, pad_h1, pad_h2), mode='constant', value=0)

            img_t = self.transforms(img_t[0])
        else:
            img_t = self.transforms(img_t)

        return img_t

    def read_vox(self, f):
        gt_size = 32

        voxel_p = f
        # read voxel
        voxel = scipy.io.loadmat(voxel_p)['voxel']

        # downsample
        voxel = downsample_voxel(voxel, 0.5, (gt_size, gt_size, gt_size))
        voxel = torch.from_numpy(voxel)
        voxel = voxel.float()
        return voxel

    def __getitem__(self, index):
        
        cat_name = self.cats_list[index]
        code_x_path = self.model_list[index] # XXX/x.npy
        code_dir = os.path.dirname(code_x_path)
        x_name = os.path.basename(code_x_path)
        code_name = x_name.replace('x', 'code')
        codeix_name = x_name.replace('x', 'codeix')

        sdf_p = code_x_path
        code_p = f'{code_dir}/{code_name}'
        codeix_p = f'{code_dir}/{codeix_name}'

        sdf = torch.from_numpy(np.load(sdf_p))
        code = torch.from_numpy(np.load(code_p))
        codeix = torch.from_numpy(np.load(codeix_p))

        # load img; randomly sample 1
        imgs = []
        img_paths = []
        imgs_all_view = self.img_list[index]
        sample_ixs = np.random.choice(len(imgs_all_view), self.n_view)
        for ix in sample_ixs:
            p = imgs_all_view[ix]
            im = Image.open(p).convert('RGB')
            im = self.process_img(im)
            imgs.append(im)
            img_paths.append(p)

        imgs = torch.stack(imgs)
        img = imgs[0]
        img_path = img_paths[0]

        gt_vox_path = self.gt_voxel_list[index]
        gt_vox = self.read_vox(gt_vox_path) # already downsample

        ret = {
            'sdf': sdf, 'z_q': code,'idx': codeix, 'path': code_x_path,
            'img': img, 'img_path': img_path, 'imgs': imgs, 'img_paths': img_paths,
            'gt_vox': gt_vox, 'gt_vox_path': gt_vox_path,
            'cat_str': cat_name,
        }
        return ret

    def __len__(self):
        return self.N

    def name(self):
        return 'Pix3DImageDataset'


####################################################################################################

def load_pix3d_info(opt):

    with open(f'{dataroot}/pix3d/pix3d.json', 'r') as f:
        pix3d_info = json.load(f)

    # map 
    map_input_to_pix3dinfo = {}
    map_obj_to_imgs = {}

    for d in pix3d_info:
        cat = d['category']
        img_name = d['img']
        obj_file = d['model']
        model_id = obj_file.split('/')[-2]
        # img_name = os.path.splitext(img_name)[0]
        map_input_to_pix3dinfo[img_name] = d

        img_basename = os.path.basename(img_name)
        if opt.pix3d_mode == 'hasBG':
            our_img_name = f"{dataroot}/pix3d/imgs_by_model/{cat}/{model_id}/{img_basename}"
        else:
            our_img_name = f"{dataroot}/pix3d/imgs_by_model_noBG/{cat}/{model_id}/{img_basename}"

        if obj_file not in map_obj_to_imgs:
            map_obj_to_imgs[obj_file] = [our_img_name]
        else:
            map_obj_to_imgs[obj_file].append(our_img_name)

    return pix3d_info, map_input_to_pix3dinfo, map_obj_to_imgs

def load_custom_splits(dset_instance, cat, phase, opt): 
    pix3d_info, map_input_to_pix3dinfo, map_obj_to_imgs = load_pix3d_info(opt)
    
    # get dataset's info dict (different from pix3d's info file from official info)
    info_file = json_f_dict[hostname]['pix3d']
    info_path = f'preprocess/info_files/{info_file}'
    with open(info_path) as f:
        dset_instance.info = json.load(f)
    
    # check chair "test images" from official split
    with open(f'{dataroot}/pix3d/input.txt', 'r') as f:
        lines = [l.rstrip('\n') for l in f.readlines()]
        official_test_imgs = [l.replace('../', '') for l in lines]

    if cat == 'all':
        cats = dset_instance.info['all_cats']
    else:
        cats = [cat]

    img_list = []
    sdf_list = []
    cats_list = []

    for c in cats:
        
        if opt.pix3d_mode == 'hasBG':
            lst_f = f'{dataroot}/pix3d/filelists/{c}_{phase}.lst'
        else:
            lst_f = f'{dataroot}/pix3d/filelists_noBG/{c}_{phase}.lst'

        with open(lst_f) as f:
            lines = f.readlines()
            lines = [l.rstrip('\n') for l in lines]

        img_list_s = []
        sdf_list_s = []

        all_cat_imgs = glob.glob(f'{dataroot}/pix3d/img/{c}/*')

        for i, img_path in enumerate(all_cat_imgs):
            img_basename = os.path.basename(img_path)
            img_name_as_key = f'img/{c}/{img_basename}'
            info_i = map_input_to_pix3dinfo[img_name_as_key]
            
            obj_f = info_i['model']
            model_id = obj_f.split('/')[-2]
            
            # get our img_path
            if opt.pix3d_mode == 'hasBG':
                our_img_name = f"{dataroot}/pix3d/imgs_by_model/{c}/{model_id}/{img_basename}"
            else:
                our_img_name = f"{dataroot}/pix3d/imgs_by_model_noBG/{c}/{model_id}/{img_basename}"

            # check whether in lines or not
            is_in_split = False
            if c == 'chair':
                if our_img_name.replace(dataroot, '') in lines:
                    is_in_split = True
            else:
                if model_id in lines:
                    is_in_split = True
            
            if is_in_split:
                img_list_s.append([our_img_name])

                obj_name = os.path.basename(obj_f)
                obj_name = obj_name.replace('.obj', '').replace('model', '')
                
                if obj_name != '':
                    sdf_name_noext = f'ori_sample_grid{obj_name}'
                else:
                    sdf_name_noext = f'ori_sample_grid'

                # find code
                sdf_path = f'{dataroot}/pix3d/SDF_v1_64/{c}/{model_id}/{sdf_name_noext}.h5'
                # if not os.path.exists(sdf_path):
                    # import pdb; pdb.set_trace()
                sdf_list_s.append(sdf_path)

        # sanity check
        if c == 'chair':
            all_chair_imgs = []
            for s in img_list_s:
                all_chair_imgs += s
                
        nimgs_img_list_s = len(img_list_s)
        nimgs_to_take = int(nimgs_img_list_s * opt.ratio)

        img_list += img_list_s[:nimgs_to_take]
        sdf_list += sdf_list_s[:nimgs_to_take]
        cats_list += [c] * len(img_list_s[:nimgs_to_take])

        assert len(img_list) == len(sdf_list) == len(cats_list)

        print('[*] %d samples for %s.' % (len(sdf_list_s), c))

    cprint('[*] (Pix3DDataset) there are %d categories.' % (len(cats)), 'yellow')

    """ sanity check """
    with open(f'{dataroot}/pix3d/input.txt', 'r') as f:
        official_test_imgs = [l.rstrip('\n') for l in f.readlines()]

    bnames1 = [os.path.basename(f) for f in official_test_imgs]
    bnames2 = [os.path.basename(f) for f in all_chair_imgs]

    if phase == 'test':
        assert set(bnames1).intersection(set(bnames2)) == set(bnames1)
    else:
        assert len(set(bnames1).intersection(set(bnames2))) == 0

    return sdf_list, cats_list


def load_custom_splits_for_img_dset(dset_instance, cat, phase, opt): 
    pix3d_info, map_input_to_pix3dinfo, map_obj_to_imgs = load_pix3d_info(opt)
    
    # get dataset's info dict (different from pix3d's info file from official info)
    info_file = json_f_dict[hostname]['pix3d']
    info_path = f'preprocess/info_files/{info_file}'
    with open(info_path) as f:
        dset_instance.info = json.load(f)
    
    # check chair "test images" from official split
    with open(f'{dataroot}/pix3d/input.txt', 'r') as f:
        lines = [l.rstrip('\n') for l in f.readlines()]
        official_test_imgs = [l.replace('../', '') for l in lines]

    if cat == 'all':
        cats = dset_instance.info['all_cats']
    else:
        cats = [cat]

    img_list = []
    model_list = []
    cats_list = []
    gt_voxel_list = []

    for c in cats:
        
        if opt.pix3d_mode == 'hasBG':
            lst_f = f'{dataroot}/pix3d/filelists/{c}_{phase}.lst'
        else:
            lst_f = f'{dataroot}/pix3d/filelists_noBG/{c}_{phase}.lst'

        with open(lst_f) as f:
            lines = f.readlines()
            lines = [l.rstrip('\n') for l in lines]

        img_list_s = []
        model_list_s = []
        gt_voxel_list_s = []

        all_cat_imgs = glob.glob(f'{dataroot}/pix3d/img/{c}/*')

        for i, img_path in enumerate(all_cat_imgs):
            img_basename = os.path.basename(img_path)
            img_name_as_key = f'img/{c}/{img_basename}'
            info_i = map_input_to_pix3dinfo[img_name_as_key]

            obj_f = info_i['model']
            model_id = obj_f.split('/')[-2]
            
            # get our img_path
            if opt.pix3d_mode == 'hasBG':
                our_img_name = f"{dataroot}/pix3d/imgs_by_model/{c}/{model_id}/{img_basename}"
            else:
                our_img_name = f"{dataroot}/pix3d/imgs_by_model_noBG/{c}/{model_id}/{img_basename}"

            # check whether in lines or not
            is_in_split = False
            if c == 'chair':
                if our_img_name.replace(dataroot, '') in lines:
                    is_in_split = True
            else:
                if model_id in lines:
                    is_in_split = True
            
            if is_in_split:
                img_list_s.append([our_img_name])

                obj_name = os.path.basename(obj_f)
                obj_name = obj_name.replace('.obj', '')

                # find code
                code_x_name = obj_name.replace('model', 'x')
                code_x_path = f'{dset_instance.code_dir}/{c}/{model_id}/{code_x_name}.npy'
                model_list_s.append(code_x_path)

                # find gt voxel
                gt_vox_name = obj_name.replace('model', 'voxel')
                gt_vox_path = f'{dataroot}/pix3d/model/{c}/{model_id}/{gt_vox_name}.mat'
                gt_voxel_list_s.append(gt_vox_path)


        # sanity check
        if c == 'chair':
            all_chair_imgs = []
            for s in img_list_s:
                all_chair_imgs += s

        nimgs_img_list_s = len(img_list_s)
        nimgs_to_take = int(nimgs_img_list_s * opt.ratio)

        img_list += img_list_s[:nimgs_to_take]
        model_list += model_list_s[:nimgs_to_take]
        gt_voxel_list += gt_voxel_list_s[:nimgs_to_take]
        cats_list += [c] * len(img_list_s[:nimgs_to_take])

        try:
            assert len(img_list) == len(model_list) == len(gt_voxel_list) == len(cats_list)
        except:
            import pdb; pdb.set_trace()

        print('[*] %d samples for %s.' % (len(model_list_s), c))

    cprint('[*] (Pix3DDataset) there are %d categories.' % (len(cats)), 'yellow')

    """ sanity check """
    with open(f'{dataroot}/pix3d/input.txt', 'r') as f:
        official_test_imgs = [l.rstrip('\n') for l in f.readlines()]

    bnames1 = [os.path.basename(f) for f in official_test_imgs]
    bnames2 = [os.path.basename(f) for f in all_chair_imgs]

    if phase == 'test':
        assert set(bnames1).intersection(set(bnames2)) == set(bnames1)
    else:
        assert len(set(bnames1).intersection(set(bnames2))) == 0

    return img_list, model_list, cats_list, gt_voxel_list