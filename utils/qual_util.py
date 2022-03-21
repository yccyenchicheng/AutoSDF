import cv2
import numpy as np
import imageio
from PIL import Image
from einops import rearrange
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.utils as vutils
import torchvision.transforms as transforms

from pytorch3d import structures

from .util_3d import sdf_to_mesh, render_mesh, rotate_mesh_360


def make_batch(data, B=16):
    x = data['sdf']
    x_idx = data['idx']
    z_q = data['z_q']
    bs = x.shape[1]
    if bs > B:
        return data

    data['sdf'] = x.repeat(B//bs, 1, 1, 1, 1)
    data['idx'] = x_idx.repeat(B//bs, 1, 1, 1)
    data['z_q'] = z_q.repeat(B//bs, 1, 1, 1, 1)
    return data

def get_partial_shape_by_range(sdf, input_range, thres=0.2):
    sdf = torch.clamp(sdf, min=-thres, max=thres)
    
    min_x, max_x = input_range['x1'], input_range['x2']
    min_y, max_y = input_range['y1'], input_range['y2']
    min_z, max_z = input_range['z1'], input_range['z2']
    
    bins_x = np.linspace(-1, 1, num=9)
    bins_y = np.linspace(-1, 1, num=9)
    bins_z = np.linspace(-1, 1, num=9)


    # -1: 1, 1: 9
    # find cube idx
    x_inds = np.digitize([min_x, max_x], bins_x)
    y_inds = np.digitize([min_y, max_y], bins_y)
    z_inds = np.digitize([min_z, max_z], bins_z)

    x_inds -= 1
    y_inds -= 1
    z_inds -= 1

    cube_x1, cube_x2 = x_inds
    cube_y1, cube_y2 = y_inds
    cube_z1, cube_z2 = z_inds

    x1, x2 = cube_x1 * 8, (cube_x2) * 8
    y1, y2 = cube_y1 * 8, (cube_y2) * 8
    z1, z2 = cube_z1 * 8, (cube_z2) * 8

    # clone sdf
    x = sdf.clone()
    x_missing = sdf.clone()
    gen_order = torch.arange(512).cuda()
    gen_order = gen_order.view(8, 8, 8)

    x[:, :, :x1, :, :] = 0.2
    gen_order[:cube_x1, :, :] = -1
    x[:, :, x2:, :, :] = 0.2
    gen_order[cube_x2:, :, :] = -1

    x[:, :, :, :y1, :] = 0.2
    gen_order[:, :cube_y1, :] = -1
    x[:, :, :, y2:, :] = 0.2
    gen_order[:, cube_y2:, :] = -1

    x[:, :, :, :, :z1] = 0.2
    gen_order[:, :, :cube_z1] = -1
    x[:, :, :, :, z2:] = 0.2
    gen_order[:, :, cube_z2:] = -1

    x_missing[:, :, x1:x2, y1:y2, z1:z2] = 0.2

    gen_order = gen_order[gen_order != -1]
    gen_order = gen_order.view(-1)

    return {'sdf' : x, 'sdf_missing': x_missing, 'gen_order': gen_order}


def get_shape_comp_input_mesh(sdf_partial, sdf_missing):
    ############################################
    ## make red cuboid for the partial shapes ##
    ############################################
    # x_p = test_comp_data['sdf'].clone()
    # x_res = test_comp_data['sdf_res'].clone()
    x_p = sdf_partial
    x_res = sdf_missing
    
    mesh_part = sdf_to_mesh(x_p[:1])
    mesh_res = sdf_to_mesh(x_res, color=[1, .6, .6])
    
    if mesh_part is None or mesh_res is None:
        import pdb; pdb.set_trace()
    
    # combine
    mesh_comb = structures.join_meshes_as_scene([mesh_part, mesh_res])
    
    return mesh_comb

def save_mesh_as_gif(mesh_renderer, mesh, nrow=3, out_name='1.gif'):
    """ save batch of mesh into gif """

    # img_comb = render_mesh(mesh_renderer, mesh, norm=False)    

    # rotate
    rot_comb = rotate_mesh_360(mesh_renderer, mesh) # save the first one
    
    # gather img into batches
    nimgs = len(rot_comb)
    nrots = len(rot_comb[0])
    H, W, C = rot_comb[0][0].shape
    rot_comb_img = []
    for i in range(nrots):
        img_grid_i = torch.zeros(nimgs, H, W, C)
        for j in range(nimgs):
            img_grid_i[j] = torch.from_numpy(rot_comb[j][i])
            
        img_grid_i = img_grid_i.permute(0, 3, 1, 2)
        img_grid_i = vutils.make_grid(img_grid_i, nrow=nrow)
        img_grid_i = img_grid_i.permute(1, 2, 0).numpy().astype(np.uint8)
            
        rot_comb_img.append(img_grid_i)
    
    with imageio.get_writer(out_name, mode='I', duration=.08) as writer:
        
        # combine them according to nrow
        for rot in rot_comb_img:
            writer.append_data(rot)

##################### util function for single-view recon #####################

# use for cropping pix3d image given mask. copy from preprocess/create_pix3d_split.py
def crop_square(img, bbox, img_size_h=256, img_size_w=256):
    # from pix2vox
    img_height, img_width, c = img.shape

    x0, y0, x1, y1 = bbox

    # Calculate the size of bounding boxes
    bbox_width = x1 - x0
    bbox_height = y1 - y0
    bbox_x_mid = (x0 + x1) * .5
    bbox_y_mid = (y0 + y1) * .5

    # Make the crop area as a square
    square_object_size = max(bbox_width, bbox_height)
    x_left = int(bbox_x_mid - square_object_size * .5)
    x_right = int(bbox_x_mid + square_object_size * .5)
    y_top = int(bbox_y_mid - square_object_size * .5)
    y_bottom = int(bbox_y_mid + square_object_size * .5)

    # If the crop position is out of the image, fix it with padding
    pad_x_left = 0
    if x_left < 0:
        pad_x_left = -x_left
        x_left = 0
    pad_x_right = 0
    if x_right >= img_width:
        pad_x_right = x_right - img_width + 1
        x_right = img_width - 1
    pad_y_top = 0
    if y_top < 0:
        pad_y_top = -y_top
        y_top = 0
    pad_y_bottom = 0
    if y_bottom >= img_height:
        pad_y_bottom = y_bottom - img_height + 1
        y_bottom = img_height - 1

    # Padding the image and resize the image
    processed_image = np.pad(img[y_top:y_bottom + 1, x_left:x_right + 1],
                                ((pad_y_top, pad_y_bottom), (pad_x_left, pad_x_right), (0, 0)),
                                mode='edge')
    pil_img = Image.fromarray(processed_image)
    pil_img = pil_img.resize((img_size_w, img_size_h))

    return pil_img

def preprocess_img(img_path, img_mask_path, dataset_mode='pix3d'):
    
    # read img and mask
    img = Image.open(img_path).convert('RGB')
    img_mask = np.array(Image.open(img_mask_path).convert('1'))

    img = np.array(Image.open(img_path).convert('RGB'))
    img_mask = np.array(Image.open(img_mask_path).convert('1')).astype(np.uint8) * 255

    # compute bbox from mask
    x, y, w, h = cv2.boundingRect(img_mask)
    # cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
    bbox = [x, y, x+w, y+h]

    # obtain clean img based on mask
    img_clean = img * (img_mask.astype(np.float32) / 255.)[:, :, None]

    # crop based on bbox
    img_crop = crop_square(img_clean.astype(np.uint8), bbox) # pil img


    # copy from pix3d_dataset    
    mean, std = [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]
    to_tensor = transforms.ToTensor()
    basic_transforms = transforms.Compose([
        transforms.Normalize(mean, std),
        transforms.Resize((256, 256)),
    ])

    img_t = to_tensor(img_crop)

    if dataset_mode == 'pix3d':
        # currently: noBG
        _, oh, ow = img_t.shape

        ls = max(oh, ow)

        pad_h1, pad_h2 = (ls - oh) // 2, (ls - oh) - (ls - oh) // 2
        pad_w1, pad_w2 = (ls - ow) // 2, (ls - ow) - (ls - ow) // 2
        img_t = F.pad(img_t[None, ...], (pad_w1, pad_w2, pad_h1, pad_h2), mode='constant', value=0)
        img_t = basic_transforms(img_t[0])
        
    else:
        img_t = basic_transforms(img_t)
    
    if img_t.dim() == 3:
        img_t = img_t.unsqueeze(0)
    return img_t
   
# copy from quant/test_iou.py
def get_img_prob(resnet2vq_model, test_data, opt=None):
    img = test_data['img'].cuda()
    
    img_logits = resnet2vq_model(img) # bs c d h w

    # logsoftmax
    img_logprob = F.log_softmax(img_logits, dim=1) # compute the prob. of next ele
    # img_logprob = torch.sum(img_logprob, dim=1) # multiply the image priors
    img_logprob = rearrange(img_logprob, 'bs c d h w -> (d h w) bs c')

    # ret = img_prob
    return img_logprob 
       
def load_resnet2vq_model(
        opt,
        resnet2vq_ckpt='saved_ckpt/resnet2vq-pix3d_img-all-LR1e-4-cleanCode-pix3dMode-noBG-epoch40.pth',
        ):
    from models.networks.resnet2vq_net import ResNet2VQ
    
    net = ResNet2VQ(opt)
    # resnet2vq_ckpt = resnet2vq_ckpt
    state_dict = torch.load(resnet2vq_ckpt)
    net.load_state_dict(state_dict['resnet2vq'])
    net.eval()
    net.to(opt.device)
    
    return net

##################### util function for lang-guided gen ############`#########

def load_bert2vq_model(opt):
    from models.networks.bert2vq_net import BERT2VQ
    
    net = BERT2VQ(opt)
    # bert2vq_ckpt = '/home/paritosh/Desktop/Capstone/clean-code/generative_transformers/logs/bert2vq-shapenet_lang-all-LR1e-4-cleanCode-langMode-/ckpt/bert2vq_epoch-145.pth'
    bert2vq_ckpt = 'saved_ckpt/bert2vq_epoch-145.pth'
    state_dict = torch.load(bert2vq_ckpt)
    net.load_state_dict(state_dict['bert2vq'])
    net.eval()
    net.to(opt.device)
    
    return net

def get_lang_prob(bert_model,test_data,opt=None):
    lang_logits = bert_model(test_data)
    lang_logprob = F.log_softmax(lang_logits, dim=1) # compute the prob. of next ele
    # img_logprob = torch.sum(img_logprob, dim=1) # multiply the image priors
    lang_logprob = rearrange(lang_logprob, 'bs c d h w -> (d h w) bs c')
    return lang_logprob
    
def load_baseline_je_model(opt):
    from models.base_model import create_model
    # load baselineå stuff
    opt.model='baseline_je'
    opt.tf_cfg='configs/baseline_lang_je_code.yaml'
    opt.ckpt = 'logs/baseline_je-shapenet_lang-all-LR1e-4-clean/ckpt/lang_je_epoch-latest.pth'
    # load vq stuff
    opt.vq_model='pvqvae'
    opt.vq_cfg='configs/pvqvae_snet.yaml'
    opt.vq_ckpt='saved_ckpt/pvqvae-snet-all-LR1e-4-T0.2-rerun-epoch140.pth'
    
    ### opt.vq_dset='sdf_code' # original
    opt.vq_dset='snet'

    model = create_model(opt)
    print(f'[*] "{opt.model}" initialized.')
    model.load_ckpt(opt.ckpt)
        
    return model