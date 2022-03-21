import os
import numpy as np
import imageio
from einops import rearrange
from termcolor import colored, cprint
from tqdm import tqdm
from omegaconf import OmegaConf

import torch
import torch.backends.cudnn as cudnn
import torchvision.utils as vutils
cudnn.benchmark = True

from options.test_options import TestOptions
from datasets.dataloader import CreateDataLoader, get_data_generator
from models.base_model import create_model

from configs.paths import dataroot
from utils import util

opt = TestOptions().parse()
opt.phase = 'test'

seed = opt.seed
util.seed_everything(seed)

train_dl, test_dl = CreateDataLoader(opt)
train_ds, test_ds = train_dl.dataset, test_dl.dataset

if opt.dataset_mode == 'snet':
    assert len(list( set(train_ds.model_list) & set(test_ds.model_list) )) == 0, 'should not have duplicate data in train/test'
elif opt.dataset_mode == 'pix3d':
    tr_model_list = [m for m in train_ds.sdf_list if 'chair' not in m]
    tt_model_list = [m for m in test_ds.sdf_list if 'chair' not in m]
    assert len(list( set(tr_model_list) & set(tt_model_list) )) == 0, 'should not have duplicate data in train/test'

assert opt.batch_size == 1, 'only supports batch_size=1'

test_dg = get_data_generator(test_dl)

cprint('[*] # training images = %d' % len(train_ds), 'yellow')
cprint('[*] # testing images = %d' % len(test_ds), 'yellow')

# main loop
model = create_model(opt)
cprint(f'[*] "{opt.model}" initialized.', 'cyan')

# load ckpt
model.load_ckpt(opt.ckpt)

output_dir_name = f'{opt.model}-{opt.dataset_mode}-{opt.cat}-T{opt.trunc_thres}'
if opt.vq_note != 'default':
    output_dir_name = f'{output_dir_name}-{opt.vq_note}'
code_output_dir = '%s/extracted_code/%s' % (dataroot, output_dir_name)
if not os.path.exists(code_output_dir):
    os.makedirs(code_output_dir)

print(colored('[*] Start extracting code from %s' % opt.ckpt, 'magenta'))
total_steps = 0

print(colored('[*] Dumping relevent files...', 'magenta'))

model_configs_name = 'pvqvae_cfg.yaml' #% code_output_dir
model_weight_name = 'pvqvae_weight.pth' #% code_output_dir
codebook_weight_name = 'codebook.pth' #% code_output_dir

codebook_state_dict = {
    'codebook': model.get_codebook_weight()
}

os.system('cp %s %s/%s' % (opt.vq_cfg, code_output_dir, model_configs_name))
os.system('cp %s %s/%s' % (opt.ckpt, code_output_dir, model_weight_name))
# os.system('cp %s %s' % (opt.ckpt, codebook_weight_p))
torch.save(codebook_state_dict, '%s/%s' % (code_output_dir, codebook_weight_name))

code_configs = {
    'pvqvae_cfg_file': model_configs_name,
    'pvqvae_pth': model_weight_name,
    
    'codebook_weight': codebook_weight_name,
    'dataset_mode': opt.dataset_mode,
    'cat': opt.cat,
    'threshold': opt.trunc_thres,
}

code_conf = OmegaConf.create(code_configs)
with open('%s/code_configs.yaml' % code_output_dir, 'w') as f:
    OmegaConf.save(config=code_conf, f=f)

train_dl.dataset[0]

for phase in ['train', 'test']:
    if phase == 'train':
        dl = train_dl
    else:
        dl = test_dl

    for i, data in tqdm(enumerate(dl), total=len(dl), desc='Extracting: %s split...' % phase):

        # self.x_recon, self.quant, _, self.info = 
        model.inference(data, verbose=True)

        # if i == 10:
            # break

        if opt.model == 'pvqvae':
            x_recon, quant, info = model.x_recon, model.zq_voxels, model.info
        else:
            x_recon, quant, info = model.x_recon, model.quant, model.info
        _, _, quant_ix = info

        bs = quant.shape[0]

        if len(quant.shape) == 5:
            d, h, w = quant.shape[-3:]
            quant_ix = rearrange(quant_ix, '(b d h w) -> b d h w', b=bs, d=d, h=h, w=w)
        else:
            h, w = quant.shape[-2:]
            quant_ix = rearrange(quant_ix, '(b h w) -> b h w', b=bs, h=h, w=w)

        x = model.x
        x = x.detach().cpu().numpy()

        quant_np = quant.detach().cpu().numpy()
        quant_ix_np = quant_ix.detach().cpu().numpy()

        path_list = data['path']
        for b in range(bs):
            data_ix = i * opt.batch_size + b

            # create dir like sdf_dataset here
            path = path_list[b]

            # handle differently for pix3d
            if opt.dataset_mode == 'pix3d':
                cat_name, model_id, obj_name = path.split('/')[-3:]
                obj_name_noext = obj_name.replace('ori_sample_grid', '').replace('.h5', '')
                
                model_dir = f'{code_output_dir}/{cat_name}/{model_id}'
                if not os.path.exists(model_dir):
                    os.makedirs(model_dir)           
                
                if obj_name_noext == '':
                    code_out = f'{model_dir}/code.npy'
                    code_entry_out = f'{model_dir}/codeix.npy'
                    x_out = f'{model_dir}/x.npy'   
                else:
                    code_out = f'{model_dir}/code{obj_name_noext}.npy'
                    code_entry_out = f'{model_dir}/codeix{obj_name_noext}.npy'
                    x_out = f'{model_dir}/x{obj_name_noext}.npy'
            else:
                synset_id, model_id, _ = path.split('/')[-3:]

                model_dir = f'{code_output_dir}/{synset_id}/{model_id}'
                if not os.path.exists(model_dir):
                    os.makedirs(model_dir)
                    
                code_out = f'{model_dir}/code.npy'
                code_entry_out = f'{model_dir}/codeix.npy'
                x_out = f'{model_dir}/x.npy'
                
            np.save(x_out, x[b])
            
            np.save(code_out, quant_np[b])
            np.save(code_entry_out, quant_ix_np[b])
        
            # debug
            # from utils.util_3d import init_mesh_renderer, render_sdf
            # img = render_sdf(model.renderer, model.x)
            # img_recon = render_sdf(model.renderer, model.x_recon)
            # vutils.save_image(img, 'img.png')
            # vutils.save_image(img_recon, 'img_r.png')
            # import pdb; pdb.set_trace()
            

