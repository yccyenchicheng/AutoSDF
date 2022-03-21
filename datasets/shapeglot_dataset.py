"""
    adopted from: https://github.com/shubhtuls/PixelTransformer/blob/03b65b8612fe583b3e35fc82b446b5503dd7b6bd/data/shapenet.py
"""
import collections
import csv
import glob
import json
import os
import socket

import numpy as np

from termcolor import cprint

import torch

from configs.paths import dataroot
from datasets.base_dataset import BaseDataset



hostname = socket.gethostname()


class ShapeGlotConditionalDataset(BaseDataset):

    # def initialize(self, opt, phase='train', cat='chair'):
    def initialize(self, opt, phase='train', cat='all'):
        self.opt = opt
        self.max_dataset_size = opt.max_dataset_size
        self.phase = phase
        
        with open(f'{dataroot}/ShapeNet/info.json') as f:
            self.info = json.load(f)
        
        self.cat_to_id = self.info['cats']
        self.id_to_cat = {v: k for k, v in self.cat_to_id.items()}
        # TODO(Paritosh): Change to opt.vq_model
        code_setting = f'{opt.vq_model}-{opt.vq_dset}-{opt.vq_cat}-T{opt.trunc_thres}'
        print(code_setting)
        # vq_model = "sdf"
        # code_setting = f'{vq_model}_{opt.vq_dset}-{opt.vq_cat}-T{opt.trunc_thres}'
        self.code_dir = f'{dataroot}/extracted_code/{code_setting}'
        assert os.path.exists(self.code_dir), f'{self.code_dir} should exist.'

        
        # NOTE: set code_root here for transformer_model to load
        cprint('[*] setting opt.code_dir = %s in SDFCodeDataset.' % self.code_dir, 'yellow')
        # opt.code_dir = self.code_dir
        self.shapenet_lang = collections.defaultdict(list)
        with open(f'{dataroot}/ShapeGlot/data/main_data_for_chairs/language/shapenet_chairs.csv') as f:
            shapeglot_info = csv.reader(f)
            next(shapeglot_info)
            for r in shapeglot_info:
                target = int(r[7])
                shapenet_name = r[target]
                text = r[8]
                self.shapenet_lang[shapenet_name].append(text)

        
        cats = ['chair']
        self.lang_list = []
        self.model_list = []
        self.cats_list = []
        for c in cats:
            synset = self.info['cats'][c]

            with open(f'{dataroot}/ShapeNet/filelists/{synset}_{phase}.lst') as f:
                lang_list_s = []
                model_list_s = []
                for l in f.readlines():
                    model_id = l.rstrip('\n')
                    
                    model_texts = self.shapenet_lang[model_id]
                    for model_text in model_texts:
                        lang_list_s.append(model_text)
                    n_model_texts = len(model_texts)
                    
                    # for code
                    code_path = f'{self.code_dir}/{synset}/{model_id}'
                    for _ in range(n_model_texts):
                        model_list_s.append(code_path)
                    

                self.lang_list += lang_list_s
                self.model_list += model_list_s
                self.cats_list += [synset] * len(lang_list_s)
                print('[*] %d samples for %s (%s).' % (len(lang_list_s), self.id_to_cat[synset], synset))

        idx = np.arange(len(self.lang_list))
        np.random.default_rng(seed=0).shuffle(idx)
        
        self.lang_list = np.array(self.lang_list)[idx]
        self.model_list = np.array(self.model_list)[idx]
        self.cats_list = np.array(self.cats_list)[idx]
        
        self.lang_list = self.lang_list[:self.max_dataset_size]
        self.model_list = self.model_list[:self.max_dataset_size]
        cprint('[*] %d lang_list loaded.' % (len(self.lang_list)), 'yellow')
        cprint('[*] %d code loaded.' % (len(self.model_list)), 'yellow')
        
        self.N = len(self.lang_list)

        
    def __getitem__(self, index):
        
        synset = self.cats_list[index]
        model = self.model_list[index]
        text = self.lang_list[index]
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
            'text': text,
            'cat_id': synset,
            'cat_str': self.id_to_cat[synset],
            'path': model,
        }

        return ret
    def __len__(self):
        return self.N

    def name(self):
        return 'ShapeGlotMarginalDataset'

