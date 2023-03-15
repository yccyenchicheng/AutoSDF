# AutoSDF: Shape Priors for 3D Completion, Reconstruction and Generation
[[`arXiv`](https://arxiv.org/abs/2203.09516)]
[[`Project Page`](https://yccyenchicheng.github.io/AutoSDF/)]
[[`BibTex`](#citation)]

Code release for the CVPR 2022 paper "AutoSDF: Shape Priors for 3D Completion, Reconstruction and Generation".

https://user-images.githubusercontent.com/27779063/159215086-6889da7c-07c6-4909-b51f-bc04364072cf.mp4

# Installation
Please install [`pytorch`](https://pytorch.org/) and [`pytorch3d`](https://github.com/facebookresearch/pytorch3d). Or you can setup the environment using `conda`:

```
conda env create -f autosdf.yaml
conda activate autosdf
```

However, the environments varies by each machine. We tested the code on `Ubuntu 20.04`, `cuda=11.3`, `python=3.8.11`, `pytorch=1.9.0`, `pytorch3d=0.5.0`.

# Demo
We provide a jupyter notebook for demo. First download the pretrained weights from [this link](https://drive.google.com/drive/folders/1n8W_8CfQ7uZDYNrv487sd0oyhRoNLfGo?usp=sharing), and put them under `saved_ckpt`. Then start the notebook server with
```
jupyter notebook
```
And run:
- `demo_shape_comp.ipynb` for shape completion
- `demo_single_view_recon.ipynb` for single-view reconstruction
- `demo-lang-conditional.ipynb` for language-guided generation

# Preparing the Data
1. [ShapeNet](https://www.shapenet.org)

First you need to download the `ShapeNetCore.v1` following the instruction of `https://www.shapenet.org/account/`. Put them under `data/ShapeNet`. Then unzip the downloaded zip file. We assume the path to the unzipped folder is `data/ShapeNet/ShapeNetCore.v1`. To extract SDF values, we followed the [preprocessing steps from DISN](https://github.com/laughtervv/DISN/blob/master/preprocessing/create_point_sdf_grid.py).

2. [Pix3D](https://github.com/xingyuansun/pix3d)

The Pix3D dataset can be downloaded here: https://github.com/xingyuansun/pix3d.

# Training
1. First train the `P-VQ-VAE` on `ShapeNet`:
```
./launchers/train_pvqvae_snet.sh
```

2. Then extract the code for each sample of ShapeNet (caching them for training the transformer):
```
./launchers/extract_pvqvae_snet.sh
```

3. Train the random-order-transformer to learn the shape prior:
```
./launchers/train_rand_tf_snet_code.sh
```

4. To train the image marginal on Pix3D, first extract the code for each training data of Pix3D
```
./launchers/extract_pvqvae_pix3d.sh
```

5. Train the image marginal on Pix3D
```
./launchers/train_resnet2vq_pix3d_img.sh
```

# Issues and FAQ

## 1. Regarding `mcubes` functions
We originally use the implementation of the marching cubes from this repo: https://github.com/JustusThies/PyMarchingCubes. However, some of the dependencies seems to be outdated and makes the installation troublesome. Currently the quick workaround is installing `mcubes` from https://github.com/pmneila/PyMCubes:
```
pip install PyMCubes
```
and replace all the lines `import marching_cubes as mcubes` in our code with `import mcubes`. 

# <a name="citation"></a>Citing AutoSDF

If you find this code helpful, please consider citing:

```BibTeX
@inproceedings{autosdf2022,
  title={{AutoSDF}: Shape Priors for 3D Completion, Reconstruction and Generation},
  author={Mittal, Paritosh and Cheng, Yen-Chi and Singh, Maneesh and Tulsiani, Shubham},
  booktitle={CVPR},
  year={2022}
}
```

# Acknowledgement
This code borrowed heavily from [Cycle-GAN](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix), [VQ-GAN](https://github.com/CompVis/taming-transformers). Thanks for the efforts for making their code available!
