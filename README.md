# Provably Convergent Plug-and-Play Quasi-Newton Methods

Code for the paper "Provably Convergent Plug-and-Play Quasi-Newton Methods", submitted to SIAM Imaging Sciences.

[[Paper](https://arxiv.org/abs/2303.07271)]


## Prerequisites


The code was computed using Python 3.10, Pytorch **1.12.1**. The code was originally computed with Python 3.8.10, PyTorch Lightning 1.2.6, PyTorch 1.7.1. 


## Prox-Denoiser (Prox-DRUNet)

The code relative to the Proximal (Gradient Step) Denoiser can be found in the ```GS_denoising``` directory. This repository contains, along with the proposed PnP-LBFGS1 and PnP-LBFGS2 methods, the following competing methods:
- PnP-PGD, PnP-DRS, PnP-DRSdiff from the Prox-PnP papers
- Relaxed PnP-PGD, or PnP-aPGD from [this](https://arxiv.org/pdf/2301.13731.pdf) followup paper.
- PnP-FISTA
- (Optional) DPIR, which is PnP-HQS. For this, clone the DPIR repository into the main folder (so the top level looks like ``GS_denoising, PnP_restoration, DPIR, ...```.
  - For DPIR, please copy ```utils_model, utils_image``` from their utils folder to the utils folder in ```PnP_restoration/utils```, and download the ```drunet_color.pth``` model from [their page](https://github.com/cszn/DPIR/tree/master/model_zoo). 

### Pre-trained model

- **Download pretrained checkpoint from https://plmbox.math.cnrs.fr/f/faf7d62213e449fa9c8a/?dl=1 and save it as ```GS_denoising/ckpts/Prox_DRUNet.ckpt```**

## Implementation
The implementation for the BFGS method can be found in ```PnP_restoration/prox_PnP_restoration```.
## Usage
A batch file containing a script to test all methods, as well as produce a log for hyperparameter grid search, can be found in the repository.

Example:
```
cd PnP_restoration
python (deblur|SR).py --dataset_name (set3c|CBSD10|CBSD68) --PnP_algo (BFGS|BFGS2|PGD|aPGD|DRS|DRSdiff) --noise_level_img (2.25|7.65|12.75) (--extract_curves) (--extract_images) (--sf 2) (--params)

CUDA_VISIBLE_DEVICES=1 python SR.py --dataset_name CBSD68 --PnP_algo BFGS2 --noise_level_img 2.55 --extract_curves --extract_images --sigma_multi=2.0 --lamb=4.0 --gamma=1.0 --beta=0.01 --alpha=0.5 --maxitr=100 --sf=2


python deblur.py --dataset_name CBSD68 --PnP_algo BFGS --noise_level_img 2.25 --extract_curves --extract_images --sigma_multi=1.0 --lamb=1.0 --gamma=1.0 --beta=0.01 --alpha=0.5 --maxitr=100
python deblur.py --dataset_name CBSD68 --PnP_algo BFGS --noise_level_img 7.65 --extract_curves --extract_images --sigma_multi=0.75 --lamb=1.0 --gamma=0.85 --beta=0.01 --alpha=0.5 --maxitr=100
python deblur.py --dataset_name CBSD68 --PnP_algo BFGS --noise_level_img 12.75 --extract_curves --extract_images --sigma_multi=0.75 --lamb=1.0 --gamma=1.0 --beta=0.01 --alpha=0.7 --maxitr=100
```


### Other things
We updated the fft transforms from Pytorch 1.7.1 to Pytorch 1.12.1. We also added the relaxed proximal denoiser as in [this followup paper](https://arxiv.org/pdf/2301.13731.pdf).

## Acknowledgments
This repo is based on the following repo:
- Proximal denoiser for convergent plug-and-play optimization with nonconvex regularization : https://github.com/samuro95/Prox-PnP

This repo contains parts of code taken from : 
- Deep Plug-and-Play Image Restoration (DPIR) : https://github.com/cszn/DPIR 
- Gradient Step Denoiser for convergent Plug-and-Play (GS-PnP) : https://github.com/samuro95/GSPnP

