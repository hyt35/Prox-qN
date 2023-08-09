# Provably Convergent Plug-and-Play Quasi-Newton Methods

Code for the paper "Provably Convergent Plug-and-Play Quasi-Newton Methods", submitted to SIAM Imaging Sciences.

[[Paper](https://arxiv.org/abs/2303.07271)]


## Prerequisites


The code was computed using Python 3.10, Pytorch **1.12.1**. The code was originally computed with Python 3.8.10, PyTorch Lightning 1.2.6, PyTorch 1.7.1. 


## Prox-Denoiser (Prox-DRUNet)

The code relative to the Proximal (Gradient Step) Denoiser can be found in the ```GS_denoising``` directory.

### Pre-trained model

- **Download pretrained checkpoint from https://plmbox.math.cnrs.fr/f/faf7d62213e449fa9c8a/?dl=1 and save it as ```GS_denoising/ckpts/Prox_DRUNet.ckpt```**

## Implementation
The implementation for the BFGS method can be found in ```PnP_restoration/prox_PnP_restoration```.
## Usage
A batch file containing a script to test all methods, as well as produce a log for hyperparameter grid search, can be found in the repository.

Example:
```
cd PnP_restoration
python SR.py --dataset_name CBSD68 --PnP_algo DRS --noise_level_img 7.65 --sf 2

python deblur.py --dataset_name CBSD68 --PnP_algo BFGS --noise_level_img 2.25 --extract_curves --extract_images --sigma_multi=1.0 --lamb=1.0 --gamma=1.0 --beta=0.01 --alpha=0.5 --maxitr=100
python deblur.py --dataset_name CBSD68 --PnP_algo BFGS --noise_level_img 7.65 --extract_curves --extract_images --sigma_multi=0.75 --lamb=1.0 --gamma=0.85 --beta=0.01 --alpha=0.5 --maxitr=100
python deblur.py --dataset_name CBSD68 --PnP_algo BFGS --noise_level_img 12.75 --extract_curves --extract_images --sigma_multi=0.75 --lamb=1.0 --gamma=1.0 --beta=0.01 --alpha=0.7 --maxitr=100
```



## Acknowledgments
This repo is based on the following repo:
- Proximal denoiser for convergent plug-and-play optimization with nonconvex regularization : https://github.com/samuro95/Prox-PnP

This repo contains parts of code taken from : 
- Deep Plug-and-Play Image Restoration (DPIR) : https://github.com/cszn/DPIR 
- Gradient Step Denoiser for convergent Plug-and-Play (GS-PnP) : https://github.com/samuro95/GSPnP

