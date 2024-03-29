import os
import numpy as np
import hdf5storage
from scipy import ndimage
from argparse import ArgumentParser
from utils.utils_restoration import single2uint,crop_center, imread_uint, imsave, modcrop
from natsort import os_sorted
from prox_PnP_restoration import PnP_restoration
from utils.utils_sr import numpy_degradation
import logging
import time
import warnings
warnings.filterwarnings("ignore")

def SR():
    startstart = time.time()
    parser = ArgumentParser()
    parser.add_argument('--sf', type=int, default=2)
    parser.add_argument('--kernel_path', type=str, default=os.path.join('kernels', 'kernels_12.mat'))
    parser = PnP_restoration.add_specific_args(parser)
    hparams = parser.parse_args()

    # SR specific hyperparameters
    hparams.degradation_mode = 'SR'

    # logging.basicConfig(filename='logs_SR/'+hparams.PnP_algo+hparams.dataset_name+str(hparams.noise_level_img)+'a'+str(hparams.alpha)
    #                     +'la'+str(hparams.lamb)+'g'+str(hparams.gamma)+'sf'+str(hparams.sf)+'sm'+str(hparams.sigma_multi)+'.log', filemode='w', format='%(name)s - %(levelname)s - %(message)s', level=logging.INFO)
    # logger = logging.getLogger()

    logging.basicConfig(filename='SR_comb.log', filemode='a', format='%(name)s - %(levelname)s - %(message)s', level=logging.WARNING)
    logger = logging.getLogger('SR')
    logger.setLevel(logging.INFO)
    logger.info('sf '+ str(hparams.sf) +  ' noise ' + str(hparams.noise_level_img) + ' sigma_mult '+ str(hparams.sigma_multi) + ' alpha '+str(hparams.alpha)+' lamb '+str(hparams.lamb)+' gam '+str(hparams.gamma))
    logger.info(str(hparams.PnP_algo) + ' ' + str(hparams.dataset_name))

    if hparams.PnP_algo == 'DRS':
        hparams.alpha = 0.5
        if hparams.noise_level_img == 2.55:
            hparams.lamb = 5
            hparams.sigma_denoiser = 2 * hparams.noise_level_img
        if hparams.noise_level_img == 7.65:
            hparams.lamb = 1.5
            hparams.sigma_denoiser = 1 * hparams.noise_level_img
        if hparams.noise_level_img == 12.75:
            hparams.lamb = 1.
            hparams.sigma_denoiser = 0.75 * hparams.noise_level_img
    elif hparams.PnP_algo == 'BFGS' or hparams.PnP_algo == 'BFGS2':
        # hparams.sigma_denoiser = max(0.5 * hparams.noise_level_img, 1.9)

        if hparams.noise_level_img == 2.55:
            hparams.sigma_denoiser = 2 * hparams.noise_level_img
        if hparams.noise_level_img == 7.65:
            hparams.sigma_denoiser = 1 * hparams.noise_level_img
        if hparams.noise_level_img == 12.75:
            hparams.sigma_denoiser = 0.75 * hparams.noise_level_img

        # hparams.sigma_denoiser = hparams.sigma_multi * hparams.noise_level_img
    elif hparams.PnP_algo == 'aPGD':
        hparams.sigma_denoiser = hparams.sigma_multi * hparams.noise_level_img
    elif hparams.PnP_algo == 'DPIR' or hparams.PnP_algo == 'DPIR2':
        hparams.sigma_denoiser = hparams.noise_level_img
    else:
        hparams.sigma_denoiser = max(0.5 * hparams.noise_level_img, 1.9)
        # hparams.lamb = 0.99

    # PnP_restoration class
    PnP_module = PnP_restoration(hparams)

    # Set input image paths
    input_path = os.path.join(hparams.dataset_path, hparams.dataset_name)
    input_path = os.path.join(input_path, os.listdir(input_path)[0])
    input_paths = os_sorted([os.path.join(input_path, p) for p in os.listdir(input_path)])

    # Output images and curves paths
    if hparams.extract_images or hparams.extract_curves or hparams.print_each_step:
        den_out_path = 'SR'
        if not os.path.exists(den_out_path):
            os.mkdir(den_out_path)
        exp_out_path = os.path.join(den_out_path, hparams.PnP_algo)
        if not os.path.exists(exp_out_path):
            os.mkdir(exp_out_path)
        exp_out_path = os.path.join(exp_out_path, hparams.dataset_name)
        if not os.path.exists(exp_out_path):
            os.mkdir(exp_out_path)
        exp_out_path = os.path.join(exp_out_path, str(hparams.noise_level_img))
        if not os.path.exists(exp_out_path):
            os.mkdir(exp_out_path)
        exp_out_path = os.path.join(exp_out_path, str(hparams.sf))
        if not os.path.exists(exp_out_path):
            os.mkdir(exp_out_path)
        exp_out_path = os.path.join(exp_out_path, str(hparams.noise_level_img)+str(hparams.gamma)+str(hparams.lamb)+str(hparams.alpha))
        if not os.path.exists(exp_out_path):
            os.mkdir(exp_out_path)

    # Load the 8 blur kernels
    kernels = hdf5storage.loadmat(hparams.kernel_path)['kernels']
    # Kernels follow the order given in the paper (Table 3)
    if hparams.full:
        k_list = range(4)
    else:
        k_list = [1]
    psnr_list = []
    F_list = []

    print(
        '\n Prox-PnP ' + hparams.PnP_algo + ' super-resolution with image sigma:{:.3f}, model sigma:{:.3f}, lamb:{:.3f} \n'.format(
            hparams.noise_level_img, hparams.sigma_denoiser, hparams.lamb))


    for k_index in k_list: # For each kernel
        start = time.time()
        PnP_module.reset_filters()
        psnr_k_list = []
        psnrY_k_list = []

        k = kernels[0, k_index].astype(np.float64)

        if hparams.extract_images or hparams.extract_curves:
            kout_path = os.path.join(exp_out_path, 'kernel_' + str(k_index))
            if not os.path.exists(kout_path):
                os.mkdir(kout_path)

        if hparams.extract_curves:
            PnP_module.initialize_curves()

        for i in range(min(len(input_paths),hparams.n_images)): # For each image

            print('__ kernel__',k_index, '__ image__',i)
            # logger.info('__ kernel__'+str(k_index)+'__ image__'+str(i))
            ## load image
            input_im_uint = imread_uint(input_paths[i])
            if hparams.patch_size < min(input_im_uint.shape[0], input_im_uint.shape[1]):
               input_im_uint = crop_center(input_im_uint, hparams.patch_size,hparams.patch_size)
            # Degrade image
            input_im_uint = modcrop(input_im_uint, hparams.sf)
            input_im = np.float32(input_im_uint / 255.)
            blur_im = numpy_degradation(input_im, k, hparams.sf)
            np.random.seed(seed=0)
            noise = np.random.normal(0, hparams.noise_level_img/255., blur_im.shape)
            blur_im += noise
            init_im = blur_im

            # PnP restoration
            if hparams.extract_images or hparams.extract_curves or hparams.print_each_step:
                deblur_im, output_psnr, output_psnrY, x_list, z_list, y_list, Dg_list, psnr_tab, Dx_list, g_list, F_list, Psi_list = PnP_module.restore(blur_im, init_im, input_im, k, extract_results=True)
            else:
                deblur_im, output_psnr, output_psnrY = PnP_module.restore(blur_im, init_im, input_im, k)

            print('PSNR: {:.2f}dB'.format(output_psnr))
            # logger.info('PSNR: {:.2f}dB'.format(output_psnr))
            psnr_k_list.append(output_psnr)
            psnrY_k_list.append(output_psnrY)
            psnr_list.append(output_psnr)

            if hparams.extract_curves:
                # Create curves
                PnP_module.update_curves(x_list, psnr_tab, F_list, Psi_list)

            if hparams.extract_images:
                # Save images
                save_im_path = os.path.join(kout_path, 'images')
                if not os.path.exists(save_im_path):
                    os.mkdir(save_im_path)
                save_im_path = os.path.join(save_im_path, hparams.PnP_algo)
                if not os.path.exists(save_im_path):
                    os.mkdir(save_im_path)

                imsave(os.path.join(save_im_path, 'img_' + str(i) + '_input.png'), input_im_uint)
                imsave(os.path.join(save_im_path, 'img_' + str(i) + '_deblur.png'), single2uint(deblur_im))
                imsave(os.path.join(save_im_path, 'img_' + str(i) + '_blur.png'), single2uint(blur_im))
                imsave(os.path.join(save_im_path, 'img_' + str(i) + '_init.png'), single2uint(init_im))
                # print('output image saved at ', os.path.join(save_im_path, 'img_' + str(i) + '_deblur.png'))

        if hparams.extract_curves:
            # Save curves
            save_curves_path = os.path.join(kout_path, 'curves')
            if not os.path.exists(save_curves_path):
                os.mkdir(save_curves_path)
            PnP_module.save_curves(save_curves_path)
            print('output curves saved at ', save_curves_path)

        avg_k_psnr = np.mean(np.array(psnr_k_list))
        print('avg RGB psnr on kernel {}: {:.2f}dB'.format(k_index, avg_k_psnr))
        logger.info('avg RGB psnr on kernel {}: {:.2f}dB'.format(k_index, avg_k_psnr))
        avg_k_psnrY = np.mean(np.array(psnrY_k_list))
        print('avg Y psnr on kernel {} : {:.2f}dB'.format(k_index, avg_k_psnrY))
        # logger.info('avg Y psnr on kernel {} : {:.2f}dB'.format(k_index, avg_k_psnrY))
        # logger.info('kernel' + str(k_index) + " time " + str(time.time()-start))

    print(np.mean(np.array(psnr_list)))
    logger.info("Final mean PSNR"+str(np.mean(np.array(psnr_list))))
    logger.info("Time" + str(time.time()-startstart))
    return np.mean(np.array(psnr_list))

if __name__ == '__main__':
    SR()
