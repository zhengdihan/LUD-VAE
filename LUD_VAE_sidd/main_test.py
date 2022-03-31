import os.path
import logging

import numpy as np
from collections import OrderedDict

import torch

from utils import utils_logger
from utils import utils_image as util
from utils.utils_eval import eval_split

import cv2

def main():

    # ----------------------------------------
    # Preparation
    # ----------------------------------------

    model_name = 'ludvae'
    testset_name = 'sidd'
    border = 0

    normalize = False

    H_noise_level = 40
    L_noise_level = 0
    temperature = 1.0

    n_max = 100

    model_pool = 'trained_models/LUDVAE_models'             # fixed
    results = 'results'                  # fixed
    task_current = 'trans'
    result_name = testset_name + '_' + task_current + '_' + model_name

    model_path = os.path.join(model_pool, model_name+'.pth')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.cuda.empty_cache()

    # ----------------------------------------
    # L_path, E_path, H_path
    # ----------------------------------------

    L_path = '/home/dihan/dataset/SIDD_Small_sRGB/val_noisy/'
    H_path = '/home/dihan/dataset/SIDD_Small_sRGB/val_clean/'
    E_path = os.path.join(results, result_name)   # E_path, for Estimated images
    util.mkdir(E_path)

    logger_name = result_name
    utils_logger.logger_info(logger_name, log_path=os.path.join(E_path, logger_name+'.log'))
    logger = logging.getLogger(logger_name)

    # ----------------------------------------
    # load model
    # ----------------------------------------
    from models.network_ludvae import LUDVAE
    model = LUDVAE()
    states = torch.load(model_path)
    model.load_state_dict(states, strict=True)
    model.eval()

    for k, v in model.named_parameters():
        v.requires_grad = False

    model = model.to(device)

    test_results = OrderedDict()
    test_results['psnr'] = []
    test_results['ssim'] = []

    logger.info('model_name:{}'.format(model_name))
    logger.info(L_path)
    L_paths = util.get_image_paths(L_path)

    logger.info(H_path)
    H_paths = util.get_image_paths(H_path)

    for idx, img in enumerate(L_paths):

        # ------------------------------------
        # (1) img_L
        # ------------------------------------

        img_name, ext = os.path.splitext(os.path.basename(img))
        img_L = util.imread_uint(img, n_channels=3)
        img_H = util.imread_uint(H_paths[idx], n_channels=3)

        img_hL = img_L.copy()
        img_hH = img_H.copy()

        img_L = util.uint2tensor4(img_L).to(device)
        img_hL = util.uint2tensor4(img_hL).to(device)
        img_hL = img_hL + torch.randn_like(img_hL) * L_noise_level / 255

        img_H = util.uint2tensor4(img_H).to(device)
        img_hH = util.uint2tensor4(img_hH).to(device)
        img_hH = img_hH + torch.randn_like(img_hH) * H_noise_level / 255

        if normalize:
            img_TL = torch.clamp(((img_L - mean_noisy)/std_noisy)*std_clean + mean_clean, 0, 1)
            img_ThL = torch.clamp(((img_hL - mean_noisy)/std_noisy)*std_clean + mean_clean, 0, 1)
        else:
            img_TL = img_L
            img_ThL = img_hL

        # ------------------------------------
        # (2) img_E
        # ------------------------------------

        label_H = torch.zeros(1, 1, 1, 1).long().to(device)
        label_L = torch.ones(1, 1, 1, 1).long().to(device)

        img_E = model.translate(img_TL, img_ThL, label_L)
        img_G = model.translate(img_H, img_hH, label_H, temperature=temperature)

        if normalize:
            img_G = (((img_G - mean_clean)/std_clean)*std_noisy) + mean_noisy

        img_E = util.tensor2uint(img_E)
        img_G = util.tensor2uint(img_G)
        img_L = util.tensor2uint(img_L)
        img_H = util.tensor2uint(img_H)

        # --------------------------------
        # PSNR and SSIM
        # --------------------------------

        psnr = util.calculate_psnr(img_E, img_H, border=border)
        ssim = util.calculate_ssim(img_E, img_H, border=border)
        test_results['psnr'].append(psnr)
        test_results['ssim'].append(ssim)
        logger.info('{:s} - PSNR: {:.2f} dB; SSIM: {:.4f}.'.format(img_name+ext, psnr, ssim))

        # ------------------------------------
        # save results
        # ------------------------------------

        util.imsave(img_E, os.path.join(E_path, img_name+'_E'+ext))
        util.imsave(img_L, os.path.join(E_path, img_name+'_L'+ext))
        util.imsave(img_H, os.path.join(E_path, img_name+'_H'+ext))
        util.imsave(img_G, os.path.join(E_path, img_name+'_G'+ext))

        if idx >= n_max:
            break

    ave_psnr = sum(test_results['psnr']) / len(test_results['psnr'])
    ave_ssim = sum(test_results['ssim']) / len(test_results['ssim'])
    logger.info('Average PSNR/SSIM(RGB) - {} - PSNR: {:.2f} dB; SSIM: {:.4f}'.format(result_name, ave_psnr, ave_ssim))


if __name__ == '__main__':

    main()
