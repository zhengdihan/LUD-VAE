import os.path
import logging

import numpy as np
from datetime import datetime
from collections import OrderedDict

import torch

from utils import utils_logger
from utils import utils_model
from utils import utils_image as util

def main():

    # ----------------------------------------
    # Preparation
    # ----------------------------------------

    model_name = 'dncnn'            
    testset_name = 'sidd'           # test set, 'sidd' | 'polyu'

    model_pool = 'model_zoo'        # fixed

    results = 'results'             # fixed
    result_name = testset_name + '_' + model_name  # fixed
    model_path = os.path.join(model_pool, model_name+'.pth')

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

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # ----------------------------------------
    # load model
    # ----------------------------------------

    from models.network_dncnn import DnCNN as net
    model = net(in_nc=3, out_nc=3, nc=64, nb=17, act_mode='BR')
    model.load_state_dict(torch.load(model_path), strict=True)
    model.eval()
    for k, v in model.named_parameters():
        v.requires_grad = False
    model = model.to(device)
    logger.info('Model path: {:s}'.format(model_path))
    number_parameters = sum(map(lambda x: x.numel(), model.parameters()))
    logger.info('Params number: {}'.format(number_parameters))

    test_results = OrderedDict()
    test_results['psnr'] = []
    test_results['ssim'] = []

    logger.info('model_name:{}'.format(model_name))
    logger.info(L_path)
    L_paths = util.get_image_paths(L_path)
    H_paths = util.get_image_paths(H_path)

    for idx, img in enumerate(L_paths):

        # ------------------------------------
        # (1) img_L
        # ------------------------------------

        img_name, ext = os.path.splitext(os.path.basename(img))
        # logger.info('{:->4d}--> {:>10s}'.format(idx+1, img_name+ext))
        img_L = util.imread_uint(img, n_channels=3)
        img_L = util.uint2single(img_L)

        img_L = util.single2tensor4(img_L)
        img_L = img_L.to(device)

        # ------------------------------------
        # (2) img_E
        # ------------------------------------
        
        img_E = model(img_L)
        img_E = util.tensor2uint(img_E)

        # --------------------------------
        # (3) img_H
        # --------------------------------

        img_H = util.imread_uint(H_paths[idx], n_channels=3)
        img_H = img_H.squeeze()

        # --------------------------------
        # PSNR and SSIM
        # --------------------------------

        psnr = util.calculate_psnr(img_E, img_H, border=0)
        ssim = util.calculate_ssim(img_E, img_H, border=0)
        test_results['psnr'].append(psnr)
        test_results['ssim'].append(ssim)
        logger.info('{:s} - PSNR: {:.2f} dB; SSIM: {:.4f}.'.format(img_name+ext, psnr, ssim))

        # ------------------------------------
        # save results
        # ------------------------------------

        util.imsave(img_E, os.path.join(E_path, img_name+ext))

    ave_psnr = sum(test_results['psnr']) / len(test_results['psnr'])
    ave_ssim = sum(test_results['ssim']) / len(test_results['ssim'])
    logger.info('Average PSNR/SSIM(RGB) - {} - PSNR: {:.2f} dB; SSIM: {:.4f}'.format(result_name, ave_psnr, ave_ssim))

if __name__ == '__main__':

    main()
