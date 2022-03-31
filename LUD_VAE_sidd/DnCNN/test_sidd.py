from scipy.io import loadmat, savemat

import os.path
import logging

import numpy as np
from datetime import datetime
from collections import OrderedDict
# from scipy.io import loadmat

import torch

from utils import utils_logger
from utils import utils_model
from utils import utils_image as util

def main():

    # ----------------------------------------
    # Preparation
    # ----------------------------------------

    model_name = 'dncnn'  
    testset_name = 'sidd'           # test set, 'bsd68' | 'set12'

    model_pool = 'model_zoo'  # fixed

    model_path = os.path.join(model_pool, model_name+'.pth')

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

    noise_im_test_root = '/home/dihan/dataset/SIDD_Small_sRGB/BenchmarkNoisyBlocksSrgb.mat'
    result_root = './SIDD_results.mat'
    
    noise_im_test = loadmat(noise_im_test_root)
    noise_im_test = noise_im_test['BenchmarkNoisyBlocksSrgb']
    
    result = np.zeros((40, 32, 256, 256, 3))
    
    for i in range(40):
        for j in range(32):
            noisy = np.array(noise_im_test[i, j, ...])
            
            noisy = util.uint2tensor4(noisy)
            noisy = noisy.to(device)
            out = model(noisy)        
            result[i, j, ...] = util.tensor2uint(out)
            
            print(i, j)
            
    result = result.astype(np.uint8)
    
    savemat(result_root, {'result': result})

if __name__ == '__main__':

    main()
