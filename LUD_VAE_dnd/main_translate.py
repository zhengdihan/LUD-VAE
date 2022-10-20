import os.path
import logging

import numpy as np
from collections import OrderedDict

import torch

from utils import utils_logger
from utils import utils_image as util
from utils.utils_eval import eval_split

from tqdm import tqdm
import random

def main():

    # ----------------------------------------
    # Preparation
    # ----------------------------------------

    model_name = 'ludvae'

    noise_level = [10, 30]
    temperature = [1, 3]

    model_pool = 'trained_models/LUDVAE_models' # fixed

    model_path = os.path.join(model_pool, model_name+'.pth')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.cuda.empty_cache()

    H_path = '/home/dihan/dataset/SIDD_Small_sRGB/clean/'
    G_path = './datasets/DND/noisy_syn'
    util.mkdir(G_path)

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

    H_paths = util.get_image_paths(H_path)

    for idx, img in enumerate(tqdm(H_paths)):

        # ------------------------------------
        # (1) img_H
        # ------------------------------------

        img_name, ext = os.path.splitext(os.path.basename(img))
        img_H = util.imread_uint(img, n_channels=3)
        img_hH = img_H.copy()

        img_H = util.uint2tensor4(img_H).to(device)
        img_hH = util.uint2tensor4(img_hH).to(device)

        noise_level_min = noise_level[0]
        noise_level_max = noise_level[1]

        sigma = random.uniform(noise_level_min, noise_level_max)
        img_hH = img_hH + torch.randn_like(img_hH) * sigma / 255.0

        # ------------------------------------
        # (2) img_G
        # ------------------------------------

        label_H = torch.zeros(1, 1, 1, 1).long().to(device)
        
        t_min = temperature[0]
        t_max = temperature[1]
        
        t = random.uniform(t_min, t_max)

        img_G = model.translate(img_H, img_hH, label_H, temperature=t)
        img_G = util.tensor2uint(img_G)

        # ------------------------------------
        # save results
        # ------------------------------------
        util.imsave(img_G, os.path.join(G_path, img_name + ext))

if __name__ == '__main__':

    main()
