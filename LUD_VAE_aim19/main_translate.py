import os.path
import logging

import numpy as np
from collections import OrderedDict

import torch

from utils import utils_logger
from utils import utils_image as util
from utils.utils_eval import eval_split


def main():

    # ----------------------------------------
    # Preparation
    # ----------------------------------------

    model_name = 'ludvae'
    normalize = True

    noise_level = 15
    temperature = 1.0

    model_pool = 'trained_models/LUDVAE_models'             # fixed

    model_path = os.path.join(model_pool, model_name+'.pth')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.cuda.empty_cache()

    H_path = '/home/dihan/dataset/AIM-RWSR/train-clean-images/4x/'
    G_path = './datasets/AIM-RWSR/4x_degraded/'
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

    if normalize:
        mean_clean = torch.tensor([114.69150172, 111.67491841, 103.26490806]).reshape(1,3,1,1)/255
        mean_noisy = torch.tensor([120.96351636, 115.84200238, 104.9237979]).reshape(1,3,1,1)/255
        std_clean = torch.tensor([70.6325031, 67.0257846, 72.95698077]).reshape(1,3,1,1)/255
        std_noisy = torch.tensor([61.84091623, 59.79885696, 65.42669225]).reshape(1,3,1,1)/255

        mean_clean = mean_clean.to(device)
        mean_noisy = mean_noisy.to(device)
        std_clean = std_clean.to(device)
        std_noisy = std_noisy.to(device)

    for idx, img in enumerate(H_paths):

        # ------------------------------------
        # (1) img_H
        # ------------------------------------

        img_name, ext = os.path.splitext(os.path.basename(img))
        img_H = util.imread_uint(img, n_channels=3)
        img_hH = img_H.copy()

        img_H = util.uint2tensor4(img_H).to(device)
        img_hH = util.uint2tensor4(img_hH).to(device)
        img_hH = img_hH + torch.randn_like(img_hH) * noise_level / 255.0

        # ------------------------------------
        # (2) img_G
        # ------------------------------------

        label_H = torch.zeros(1, 1, 1, 1).long().to(device)

        img_G = model.translate(img_H, img_hH, label_H, temperature=temperature)

        if normalize:
            img_G = (((img_G - mean_clean)/std_clean)*std_noisy) + mean_noisy

        img_G = util.tensor2uint(img_G)

        # ------------------------------------
        # save results
        # ------------------------------------
        util.imsave(img_G, os.path.join(G_path, img_name + ext))

if __name__ == '__main__':

    main()
