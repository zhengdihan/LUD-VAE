from dnd_submission_kit.py.dnd_denoise import denoise_srgb
from dnd_submission_kit.py.pytorch_wrapper import pytorch_denoiser
from dnd_submission_kit.py.bundle_submissions import bundle_submissions_srgb

import os.path
import torch
import os


def main():

    # ----------------------------------------
    # Preparation
    # ----------------------------------------

    model_name = 'dncnn'

    model_pool = 'model_zoo'             # fixed

    model_path = os.path.join(model_pool, model_name+'.pth')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.cuda.empty_cache()

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

    data_folder = '/home/dihan/dataset/dnd_2017/'
    out_folder = './dnd_submit/'
    
    os.system('mkdir -p '+ out_folder) 
    def model_restoration(x):
        return model(x)
    
    denoiser = pytorch_denoiser(model_restoration, use_cuda=True)
    denoise_srgb(denoiser, data_folder, out_folder)
    bundle_submissions_srgb(out_folder)

if __name__ == '__main__':

    main()
