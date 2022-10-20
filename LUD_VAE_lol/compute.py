import numpy as np
from utils import utils_image as util
from collections import OrderedDict
import os.path

def main():

    # ----------------------------------------
    # E_path, H_path
    # ----------------------------------------

    E_path = ''
    H_path = ''

    test_results = OrderedDict()
    test_results['psnr'] = []
    test_results['ssim'] = []

    E_paths = util.get_image_paths(E_path)
    H_paths = util.get_image_paths(H_path)

    for idx, img in enumerate(E_paths):

        img_name, ext = os.path.splitext(os.path.basename(img))

        img_E = util.imread_uint(img, n_channels=3)
        img_H = util.imread_uint(H_paths[idx], n_channels=3)

        # --------------------------------
        # PSNR and SSIM
        # --------------------------------

        psnr = util.calculate_psnr(img_E, img_H, border=0)
        ssim = util.calculate_ssim(img_E, img_H, border=0)
        test_results['psnr'].append(psnr)
        test_results['ssim'].append(ssim)
        print('{:s} - PSNR: {:.2f} dB; SSIM: {:.4f}.'.format(img_name+ext, psnr, ssim))

    ave_psnr = sum(test_results['psnr']) / len(test_results['psnr'])
    ave_ssim = sum(test_results['ssim']) / len(test_results['ssim'])

    print('Average PSNR/SSIM(RGB) - PSNR: {:.2f} dB; SSIM: {:.4f}'.format(ave_psnr, ave_ssim))

if __name__ == '__main__':
    main()
