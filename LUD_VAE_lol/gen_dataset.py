from PIL import Image
import numpy as np
import glob
import os
import random

def gamma_correction(image, gamma, c):
    return c * (image ** gamma)

image_roots = glob.glob('./trainB/*')

gamma_lb = 1.5
gamma_ub = 2.5
c_lb = 0.15
c_ub = 0.4

result_root = './trainB_ll_c0d15_0d4_gamma1d5_2d5/'
os.system('mkdir -p ' + result_root)

for image_root in image_roots:
    name = image_root.split('/')[-1]
    
    image = Image.open(image_root)
    image = np.array(image) / 255.0
    
    gamma = random.uniform(gamma_lb, gamma_ub)
    c = random.uniform(c_lb, c_ub)
    
    image_corrected = gamma_correction(image, gamma, c)
    image_corrected = np.uint8((image_corrected*255.0).round())
    
    image_corrected = Image.fromarray(image_corrected)
    
    image_corrected.save(result_root + name)
























