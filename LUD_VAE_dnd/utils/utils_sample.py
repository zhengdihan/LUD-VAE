# -*- coding: utf-8 -*-
"""
Created on Mon Apr  5 14:19:14 2021

@author: SonataCube
"""
import torch
import numpy as np

# def get_sigmas(sigma_begin, sigma_end, num_classes, sigma_dist):
#     if sigma_dist == 'geometric':
#         sigmas = torch.tensor(
#             np.exp(np.linspace(np.log(sigma_begin), np.log(sigma_end),
#                                num_classes))).float()
#     elif sigma_dist == 'uniform':
#         sigmas = torch.tensor(
#             np.linspace(sigma_begin, sigma_end, num_classes)
#         ).float()
# 
#     else:
#         raise NotImplementedError('sigma distribution not supported')
#     return sigmas

def get_sigmas(sigma_begin, sigma_end, num_classes, sigma_dist):
    if sigma_dist == 'geometric':
        sigmas = np.exp(np.linspace(np.log(sigma_begin), np.log(sigma_end), num_classes))
    elif sigma_dist == 'uniform':
        sigmas = np.linspace(sigma_begin, sigma_end, num_classes)
    else:
        raise NotImplementedError('sigma distribution not supported')

    return sigmas
    
def get_sigmas_multi_stage(sigma_milestones, num_classes_each, sigma_dist):
    sigmas = np.array([])
    
    for i in range(len(sigma_milestones)-1):
        sigma_begin = sigma_milestones[i]
        sigma_end = sigma_milestones[i+1]
        
        if i == 0:
            t = get_sigmas(sigma_begin, sigma_end, num_classes_each, sigma_dist)
        else:
            t = get_sigmas(sigma_begin, sigma_end, num_classes_each+1, sigma_dist)
            
        if i != 0:
            t = t[1:]
        sigmas = np.concatenate((sigmas, t))
    return sigmas

def get_sigmas_dpir(modelSigma1=49.0, modelSigma2=2.55, iter_num=15, w=1.0):
    modelSigmaS = np.logspace(np.log10(modelSigma1), np.log10(modelSigma2), iter_num).astype(np.float32)
    modelSigmaS_lin = np.linspace(modelSigma1, modelSigma2, iter_num).astype(np.float32)
    sigmas = (modelSigmaS*w+modelSigmaS_lin*(1-w))/255.
    return sigmas