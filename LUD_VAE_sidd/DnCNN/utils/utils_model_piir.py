# -*- coding: utf-8 -*-
import numpy as np
import torch
import torch.nn.functional as F
import random

def test_mode(model, L, sigma, min_size=256, n_avg=1):
    N, C, H, W = L.shape
    out = torch.zeros_like(L)
    
    pad_H = (min_size - H % min_size) if H % min_size else 0
    pad_W = (min_size - W % min_size) if W % min_size else 0
    
    L_pad = F.pad(L, (pad_W//2, pad_W - pad_W//2, pad_H//2, pad_H - pad_H//2), 'replicate')    
        
    for _ in range(n_avg):
        H_pad, W_pad = L_pad.shape[-2:]
        
        n_top_pad = random.randint(0, min_size-1)
        n_left_pad = random.randint(0, min_size-1)
        
        pad = (n_left_pad, min_size-n_left_pad, n_top_pad, min_size-n_top_pad)
        
        L_pad_pad = F.pad(L_pad, pad, "constant", 0)
        H_dpad, W_dpad = L_pad_pad.shape[-2:]
        
        H_div = (H_dpad // min_size)
        W_div = (W_dpad // min_size)
        
        L_pad_pad_patch = L_pad_pad.reshape(N, 3, H_div, min_size, W_div, min_size)
        L_pad_pad_patch = L_pad_pad_patch.permute(0, 2, 4, 1, 3, 5)
        L_pad_pad_patch = L_pad_pad_patch.reshape(-1, 3, min_size, min_size)
        out_pad_pad = model(L_pad_pad_patch, sigma)
        out_pad_pad = out_pad_pad.reshape(N, H_div, W_div, 3, min_size, min_size)
        out_pad_pad = out_pad_pad.permute(0, 3, 1, 4, 2, 5)
        out_pad_pad = out_pad_pad.reshape(N, 3, H_dpad, W_dpad)
        
        out_pad = out_pad_pad[:, :, n_top_pad:n_top_pad+H_pad, n_left_pad:n_left_pad+W_pad]
        out += out_pad[:, :, pad_H//2:pad_H//2+H, pad_W//2:pad_W//2+W]
    
    return out / n_avg


if __name__ == '__main__':

    class Net(torch.nn.Module):
        def __init__(self, in_channels=3, out_channels=3):
            super(Net, self).__init__()
            self.conv = torch.nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1)

        def forward(self, x, temp):
            x = self.conv(x)
            return x

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    model = Net()
    model = model.eval()
    x = torch.randn((2,3,400,400))
    torch.cuda.empty_cache()
    with torch.no_grad():
        y = test_mode(model, x, 20)
        print(y.shape)
