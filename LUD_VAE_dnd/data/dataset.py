import random
import numpy as np
import torch
import torch.utils.data as data
import utils.utils_image as util


class UPMDataset(data.Dataset):
    def __init__(self, opt):
        super(UPMDataset, self).__init__()
        self.opt = opt
        self.n_channels = opt['n_channels'] if opt['n_channels'] else 3
        self.patch_size = self.opt['H_size'] if opt['H_size'] else 128
        self.dataroot = opt['dataroot']
        
        self.H_noise_level = opt['H_noise_level']
        self.L_noise_level = opt['L_noise_level']
        
        if 'normalize' in opt:
            self.normalize = True
            self.mean_clean = torch.tensor(opt['normalize']['mean_clean']).reshape(3,1,1)/255
            self.mean_noisy = torch.tensor(opt['normalize']['mean_noisy']).reshape(3,1,1)/255
            self.std_clean = torch.tensor(opt['normalize']['std_clean']).reshape(3,1,1)/255
            self.std_noisy = torch.tensor(opt['normalize']['std_noisy']).reshape(3,1,1)/255
        else:
            self.normalize = False
        
        self.paths = []
        self.labels = []
        
        for class_i, root in enumerate(self.dataroot):
            path = util.get_image_paths(root)
            
            n_max = opt['n_max']
            if n_max is not None:
                if n_max > 0:
                    path = path[:n_max]
                elif n_max < 0:
                    path = path[n_max:]
                else:
                    raise RuntimeError("Not implemented")
            
            self.paths += path
            self.labels += [class_i]*len(path)

    def __getitem__(self, index):
        
        path = self.paths[index]
        img = util.imread_uint(path, self.n_channels)
        img_c = img.copy()
        
        label = float(self.labels[index])
        label = torch.full([1, 1, 1], label).long()

        H, W = img.shape[:2]
        
        if self.opt['phase'] == 'train':
            # ---------------------------------
            # randomly crop the patch
            # ---------------------------------
            rnd_h = random.randint(0, max(0, H - self.patch_size))
            rnd_w = random.randint(0, max(0, W - self.patch_size))
            patch = img[rnd_h:rnd_h + self.patch_size, rnd_w:rnd_w + self.patch_size, :]
            patch_c = img_c[rnd_h:rnd_h + self.patch_size, rnd_w:rnd_w + self.patch_size, :]
            
            # ---------------------------------
            # augmentation - flip, rotate
            # ---------------------------------
            mode = np.random.randint(0, 8)
            patch = util.augment_img(patch, mode=mode)
            patch_c = util.augment_img(patch_c, mode=mode)
    
            # ---------------------------------
            # HWC to CHW, numpy(uint) to tensor
            # ---------------------------------
            img = util.uint2tensor3(patch)
            img_c = util.uint2tensor3(patch_c)
            
            if label.item() == 1:
                img_c = img_c + torch.randn_like(img_c) * self.L_noise_level / 255.0
            else:
                img_c = img_c + torch.randn_like(img_c) * self.H_noise_level / 255.0
            
        else:
            patch = img[H//2 - self.patch_size//2 : H//2 + self.patch_size//2, W//2 - self.patch_size//2 : W//2 + self.patch_size//2, :]
            patch_c = img_c[H//2 - self.patch_size//2 : H//2 + self.patch_size//2, W//2 - self.patch_size//2 : W//2 + self.patch_size//2, :]
            img = util.uint2tensor3(patch)
            img_c = util.uint2tensor3(patch_c)
            
            if label.item() == 1:
                img_c = img_c + torch.randn_like(img_c) * self.L_noise_level / 255.0
            else:
                img_c = img_c + torch.randn_like(img_c) * self.H_noise_level / 255.0
                
        if self.normalize and label.item() == 1:
            img = torch.clamp(((img - self.mean_noisy)/self.std_noisy)*self.std_clean + self.mean_clean, 0, 1)
            img_c = torch.clamp(((img_c - self.mean_noisy)/self.std_noisy)*self.std_clean + self.mean_clean, 0, 1)

        return {'data': img, 'data_c': img_c, 'label': label, 'path': path}

    def __len__(self):
        return len(self.paths)
