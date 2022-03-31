import os
import torch
import torch.nn as nn
from torch.nn import init
from torch.nn.parallel import DataParallel  # , DistributedDataParallel
import functools
from collections import OrderedDict
from torch.optim import Adam
from torch.optim import lr_scheduler
import numpy as np


class LUDVI():
    def __init__(self, opt):
        self.opt = opt                         # opt
        self.save_dir = opt['path']['models']  # save models
        self.device = torch.device('cuda' if opt['gpu_ids'] is not None else 'cpu')
        self.is_train = opt['is_train']        # training or not
        self.schedulers = []                   # schedulers
        
        self.model = define_net().to(self.device)
        self.model = DataParallel(self.model)
        
    """
    # ----------------------------------------
    # Preparation before training with data
    # Save model during training
    # ----------------------------------------
    """

    def init_train(self):
        self.opt_train = self.opt['train']    # training option
        self.load()                           # load model
        self.model.train()
        self.define_optimizer()               # define optimizer
        self.define_scheduler()               # define scheduler
        self.log_dict = OrderedDict()         # log 


    def load(self):
        load_path = self.opt['path']['pretrained_net']
        if load_path is not None:
            print('Loading model [{:s}] ...'.format(load_path))
            self.load_network(load_path, self.model)


    def save(self, iter_label):
        self.save_network(self.save_dir, self.model, iter_label)


    def define_optimizer(self):
        optim_params = [p for p in self.model.parameters()]
        self.optimizer = Adam(optim_params, lr=self.opt_train['optimizer_lr'], weight_decay=0)


    def define_scheduler(self):
        self.schedulers.append(lr_scheduler.MultiStepLR(self.optimizer,
                                                        self.opt_train['scheduler_milestones'],
                                                        self.opt_train['scheduler_gamma']
                                                        ))

    def feed_data(self, data):
        self.img = data['data'].to(self.device)
        self.img_c = data['data_c'].to(self.device)
        self.label = data['label'].to(self.device)


    def optimize_parameters(self, current_step):
        self.optimizer.zero_grad()
        mseloss = nn.MSELoss().to(self.device)
        
        rec_loss, kl_loss = self.model(self.img, self.img_c, self.label)
        
        if self.opt_train['KL_anneal'] == 'linear':
            kl_weight = min(current_step/self.opt_train['KL_anneal_maxiter'], self.opt_train['KL_weight'])
        else:
            kl_weight = self.opt_train['KL_weight']
        
        loss1 = rec_loss.mean()
        loss2 = kl_weight * kl_loss.mean()
        
        loss = loss1 + loss2
        loss.backward()
        
        self.optimizer.step()
        
        self.log_dict['loss'] = loss.item()
        self.log_dict['reconstruction_loss'] = loss1.item()
        self.log_dict['kl_loss'] = loss2.item()


    def test(self):
        self.model.eval()
        
        if isinstance(self.model, nn.DataParallel):
            model = self.model.module
        else:
            model = self.model
        
        if 'normalize' in self.opt['datasets']['train']:
            normalize = True
        
            mean_noisy = np.asarray(self.opt['datasets']['train']['normalize']['mean_noisy'])/255
            std_noisy = np.asarray(self.opt['datasets']['train']['normalize']['std_noisy'])/255
            mean_clean = np.asarray(self.opt['datasets']['train']['normalize']['mean_clean'])/255
            std_clean = np.asarray(self.opt['datasets']['train']['normalize']['std_clean'])/255
            
            mean_noisy = torch.tensor(mean_noisy).reshape(1,3,1,1).to(self.device) 
            std_noisy = torch.tensor(std_noisy).reshape(1,3,1,1).to(self.device) 
            mean_clean = torch.tensor(mean_clean).reshape(1,3,1,1).to(self.device) 
            std_clean = torch.tensor(std_clean).reshape(1,3,1,1).to(self.device) 
            
            denormalize = lambda x: (((x - mean_clean)/std_clean)*std_noisy) + mean_noisy
            
        else:
            normalize = False
        
        with torch.no_grad():
            if normalize and self.label.item() == 1:
                self.img_t = model.translate(self.img, self.img_c, self.label)
            elif normalize and self.label.item() == 0:
                self.img_t = model.translate(self.img, self.img_c, self.label)
                self.img_t = denormalize(self.img_t)
            else: 
                self.img_t = model.translate(self.img, self.img_c, self.label)

        self.model.train()


    def current_log(self):
        return self.log_dict


    def current_visuals(self):
        out_dict = OrderedDict()
        out_dict['img'] = self.img.detach()[0].float().cpu()
        out_dict['label'] = self.label.detach().cpu().item()
        out_dict['img_t'] = self.img_t.detach()[0].float().cpu()
        return out_dict


    def current_results(self, need_H=True):
        out_dict = OrderedDict()
        out_dict['img'] = self.img.detach()[0].float().cpu()
        out_dict['label'] = self.label.detach().cpu().item()
        out_dict['img_t'] = self.img_t.detach()[0].float().cpu()
        return out_dict


    def update_learning_rate(self, n):
        for scheduler in self.schedulers:
            scheduler.step(n)


    def current_learning_rate(self):
        return self.schedulers[0].get_lr()[0]


    """
    # ----------------------------------------
    # Information of net
    # ----------------------------------------
    """

    def print_network(self):
        msg = self.describe_network(self.model)
        print(msg)

    # ----------------------------------------
    # print params
    # ----------------------------------------
    def print_params(self):
        msg = self.describe_params(self.model)
        print(msg)

    # ----------------------------------------
    # network information
    # ----------------------------------------
    def info_network(self):
        msg = self.describe_network(self.model)
        return msg

    # ----------------------------------------
    # params information
    # ----------------------------------------
    def info_params(self):
        msg = self.describe_params(self.model)
        return msg

    # ----------------------------------------
    # network name and number of parameters
    # ----------------------------------------
    
    def describe_network(self, model):
        if isinstance(model, nn.DataParallel):
            model = model.module
            
        msg = '\n'
        msg += 'Networks name: {}'.format(model.__class__.__name__) + '\n'
        msg += 'Params number: {}'.format(sum(map(lambda x: x.numel(), model.parameters()))) + '\n'
        msg += 'Net structure:\n{}'.format(str(model)) + '\n'
        
        return msg

    # ----------------------------------------
    # parameters description
    # ----------------------------------------
    def describe_params(self, model):
        if isinstance(model, nn.DataParallel):
            model = model.module
            
        msg = '\n'
        msg += ' | {:^6s} | {:^6s} | {:^6s} | {:^6s} || {:<20s}'.format('mean', 'min', 'max', 'std', 'shape', 'param_name') + '\n'
        for name, param in model.state_dict().items():
            if not 'num_batches_tracked' in name:
                v = param.data.clone().float()
                msg += ' | {:>6.3f} | {:>6.3f} | {:>6.3f} | {:>6.3f} | {} || {:s}'.format(v.mean(), v.min(), v.max(), v.std(), v.shape, name) + '\n'
        
        return msg

    """
    # ----------------------------------------
    # Save prameters
    # Load prameters
    # ----------------------------------------
    """
    # ----------------------------------------
    # save the state_dict of the network
    # ----------------------------------------
    def save_network(self, save_dir, model, iter_label):
        save_filename = '{}.pth'.format(iter_label)
        save_path = os.path.join(save_dir, save_filename)
        if isinstance(model, nn.DataParallel):
            model = model.module
            
        model_state_dict = model.state_dict()
        
        for key, param in model_state_dict.items():
            model_state_dict[key] = param.cpu()
        
        states = model_state_dict
        torch.save(states, save_path)


    # ----------------------------------------
    # load the state_dict of the network
    # ----------------------------------------
    def load_network(self, load_path, model, strict=True):
        if isinstance(model, nn.DataParallel):
            model = model.module
            
        states = torch.load(load_path)
        model.load_state_dict(states, strict=strict)


def define_net():
    from models.network_ludvae import LUDVAE
    net = LUDVAE()
    return net
    

