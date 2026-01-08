#!/usr/bin/env python
# _*_coding:utf-8 _*_
# @Time: 2026/1/8 15:19
# @Author: Yue Yu
# @School: Politecnico di Milano
# @Email: yyu41474@gmail.com
# @Filmname: utils.py
# @Software: PyCharm
# @Theme: Fault Diagnosis

import os
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data.dataset import Dataset
from torchvision import transforms

def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def print_network(network, name):
    num_params = sum(p.numel() for p in network.parameters())
    print(f"Number of parameters of {name}: {num_params}")

def he_init(module):
    if isinstance(module, (nn.Conv2d, nn.Linear)):
        nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')
        if module.bias is not None:
            nn.init.constant_(module.bias, 0)

class Logger():
    def __init__(self, log_dir, log_name='log.txt'):
        self.log_name = os.path.join(log_dir, log_name)
        with open(self.log_name, "a") as log_file:
            log_file.write(f'================ {self.log_name} ================\n')

    def print_message(self, msg):
        print(msg, flush=True)
        with open(self.log_name, 'a') as log_file:
            log_file.write(f'{msg}\n')

class CheckpointIO(object):
    def __init__(self, fname_template, **kwargs):
        self.fname_template = fname_template
        self.module_dict = kwargs

    def save(self):
        fname = self.fname_template
        print(f'Saving checkpoint into {fname}...')
        outdict = {name: module.state_dict() for name, module in self.module_dict.items()}
        torch.save(outdict, fname)

    def load(self):
        fname = self.fname_template
        assert os.path.exists(fname), f'{fname} does not exist!'
        print(f'Loading checkpoint from {fname}...')
        map_loc = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        module_dict = torch.load(fname, map_location=map_loc)
        for name, module in self.module_dict.items():
            module.load_state_dict(module_dict[name])

class netDataset(Dataset):
    def __init__(self, path, train_lines, size):
        super(netDataset, self).__init__()
        self.train_path = path
        self.train_lines = train_lines
        self.size = size
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.train_lines)

    def __getitem__(self, index):
        annotation_line = self.train_lines[index]
        name = os.path.join(self.train_path, annotation_line)
        try:
            jpg = Image.open(name).resize((self.size, self.size), Image.BICUBIC).convert("RGB")
            label = int(annotation_line.split('.')[0].split('_')[-1])
            if self.transform is not None:
                jpg = self.transform(jpg)
            return jpg, label
        except Exception as e:
            print(f"Error loading image {name}: {e}")
            return torch.zeros(3, self.size, self.size), 0