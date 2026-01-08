#!/usr/bin/env python
# _*_coding:utf-8 _*_
# @Time: 2026/1/8 15:19
# @Author: Yue Yu
# @School: Politecnico di Milano
# @Email: yyu41474@gmail.com
# @Filmname: nifd_net.py
# @Software: PyCharm
# @Theme: Fault Diagnosis
import torch
import torch.nn as nn
from munch import Munch
from .modules import ConvNet, Codebook, GPT


class NIFDNet(nn.Module):
    def __init__(self, channel_dim=64, code_num=128, dim=64, beta=0.25, output_dim=64, class_num=13):
        super().__init__()

        # SVS (Vibration) Branch
        self.SVS_CNN = ConvNet(channel_dim)
        self.SVS_CODEBOOK = Codebook(code_num, dim, beta)
        self.SVS_CLS_Head = nn.Linear(output_dim, class_num)

        # NCS (Acoustic) Branch
        self.NCS_CNN = ConvNet(channel_dim)
        self.NCS_CODEBOOK = Codebook(code_num, dim, beta)
        self.NCS_CLS_Head = nn.Linear(output_dim, class_num)

        # Transformer for Cross-Modal Translation
        self.Trans = GPT(code_num, 64, n_layer=6, n_head=4, n_embd=64)

        # Final Fusion Classifier
        self.CLS = nn.Linear(64, class_num)

        # Munch wrapper for easy access like in original code
        self.nets = Munch(
            SVSCNN=self.SVS_CNN, SVSCODEBOOK=self.SVS_CODEBOOK, SVSCLS=self.SVS_CLS_Head,
            NCSCNN=self.NCS_CNN, NCSCODEBOOK=self.NCS_CODEBOOK, NCSCLS=self.NCS_CLS_Head,
            Trans=self.Trans, CLS=self.CLS
        )

    def forward(self):
        # The main logic is handled in the training loop functions
        pass