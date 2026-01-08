#!/usr/bin/env python
# _*_coding:utf-8 _*_
# @Time: 2026/1/8 15:20
# @Author: Yue Yu
# @School: Politecnico di Milano
# @Email: yyu41474@gmail.com
# @Filmname: pretrain.py
# @Software: PyCharm
# @Theme: Fault Diagnosis

import argparse
import os
import time
import datetime
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from munch import Munch
from monai.data import decollate_batch
from monai.metrics import CumulativeAverage, ConfusionMatrixMetric, ROCAUCMetric
from monai.transforms import Compose, Activations, AsDiscrete
from einops import rearrange

import utils
from utils import netDataset
from models.modules import ConvNet, Codebook


def get_args():
    parser = argparse.ArgumentParser(description="Pretrain Single Branch (SVS or NCS) for NIFD-Net")
    parser.add_argument('--modality', type=str, required=True, choices=['SVS', 'NCS'],
                        help='Modality to train: SVS (Vibration) or NCS (Acoustic)')
    parser.add_argument('--data_root', type=str, required=True,
                        help='Path to dataset root containing train/test folders')
    parser.add_argument('--output_dir', type=str, default='./checkpoints', help='Directory to save checkpoints')
    parser.add_argument('--epochs', type=int, default=7)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--img_size', type=int, default=32)
    parser.add_argument('--beta', type=float, default=0.25, help='Codebook commitment loss weight')
    return parser.parse_args()


class SingleBranchSolver(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Build Models
        self.nets = Munch(
            CNN=ConvNet(64),
            CODEBOOK=Codebook(128, 64, args.beta),
            CLS=nn.Linear(64, 13)
        )

        # Setup Logger
        self.ckpt_dir = os.path.join(args.output_dir, f'VQCNN_{args.modality}_vis')
        utils.mkdir(self.ckpt_dir)
        self.logger = utils.Logger(self.ckpt_dir, log_name=f'{args.modality.lower()}_log.txt')

        # Init weights
        for name, module in self.nets.items():
            module.to(self.device)
            if name != 'CODEBOOK':
                module.apply(utils.he_init)

        # Optims
        self.optims = Munch()
        self.optims.CODEBOOK = torch.optim.Adam(self.nets.CODEBOOK.parameters(), lr=args.lr * 100)
        self.optims.CNN = torch.optim.AdamW(self.nets.CNN.parameters(), lr=args.lr, weight_decay=1e-4)
        self.optims.CLS = torch.optim.AdamW(self.nets.CLS.parameters(), lr=args.lr, weight_decay=1e-4)

        # Checkpoint Saver
        self.ckpt_saver = utils.CheckpointIO(
            os.path.join(self.ckpt_dir, f'best_{args.modality}_nets.ckpt'), **self.nets
        )

        # Metrics
        self.metrics = Munch(
            loss_cls=CumulativeAverage(),
            loss_q=CumulativeAverage(),
            cm=ConfusionMatrixMetric(metric_name=["accuracy", "f1 score"], include_background=False, reduction='mean'),
            auc=ROCAUCMetric(average='micro')
        )
        self.transforms = Munch(
            post_pred=Compose([Activations(softmax=True)]),
            post_pred_argmax=Compose([AsDiscrete(argmax=True, to_onehot=13)]),
            post_label=Compose([AsDiscrete(to_onehot=13)])
        )

    def train_step(self, img, label, epoch, warmup_epochs=5):
        logits, indices, loss_cls, loss_q = self._forward(img, label, epoch, warmup_epochs)

        loss = loss_cls + loss_q if epoch >= warmup_epochs else loss_cls

        for opt in self.optims.values(): opt.zero_grad()
        loss.backward()
        self.optims.CNN.step()
        self.optims.CLS.step()
        if epoch >= warmup_epochs: self.optims.CODEBOOK.step()

        return logits, loss_cls, loss_q

    def _forward(self, img, label, epoch, warmup_epochs):
        feats = self.nets.CNN(img)

        if epoch >= warmup_epochs:
            feats = rearrange(feats, 'b c h w -> b (h w) c')
            feats_q, indices, loss_q = self.nets.CODEBOOK(feats)
            feats_q = rearrange(feats_q, 'b n c -> b c n')
            feats_pool = F.adaptive_max_pool1d(feats_q, 1).view(img.shape[0], -1)
        else:
            loss_q = torch.tensor(0.0, device=self.device)
            indices = None
            feats_pool = F.adaptive_max_pool2d(feats, 1).view(img.shape[0], -1)

        logits = self.nets.CLS(feats_pool)
        loss_cls = F.cross_entropy(logits, label)
        return logits, indices, loss_cls, loss_q

    def run_epoch(self, loader, mode, epoch):
        if mode == 'train':
            [m.train() for m in self.nets.values()]
        else:
            [m.eval() for m in self.nets.values()]

        for data in loader:
            img, label = data[0].to(self.device), data[1].to(self.device)

            if mode == 'train':
                logits, _, loss_cls, loss_q = self.train_step(img, label, epoch)
            else:
                with torch.no_grad():
                    logits, _, loss_cls, loss_q = self._forward(img, label, epoch, 5)

            # Record metrics
            self.metrics.loss_cls.append(loss_cls)
            self.metrics.loss_q.append(loss_q)

            y_onehot = [self.transforms.post_label(i) for i in decollate_batch(label)]
            y_pred = [self.transforms.post_pred(i) for i in decollate_batch(logits)]
            y_pred_argmax = [self.transforms.post_pred_argmax(i) for i in decollate_batch(logits)]

            self.metrics.cm(y_pred=y_pred_argmax, y=y_onehot)
            self.metrics.auc(y_pred=y_pred, y=y_onehot)

        # Aggregate
        res = {
            'loss_cls': self.metrics.loss_cls.aggregate().item(),
            'loss_q': self.metrics.loss_q.aggregate().item(),
            'acc': self.metrics.cm.aggregate()[0].item(),
            'f1': self.metrics.cm.aggregate()[1].item(),
            'auc': self.metrics.auc.aggregate()
        }

        # Reset
        self.metrics.loss_cls.reset()
        self.metrics.loss_q.reset()
        self.metrics.cm.reset()
        self.metrics.auc.reset()
        return res

    def fit(self, train_loader, val_loader):
        best_acc = 0.0
        start_time = time.time()

        for epoch in range(self.args.epochs):
            self.run_epoch(train_loader, 'train', epoch)
            res = self.run_epoch(val_loader, 'val', epoch)

            elapsed = str(datetime.timedelta(seconds=time.time() - start_time))[:-7]
            self.logger.print_message(
                f"Epoch {epoch + 1} | Time: {elapsed} | "
                f"Loss: {res['loss_cls']:.4f} | Acc: {res['acc']:.4f} | F1: {res['f1']:.4f} | AUC: {res['auc']:.4f}"
            )

            if epoch >= 5 and res['acc'] > best_acc:
                best_acc = res['acc']
                self.ckpt_saver.save()


if __name__ == '__main__':
    args = get_args()

    # Data Setup
    train_dir = os.path.join(args.data_root, 'train')
    test_dir = os.path.join(args.data_root, 'test')

    train_ds = netDataset(train_dir, os.listdir(train_dir), args.img_size)
    test_ds = netDataset(test_dir, os.listdir(test_dir), args.img_size)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0, drop_last=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size // 2, shuffle=False, num_workers=0, drop_last=True)

    solver = SingleBranchSolver(args)
    solver.fit(train_loader, test_loader)