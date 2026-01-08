#!/usr/bin/env python
# _*_coding:utf-8 _*_
# @Time: 2026/1/8 15:20
# @Author: Yue Yu
# @School: Politecnico di Milano
# @Email: yyu41474@gmail.com
# @Filmname: train.py
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
from einops import rearrange
from monai.data import decollate_batch
from monai.metrics import CumulativeAverage, ConfusionMatrixMetric, ROCAUCMetric
from monai.transforms import Compose, Activations, AsDiscrete

import utils
from utils import netDataset
from models.nifd_net import NIFDNet


def get_args():
    parser = argparse.ArgumentParser(description="NIFD-Net Multimodal Training")
    parser.add_argument('--svs_data', type=str, required=True, help='Path to Vibration data root')
    parser.add_argument('--ncs_data', type=str, required=True, help='Path to Acoustic data root')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints', help='Dir containing pretrained models')
    parser.add_argument('--epochs_stage2', type=int, default=100)
    parser.add_argument('--epochs_stage3', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--img_size', type=int, default=32)
    return parser.parse_args()


class NIFDSolver(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Build NIFD-Net
        self.model = NIFDNet()
        self.model.to(self.device)
        self.nets = self.model.nets  # Access munch wrapper

        # Setup Logger
        self.save_dir = os.path.join(args.checkpoint_dir, 'NIFD-Net_vis')
        utils.mkdir(self.save_dir)
        self.logger = utils.Logger(self.save_dir)

        # Optims
        self.optims = Munch()
        self.optims.Trans = torch.optim.AdamW(self.nets.Trans.parameters(), lr=args.lr, weight_decay=1e-2)
        self.optims.CLS = torch.optim.AdamW(self.nets.CLS.parameters(), lr=args.lr, weight_decay=1e-2)

        # Checkpoint IO
        self.ckpt_io = Munch(
            Trans=utils.CheckpointIO(os.path.join(self.save_dir, 'best_Trans_nets.ckpt'), Trans=self.nets.Trans),
            CLS=utils.CheckpointIO(os.path.join(self.save_dir, 'best_CLS_nets.ckpt'), CLS=self.nets.CLS)
        )

        # Metrics & Transforms
        self.metrics = Munch(
            loss=CumulativeAverage(),
            cm=ConfusionMatrixMetric(metric_name=["accuracy", "f1 score"], include_background=False, reduction='mean'),
            auc=ROCAUCMetric(average='micro')
        )
        self.transforms = Munch(
            post_pred=Compose([Activations(softmax=True)]),
            post_pred_argmax=Compose([AsDiscrete(argmax=True, to_onehot=13)]),
            post_label=Compose([AsDiscrete(to_onehot=13)])
        )

    def load_pretrained(self):
        # Load SVS
        path_svs = os.path.join(self.args.checkpoint_dir, 'VQCNN_SVS_vis', 'best_SVS_nets.ckpt')
        path_ncs = os.path.join(self.args.checkpoint_dir, 'VQCNN_NCS_vis', 'best_NCS_nets.ckpt')

        if not os.path.exists(path_svs) or not os.path.exists(path_ncs):
            raise FileNotFoundError("Pretrained SVS or NCS checkpoints not found. Run pretrain.py first.")

        print(f"Loading SVS from {path_svs}")
        svs_dict = torch.load(path_svs, map_location=self.device)
        self.nets.SVSCNN.load_state_dict(svs_dict['CNN'])
        self.nets.SVSCODEBOOK.load_state_dict(svs_dict['CODEBOOK'])
        self.nets.SVSCLS.load_state_dict(svs_dict['CLS'])

        print(f"Loading NCS from {path_ncs}")
        ncs_dict = torch.load(path_ncs, map_location=self.device)
        self.nets.NCSCNN.load_state_dict(ncs_dict['CNN'])
        self.nets.NCSCODEBOOK.load_state_dict(ncs_dict['CODEBOOK'])
        self.nets.NCSCLS.load_state_dict(ncs_dict['CLS'])

    def freeze_all_except(self, target_name):
        for name, module in self.nets.items():
            requires_grad = (name == target_name)
            for param in module.parameters():
                param.requires_grad = requires_grad
            if requires_grad:
                module.train()
            else:
                module.eval()

    def forward_trans(self, ncs_img, svs_img):
        # NCS -> Transformer -> Predict SVS Indices
        ncs_feats = self.nets.NCSCNN(ncs_img)
        ncs_feats = rearrange(ncs_feats, 'b c h w -> b (h w) c')
        _, ncs_indices, _ = self.nets.NCSCODEBOOK(ncs_feats)
        ncs_indices = ncs_indices.view(ncs_img.shape[0], -1)

        logits, _ = self.nets.Trans(ncs_indices)

        # Get SVS ground truth indices for loss
        with torch.no_grad():
            svs_feats = self.nets.SVSCNN(svs_img)
            svs_feats = rearrange(svs_feats, 'b c h w -> b (h w) c')
            _, svs_indices, _ = self.nets.SVSCODEBOOK(svs_feats)
            svs_indices = svs_indices.view(svs_img.shape[0], -1)

        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), svs_indices.reshape(-1))
        return logits, ncs_feats, loss

    def forward_cls(self, ncs_feats, svs_logits, label, stage):
        # Reconstruct SVS from logits
        svs_feats = self.nets.SVSCODEBOOK.get_vec_from_logits(svs_logits)

        # Pool
        svs_vec = F.adaptive_max_pool1d(rearrange(svs_feats, 'b n c -> b c n'), 1).view(svs_feats.shape[0], -1)
        ncs_vec = F.adaptive_max_pool1d(rearrange(ncs_feats, 'b n c -> b c n'), 1).view(ncs_feats.shape[0], -1)

        if stage == 2:
            # In stage 2, we just verify SVS reconstruction quality via the SVS classifier
            logits = self.nets.SVSCLS(svs_vec)
        else:
            # In stage 3, we concatenate and use final CLS
            concat = torch.cat([ncs_vec, svs_vec], dim=1)
            logits = self.nets.CLS(concat)

        loss = F.cross_entropy(logits, label)
        return logits, loss

    def run_epoch(self, loaders, epoch, stage, mode='train'):
        if mode == 'train':
            target = 'Trans' if stage == 2 else 'CLS'
            self.freeze_all_except(target)
        else:
            for m in self.nets.values(): m.eval()

        loader_svs, loader_ncs = loaders

        for (svs_img, label), (ncs_img, _) in zip(loader_svs, loader_ncs):
            svs_img, ncs_img, label = svs_img.to(self.device), ncs_img.to(self.device), label.to(self.device)

            # Forward
            trans_logits, ncs_feats, trans_loss = self.forward_trans(ncs_img, svs_img)
            cls_logits, cls_loss = self.forward_cls(ncs_feats, trans_logits, label, stage)

            if mode == 'train':
                self.optims.Trans.zero_grad()
                self.optims.CLS.zero_grad()

                if stage == 2:
                    loss_total = 0.01 * trans_loss + cls_loss
                    loss_total.backward()
                    self.optims.Trans.step()
                else:
                    cls_loss.backward()
                    self.optims.CLS.step()

            # Metrics (Tracking main loss based on stage)
            track_loss = trans_loss if stage == 2 else cls_loss
            self.metrics.loss.append(track_loss)

            y_onehot = [self.transforms.post_label(i) for i in decollate_batch(label)]
            y_pred = [self.transforms.post_pred(i) for i in decollate_batch(cls_logits)]
            y_pred_arg = [self.transforms.post_pred_argmax(i) for i in decollate_batch(cls_logits)]

            self.metrics.cm(y_pred=y_pred_arg, y=y_onehot)
            self.metrics.auc(y_pred=y_pred, y=y_onehot)

        res = {
            'loss': self.metrics.loss.aggregate().item(),
            'acc': self.metrics.cm.aggregate()[0].item(),
            'auc': self.metrics.auc.aggregate()
        }

        # Reset
        self.metrics.loss.reset()
        self.metrics.cm.reset()
        self.metrics.auc.reset()
        return res

    def fit(self, loaders):
        self.load_pretrained()
        train_loaders, test_loaders = loaders

        start_time = time.time()
        best_auc = 0.0

        # --- Stage 2: Train Transformer ---
        self.logger.print_message("Starting Stage 2: Transformer Training...")
        for epoch in range(self.args.epochs_stage2):
            self.run_epoch(train_loaders, epoch, stage=2, mode='train')
            res = self.run_epoch(test_loaders, epoch, stage=2, mode='eval')

            elapsed = str(datetime.timedelta(seconds=time.time() - start_time))[:-7]
            self.logger.print_message(
                f"Stage 2 Epoch {epoch + 1} | {elapsed} | Loss: {res['loss']:.4f} | Acc: {res['acc']:.4f} | AUC: {res['auc']:.4f}")

            if res['auc'] > best_auc:
                best_auc = res['auc']
                self.ckpt_io.Trans.save()

        # --- Stage 3: Train Classifier ---
        self.logger.print_message("Starting Stage 3: Classifier Training...")
        self.ckpt_io.Trans.load()  # Load best transformer
        best_auc = 0.0  # Reset for next stage

        for epoch in range(self.args.epochs_stage3):
            self.run_epoch(train_loaders, epoch, stage=3, mode='train')
            res = self.run_epoch(test_loaders, epoch, stage=3, mode='eval')

            elapsed = str(datetime.timedelta(seconds=time.time() - start_time))[:-7]
            self.logger.print_message(
                f"Stage 3 Epoch {epoch + 1} | {elapsed} | Loss: {res['loss']:.4f} | Acc: {res['acc']:.4f} | AUC: {res['auc']:.4f}")

            if res['auc'] > best_auc:
                best_auc = res['auc']
                self.ckpt_io.CLS.save()


if __name__ == "__main__":
    args = get_args()


    # Dataset Helpers
    def create_loaders(root, img_size, batch_size, shuffle):
        ds = netDataset(root, os.listdir(root), img_size)
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=0, drop_last=True)


    # SVS Loaders
    svs_train_l = create_loaders(os.path.join(args.svs_data, 'train'), args.img_size, args.batch_size, True)
    svs_test_l = create_loaders(os.path.join(args.svs_data, 'test'), args.batch_size // 2, False)

    # NCS Loaders
    ncs_train_l = create_loaders(os.path.join(args.ncs_data, 'train'), args.batch_size, True)
    ncs_test_l = create_loaders(os.path.join(args.ncs_data, 'test'), args.batch_size // 2, False)

    solver = NIFDSolver(args)
    solver.fit(loaders=((svs_train_l, ncs_train_l), (svs_test_l, ncs_test_l)))