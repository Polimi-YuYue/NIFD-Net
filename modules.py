#!/usr/bin/env python
# _*_coding:utf-8 _*_
# @Time: 2026/1/8 15:19
# @Author: Yue Yu
# @School: Politecnico di Milano
# @Email: yyu41474@gmail.com
# @Filmname: modules.py
# @Software: PyCharm
# @Theme: Fault Diagnosis
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# --- Feature Encoder (ConvNet) ---
class ConvNet(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, dim // 4, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(dim // 4),
            nn.LeakyReLU(),
            nn.MaxPool2d(2, stride=2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(dim // 4, dim // 4, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(dim // 4),
            nn.LeakyReLU(),
            nn.Conv2d(dim // 4, dim // 2, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(dim // 2),
            nn.LeakyReLU(),
            nn.MaxPool2d(2, stride=2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(dim // 2, dim // 2, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(dim // 2),
            nn.LeakyReLU(),
            nn.Conv2d(dim // 2, dim // 1, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(dim // 1),
            nn.LeakyReLU(),
            nn.MaxPool2d(2, stride=2)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(dim // 1, dim * 2, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(dim * 2),
            nn.LeakyReLU(),
            nn.Conv2d(dim * 2, dim, kernel_size=(1, 1)),
            nn.BatchNorm2d(dim),
            nn.LeakyReLU(),
            nn.AvgPool2d(2, stride=2)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        return x

# --- Vector Quantization (Codebook) ---
class Codebook(nn.Module):
    def __init__(self, num_codebook_vectors, dim, beta, use_norm=False):
        super(Codebook, self).__init__()
        self.num_codebook_vectors = num_codebook_vectors
        self.latent_dim = dim
        self.beta = beta
        self.norm = lambda x: F.normalize(x, dim=-1) if use_norm else x
        self.embedding = nn.Embedding(self.num_codebook_vectors, self.latent_dim)
        self.embedding.weight.data.uniform_()

    def forward(self, z, return_prob=False):
        # z shape: (bz, n, d)
        z_flattened_norm = self.norm(z.contiguous().view(-1, self.latent_dim))
        embedding_norm = self.norm(self.embedding.weight)

        d = torch.cdist(z_flattened_norm, embedding_norm, p=2)
        min_encoding_indices = torch.argmin(d, dim=1)
        z_q = self.embedding(min_encoding_indices).view(z.shape)

        z_norm = self.norm(z)
        z_q_norm = self.norm(z_q)

        loss = self.beta * torch.mean((z_q_norm.detach() - z_norm) ** 2) + torch.mean((z_q_norm - z_norm.detach()) ** 2)
        z_q_norm = z + (z_q_norm - z).detach()

        return z_q_norm, min_encoding_indices, loss

    def get_vec_from_logits(self, logits, temp=1.):
        prob_one_hot = F.gumbel_softmax(logits, tau=temp, hard=False, dim=1)
        embed = self.embedding.weight
        vec = prob_one_hot @ embed
        return vec

# --- Transformer (GPT) ---
class GPTConfig:
    embd_pdrop = 0.1
    resid_pdrop = 0.1
    attn_pdrop = 0.1

    def __init__(self, vocab_size, block_size, **kwargs):
        self.vocab_size = vocab_size
        self.block_size = block_size
        for k, v in kwargs.items():
            setattr(self, k, v)

class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.key = nn.Linear(config.n_embd, config.n_embd)
        self.query = nn.Linear(config.n_embd, config.n_embd)
        self.value = nn.Linear(config.n_embd, config.n_embd)
        self.attn_drop = nn.Dropout(config.attn_pdrop)
        self.resid_drop = nn.Dropout(config.resid_pdrop)
        self.proj = nn.Linear(config.n_embd, config.n_embd)
        mask = torch.tril(torch.ones(config.block_size, config.block_size))
        if hasattr(config, "n_unmasked"):
            mask[:config.n_unmasked, :config.n_unmasked] = 1
        self.register_buffer("mask", mask.view(1, 1, config.block_size, config.block_size))
        self.n_head = config.n_head

    def forward(self, x, layer_past=None):
        B, T, C = x.size()
        k = self.key(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = self.query(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = self.value(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        present = torch.stack((k, v))
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        if layer_past is None:
            att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_drop(self.proj(y))
        return y, present

class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.mlp = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd),
            nn.GELU(),
            nn.Linear(4 * config.n_embd, config.n_embd),
            nn.Dropout(config.resid_pdrop),
        )

    def forward(self, x, layer_past=None):
        attn, present = self.attn(self.ln1(x), layer_past=layer_past)
        x = x + attn
        x = x + self.mlp(self.ln2(x))
        return x, present

class GPT(nn.Module):
    def __init__(self, vocab_size, block_size, n_layer=6, n_head=4, n_embd=64,
                 embd_pdrop=0.1, resid_pdrop=0.1, attn_pdrop=0.1, n_unmasked=0):
        super().__init__()
        config = GPTConfig(vocab_size=vocab_size, block_size=block_size,
                           embd_pdrop=embd_pdrop, resid_pdrop=resid_pdrop, attn_pdrop=attn_pdrop,
                           n_layer=n_layer, n_head=n_head, n_embd=n_embd,
                           n_unmasked=n_unmasked)
        self.tok_emb = nn.Embedding(config.vocab_size, config.n_embd)
        self.pos_emb = nn.Parameter(torch.zeros(1, config.block_size, config.n_embd))
        self.drop = nn.Dropout(config.embd_pdrop)
        self.blocks = nn.Sequential(*[Block(config) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.block_size = config.block_size
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, idx, embeddings=None):
        token_embeddings = self.tok_emb(idx)
        t = token_embeddings.shape[1]
        assert t <= self.block_size, "Cannot forward, model block size is exhausted."
        position_embeddings = self.pos_emb[:, :t, :]
        x = token_embeddings + position_embeddings
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.head(x)
        return logits, None