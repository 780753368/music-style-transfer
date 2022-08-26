import tensorflow as tf
import torch
import math
import numpy as np
import torch
import torch.nn as nn

from timm.models.vision_transformer import PatchEmbed, Block

img_size=64
patch_size=16
in_chans=1
embed_dim=168
depth=24
num_heads=16
decoder_embed_dim=168
decoder_depth=8
decoder_num_heads=16
mlp_ratio=4
norm_layer=nn.LayerNorm
norm_pix_loss=False
patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
num_patches = 64
N=16
L=64
H=84
D=2
cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim), requires_grad=False)  # fixed sin-cos embedding

decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)

mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))

decoder_pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, decoder_embed_dim), requires_grad=False) 


def patchify(imgs):
    """
    imgs: (N, 3, H, W)
    x: (N, L, patch_size**2 *3)
    """

    ##imgs = torch.einsum('nhwc->nchw', imgs)
    ##p = patch_embed.patch_size[0]
    ##print('p:',p)
    ##assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0

    ##h = w = imgs.shape[2] // p
    ##x = imgs.reshape(shape=(imgs.shape[0], 3, h, p, w, p))
    ##print(x.shape)
    ##x = torch.einsum('nchpwq->nhwpqc', x)
    ##print(x.shape)
    x = imgs.reshape(shape=(imgs.shape[0], L, H*2))
    ##print(x.shape)
    return x

def unpatchify(x):
    """
    x: (N, L, patch_size**2 *3)
    imgs: (N, 3, H, W)
    """
    p = patch_embed.patch_size[0]
    ##h = w = int(x.shape[1]**.5)
    ##assert h * w == x.shape[1]
    x = x.reshape(shape=(x.shape[0],L,H,D))
    ##x = torch.einsum('nhwpqc->nchpwq', x)
    ##imgs = x.reshape(shape=(x.shape[0], L, H , D))
    ##print('out:',imgs,imgs.shape)
    ##imgs = torch.einsum('nchw->nhwc', imgs)
    return x

def random_masking(x, mask_ratio):
    """
    Perform per-sample random masking by per-sample shuffling.
    Per-sample shuffling is done by argsort random noise.
    x: [N, L, D], sequence
    """
    len_keep = int(L * (1 - mask_ratio))
    
    noise = torch.rand(N, L)  # noise in [0, 1]
    ##print('noise',noise,noise.shape)
        # sort noise for each sample
    ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
    ##print('ids_shuffle:',ids_shuffle,ids_shuffle.shape)
    ids_restore = torch.argsort(ids_shuffle, dim=1)
    ##print('ids_restore:',ids_restore,ids_restore.shape)
        # keep the first subset
    ids_keep = ids_shuffle[:, :len_keep]
    ##print('ids_keep:',ids_keep,ids_keep.shape)
    ##print('index:',ids_keep.unsqueeze(-1).repeat(1, 1, D),ids_keep.unsqueeze(-1).repeat(1, 1, D).shape)
    x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D*84))
        # generate the binary mask: 0 is keep, 1 is remove
    mask = torch.ones([N, L])
    mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
    mask = torch.gather(mask, dim=1, index=ids_restore)

    return x_masked, mask, ids_restore
x=torch.rand(16,64,84,2)
x=patchify(x)
print(x.shape)
x,mask,ids=random_masking(x, 0.75)
print(x.shape)
cls_token = cls_token + pos_embed[:, :1, :]
cls_tokens = cls_token.expand(x.shape[0], -1, -1)
x = torch.cat((cls_tokens, x), dim=1)
print(x.shape)
mask_tokens = mask_token.repeat(x.shape[0], ids.shape[1] + 1 - x.shape[1], 1)
x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)
x_ = torch.gather(x_, dim=1, index=ids.unsqueeze(-1).repeat(1, 1, x.shape[2]))
print(x_.shape,x.shape)
x = torch.cat([x[:, :1, :], x_], dim=1)
print(x.shape,decoder_pos_embed.shape)
x = x + decoder_pos_embed
print(x.shape)
x = x[:, 1:, :]
print(x.shape)
x=unpatchify(x)
print(x,x.shape)