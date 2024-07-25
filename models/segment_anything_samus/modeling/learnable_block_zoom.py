# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
DETR Transformer class.

Copy-paste from torch.nn.Transformer with modifications:
    * positional encodings are passed in MHattention
    * extra LN at the end of encoder is removed
    * decoder returns a stack of activations from all decoding layers
"""

import torch
from torch import nn, Tensor
from torch.nn import functional as F

from einops import rearrange
import copy
from typing import Any, List, Tuple, Optional

from .image_encoder import ImageEncoderViT
from .mask_decoder import MaskDecoder
from .prompt_encoder import PromptEncoder
from .common import LayerNorm2d

class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x
  
class RB(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.in_layers = nn.Sequential(
            nn.GroupNorm(32, in_channels),
            nn.SiLU(),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        )

        self.out_layers = nn.Sequential(
            nn.GroupNorm(32, out_channels),
            nn.SiLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
        )

        if out_channels == in_channels:
            self.skip = nn.Identity()
        else:
            self.skip = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        h = self.in_layers(x)
        h = self.out_layers(h)
        return h + self.skip(x)
     
class SingleConv(nn.Module):
    """Downscaling with maxpool then double conv"""
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=1, bias=False),
            LayerNorm2d(out_channels),
            nn.GELU()
        )

    def forward(self, x):
        return self.conv(x)

import numpy as np
class LearableBlockZoom(nn.Module):
    def __init__(
        self,
        image_encoder: ImageEncoderViT,
        pixel_mean: List[float] = [123.675, 116.28, 103.53],
        pixel_std: List[float] = [58.395, 57.12, 57.375],
        d_model = 256,
        num_decoder_layers = 1, #6,
        dim_feedforward=2048,
        dropout=0.1,
        activation="relu",
        normalize_before=False,
        return_intermediate_dec=False
    ) -> None:
        super().__init__()
        self.image_encoder = image_encoder
        # learnable image encoder local encoder 확인
        # Conv
        self.local_patch_embed = nn.Sequential(
            nn.Conv2d(1, int(d_model), kernel_size=3, padding=1, bias=False),
            LayerNorm2d(int(d_model)),
            nn.GELU()
        )
        self.PH = nn.Sequential(
            RB(d_model * 2, d_model), RB(d_model, d_model)
        )
        # MLP
        self.bbox_embed = MLP(d_model, d_model, 4, 3)       # hidden_dim -> hidden_dim
        self.class_embed = nn.Linear(256, 1 + 1)            
        
        self.register_buffer("pixel_mean", torch.Tensor(pixel_mean).view(-1, 1, 1), False)
        self.register_buffer("pixel_std", torch.Tensor(pixel_std).view(-1, 1, 1), False)
        
        #! Encoder & Prompt Encoder freeze
        for param in self.image_encoder.parameters():
            param.requires_grad = False
    
    def local_view_image_encoder(self, imgs, pts):
        # 1. crop - N X (32x32x3) -> N x (32x32x256)
        bs, N, _, _ = pts[0].shape # b N 1 2
        tmp1 = torch.empty([0, N*256, 32, 32]).to(device=imgs.device, dtype=imgs.dtype)
        for _bs in range(bs):
            tmp0 = torch.empty([1, 0, 32, 32]).to(device=imgs.device, dtype=imgs.dtype)
            for _n in range(N):
                cx, cy = np.array(pts[0][_bs,_n, 0].cpu()).astype(np.uint32).tolist()
                tmp0 = torch.concat(
                    [tmp0, self.local_patch_embed(imgs[_bs, :, cy-16:cy+16, cx-16:cx+16][None])], dim=1)
            tmp1 = torch.concat([tmp1, tmp0], dim=0)
        # tmp1 = [bs, Nx256, 32, 32]        
        # 2. depth-wise conv
        
        # 3. split
        return imgs
    
    def forward(
        self, 
        imgs: torch.Tensor,
        pts: Tuple[torch.Tensor, torch.Tensor], # coords : [b n 2], labels : [b n]
    ) -> torch.Tensor:
        #** 1. pass image encoder (global)
        gimage = self.image_encoder(imgs)                   # [b c(256) h(32) w(32)]
        #** 2. pass image encoder (local)
        # limage = self.local_view_image_encoder(imgs, pts)   # [b cN h w]
        # limage = self.limage_encoder(imgs)                # [b c*N h w]
        #** 3. concat
        image = torch.concat([gimage, limage], dim = 1)     # [b 2c h w]
        #** 4. Conv
        output = self.PH(image)
        #** 5. prediction head
        outputs_class = self.class_embed(output)
        outputs_coord = self.bbox_embed(output).sigmoid()
        out = {'pred_logits': outputs_class, 'pred_boxes': outputs_coord}
        return out