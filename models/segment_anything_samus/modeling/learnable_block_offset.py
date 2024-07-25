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
class LearableBlockOffset(nn.Module):
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
        self.convLayer = nn.Sequential(
            RB(d_model, d_model), RB(d_model, d_model)
        )
        self.sampling_offsets = nn.Linear(d_model, 2)
        # self.offset = nn.Conv2d(in_channels=d_model, out_channels=2, kernel_size=1, padding=0, bias=None)
        # MLP
        self.bbox_embed = MLP(d_model, d_model, 4, 3)       # hidden_dim -> hidden_dim
        self.class_embed = nn.Linear(d_model, 1 + 1)            
        
        self.register_buffer("pixel_mean", torch.Tensor(pixel_mean).view(-1, 1, 1), False)
        self.register_buffer("pixel_std", torch.Tensor(pixel_std).view(-1, 1, 1), False)
        
        #! Encoder & Prompt Encoder freeze
        for param in self.image_encoder.parameters():
            param.requires_grad = False
        self.num_layers = num_decoder_layers
        
    # def move(self, imgs, candiates):
    #     # 1. 해당하는 imgs 속 patch 조각 찾기
    #     patch_idx = (candiates // 8).to(torch.uint16)[:,:,0,:]  # [b c 2]
    #     bs, c, _ = patch_idx.shape
    #     tmp0 = torch.empty([0,c,256,1,1]).to(imgs.device)      # [bs*, c, 256, 1, 1]
    #     sampling_locations = torch.empty([0, 2]).to(imgs.device)
    #     for _bs in range(bs):
    #         tmp1 = torch.empty([0,256,1,1]).to(imgs.device)      # [c*, 256, 1, 1]
    #         for _c in range(c):
    #             iy, ix = np.array(patch_idx[_bs, _c].cpu()).tolist()
    #             tmp1 = torch.concat([tmp1, imgs[0, :, iy:iy+1, ix:ix+1][None]], dim=0)
    #             sampling_locations = torch.concat([sampling_locations, torch.tensor([[iy, ix]]).to(imgs.device)], dim=0)
    #         tmp0 = torch.concat([tmp0, tmp1[None]], dim=0)
    #     sampling_locations = rearrange(sampling_locations, '(bs c) x -> bs c x', bs=bs)
    #     # 2. 각 candiate patches를 모아 offset layer 통과
    #     # tmp0 = rearrange(tmp0, 'bs c d h w -> bs (d h w) c')    # bs c 256 1 1 => bs 256 c
    #     # tmp0 = self.offset(tmp0)
    #     offset_re = self.offset(tmp0.flatten(1, 2))             # bs 2c 1 1
    #     offset_re = rearrange(offset_re, 'b (c q) j k -> b c (q j k)', c=c)
    #     offset_normalizer = 256
    #     sampling_locations =sampling_locations + offset_re #/ offset_normalizer
    #     # 3. 각 해당하는 offset만큼 prompts 이동
    #     # Normalize sampling locations to range [-1, 1]
        
    #     return imgs, candiates
    
    def forward(
        self, 
        imgs: torch.Tensor,
        pts: Tuple[torch.Tensor, torch.Tensor], # coords : [b n 2], labels : [b n]
    ) -> torch.Tensor:
        #** 1. pass image encoder (global)
        gimage = self.image_encoder(imgs)                   # [b c(256) h(32) w(32)]
        
        cadiates = pts[0]
        patch_idx = (cadiates // 8).to(torch.uint16)[:,:,0,:]  # [b c 2]
        bs, c, _ = patch_idx.shape
        cadiates_feature = torch.empty([0, 256]).to(imgs.device)
        for _b in range(bs):
            for _c in range(c):
                iy, ix = np.array(patch_idx[_b, _c].cpu()).tolist()
                cadiates_feature = torch.concat([cadiates_feature, gimage[_b,:,iy,ix][None]], dim=0)
        # cadiates_feature = rearrange(cadiates_feature, '(bs c) x -> bs c x', bs=bs)
        # candiates간 attention 수행
        # #** 4. Conv
        output = self.convLayer(cadiates_feature[:,:,None,None]).squeeze()  # [bc 256]
        # #** 5. prediction head
        outputs_class = self.class_embed(output)
        outputs_class = rearrange(outputs_class, '(bs c) x -> bs c x', bs=bs)
        outputs_coord = self.bbox_embed(output).sigmoid()
        outputs_coord = rearrange(outputs_coord, '(bs c) x -> bs c x', bs=bs)
        out = {'pred_logits': outputs_class, 'pred_boxes': outputs_coord}
        return out