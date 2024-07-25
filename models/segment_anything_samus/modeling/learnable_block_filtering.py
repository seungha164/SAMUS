import torch
from torch import nn, Tensor
from torch.nn import functional as F

from einops import rearrange
import copy
from typing import Any, List, Tuple, Optional

from .image_encoder import ImageEncoderViT
from .mask_decoder import MaskDecoder
from .prompt_encoder import PromptEncoder


class LearableBlock_Filtering(nn.Module):
    def __init__(
        self,
        image_encoder: ImageEncoderViT,
        prompt_encoder: PromptEncoder,
        pixel_mean: List[float] = [123.675, 116.28, 103.53],
        pixel_std: List[float] = [58.395, 57.12, 57.375],
        d_model = 256,
        nhead=8,
        num_decoder_layers = 1, #6,
        dim_feedforward=2048,
        dropout=0.1,
        activation="relu",
        normalize_before=False,
        return_intermediate_dec=False
    ) -> None:
        super().__init__()
        self.image_encoder = image_encoder
        self.prompt_encoder = prompt_encoder
        #! decoder (by. DETR official code)
        self.query_embed = nn.Embedding(32 * 32, 256)           # num_queries, hidden_dim

        #! MLP
        self.class_embed = nn.Linear(256, 1 + 1)

        self.register_buffer("pixel_mean", torch.Tensor(pixel_mean).view(-1, 1, 1), False)
        self.register_buffer("pixel_std", torch.Tensor(pixel_std).view(-1, 1, 1), False)
        
        #! Encoder & Prompt Encoder freeze
        for param in self.image_encoder.parameters():
            param.requires_grad = False
        for param in self.prompt_encoder.parameters():
            param.requires_grad = False
    
    @property
    def device(self) -> Any:
        return self.pixel_mean.device
    
    def forward(
        self, 
        imgs: torch.Tensor,
        pts: Tuple[torch.Tensor, torch.Tensor],  # coords : [b n 2], labels : [b n]
    ) -> torch.Tensor:
        #** 1. pass image encoder
        imge = self.image_encoder(imgs)                 # [b, c, h, w]
        bs, d, h,w = imge.shape
        # imge = rearrange(imge, 'bs c h w -> bs h w c')
        
        #** 2. pass prompt encoder - each candiate_point
        
        se = torch.empty((bs, 0, 256), device=imge.device)#, torch.empty((bs, 0, 256), device=imge.device)
        pti = (rearrange(pts[0], 'b N k w -> (b N) k w'), rearrange(pts[1], 'b N k -> (b N) k'))
        sei, dei = self.prompt_encoder(
            points = pti,
            boxes = None,
            masks = None
        ) # [b, 2, 256], [b, 256, 32, 32]
        se = rearrange(sei[:,:1,:],'(b N) c d -> b N (c d)', b=bs)  # pad 제거
        c = se.shape[1]
        # compute similarity matrix
        sim = torch.matmul(se, imge.flatten(2,3)).reshape(bs, c, h, w) # [bs, candiates, hw]
        imge2 = torch.concat([imge[:,None,:,:,:], imge[:,None,:,:,:]], dim=1)
        imge2 = imge2 + imge2 * sim[:,:,None,:,:]
        # compute class-activate feature
        imge2 = rearrange(imge2, 'b c d h w -> (b c) (h w) d')
        
        query_embed = self.prompt_encoder.get_dense_pe()
        query_embed = rearrange(query_embed, 'bs c h w -> bs (h w) c')
        hs = self.decoder(se, imge, memory_key_padding_mask=None,
                          pos=query_embed, query_pos=None)
        
        #** 4. final FFN -> [N, 4]
        outputs_class = self.class_embed(hs)
        outputs_coord = self.bbox_embed(hs).sigmoid()
        out = {'pred_logits': outputs_class, 'pred_boxes': outputs_coord}
        return out