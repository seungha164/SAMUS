import torch
from torch import nn
from torch.nn import functional as F

from typing import List, Tuple, Type

from .common import LayerNorm2d
from .mask_decoder import MLP

class BoxDecoder(nn.Module):
    def __init__(
        self,
        *,
        transformer_dim: int,
        transformer: nn.Module,
        activation: Type[nn.Module] = nn.GELU,
        iou_head_depth: int = 3,
        iou_head_hidden_dim: int = 256,
        num_cluster: int #= 6
    ) -> None:
        """
        Predicts masks given an image and prompt embeddings, using a
        transformer architecture.

        Arguments:
          transformer_dim (int): the channel dimension of the transformer
          transformer (nn.Module): the transformer used to predict masks
          num_multimask_outputs (int): the number of masks to predict
            when disambiguating masks
          activation (nn.Module): the type of activation to use when
            upscaling masks
          iou_head_depth (int): the depth of the MLP used to predict
            mask quality
          iou_head_hidden_dim (int): the hidden dimension of the MLP
            used to predict mask quality
        """
        super().__init__()
        self.transformer_dim = transformer_dim
        self.transformer = transformer

        self.iou_token = nn.Embedding(1, transformer_dim)
        
        self.num_cluster = num_cluster
        
        self.iou_prediction_head = MLP(
            transformer_dim, iou_head_hidden_dim, self.num_cluster, iou_head_depth
        )

    def forward(
        self,
        image_embeddings: torch.Tensor,
        image_pe: torch.Tensor,
        sparse_prompt_embeddings: torch.Tensor,
        dense_prompt_embeddings: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict masks given image and prompt embeddings.

        Arguments:
          image_embeddings (torch.Tensor): the embeddings from the image encoder
          image_pe (torch.Tensor): positional encoding with the shape of image_embeddings
          sparse_prompt_embeddings (torch.Tensor): the embeddings of the points and boxes
          dense_prompt_embeddings (torch.Tensor): the embeddings of the mask inputs
          multimask_output (bool): Whether to return multiple masks or a single
            mask.

        Returns:
          torch.Tensor: batched predicted masks
          torch.Tensor: batched predictions of mask quality
        """
        boxes, iou_pred = self.predict_boxes(
            image_embeddings=image_embeddings,
            image_pe=image_pe,
            sparse_prompt_embeddings=sparse_prompt_embeddings,
            dense_prompt_embeddings=dense_prompt_embeddings,
        )
        # Prepare output
        return boxes, iou_pred

    def predict_boxes(
        self,
        image_embeddings: torch.Tensor,
        image_pe: torch.Tensor,                 # 1 256 32 32
        sparse_prompt_embeddings: torch.Tensor, # b 2 256
        dense_prompt_embeddings: torch.Tensor, # b 256 32 32
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Predicts masks. See 'forward' for more details."""
        # Concatenate output tokens
        output_tokens = torch.cat([self.iou_token.weight], dim=0) # 1 iou = (1, 256)
        output_tokens = output_tokens.unsqueeze(0).expand(sparse_prompt_embeddings.size(0), -1, -1) # b 1 256
        tokens = torch.cat((output_tokens, sparse_prompt_embeddings), dim=1) # only cat with sparse_prompt # b 7 256 when sparse_prompt point is 2

        # Expand per-image data in batch direction to be per-mask
        if len(image_embeddings.shape) == 3:
            image_embeddings =  image_embeddings.unsqueeze(0)
            src = torch.repeat_interleave(image_embeddings, tokens.shape[0], dim=0) # b 256 32 32 when the decoder operated in bs=1
        else:
            src = image_embeddings
        src = src + dense_prompt_embeddings
        pos_src = torch.repeat_interleave(image_pe, tokens.shape[0], dim=0)  # b 256 32 32
        # b, c, h, w = src.shape

        # Run the transformer
        hs, src = self.transformer(src, pos_src, tokens) # hs (b nt c), src (b N c)
        iou_token_out       = hs[:, 0:1, :]       # b 1 dim
        prompt_tokens_out   = hs[:, 1:-1, :]      # b c dim

        # Generate mask quality predictions
        iou_pred = self.iou_prediction_head(iou_token_out) # b 4

        return prompt_tokens_out, iou_pred