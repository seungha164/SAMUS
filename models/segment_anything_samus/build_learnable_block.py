import torch

from functools import partial

from .modeling import LearableBlock
from .modeling import ImageEncoderViT, PromptEncoder
from torch.nn import functional as F

def build_learnableblock_vit_b(args, checkpoint=None):
    return _build_learnable_block(
        args,
        encoder_embed_dim=768,
        encoder_depth=12,
        encoder_num_heads=12,
        encoder_global_attn_indexes=[2, 5, 8, 11],
        checkpoint=checkpoint,
    )
    

learnableblock_model_registry = {
    "vit_b": build_learnableblock_vit_b,
}

def _build_learnable_block(
    args,
    encoder_embed_dim,
    encoder_depth,
    encoder_num_heads,
    encoder_global_attn_indexes,
    checkpoint=None,
):
    prompt_embed_dim = 256
    image_size = args.encoder_input_size
    patch_size = image_size//32
    image_embedding_size = image_size // patch_size
    learnableblock = LearableBlock(
        image_encoder=ImageEncoderViT(
            depth=encoder_depth,
            embed_dim=encoder_embed_dim,
            img_size=image_size,
            mlp_ratio=4,
            norm_layer=partial(torch.nn.LayerNorm, eps=1e-6),
            num_heads=encoder_num_heads,
            patch_size= patch_size,
            qkv_bias=True,
            use_rel_pos=True,
            global_attn_indexes=encoder_global_attn_indexes,
            window_size=14,
            out_chans=prompt_embed_dim,
        ),
        prompt_encoder=PromptEncoder(
            embed_dim=prompt_embed_dim,
            image_embedding_size=(image_embedding_size, image_embedding_size),
            input_image_size=(image_size, image_size),
            mask_in_chans=16,
        ),
    )
    learnableblock.eval()
    if checkpoint is not None:
        with open(checkpoint, "rb") as f:
            state_dict = torch.load(f)
        try:
            learnableblock.load_state_dict(state_dict)
        except:
            new_state_dict = load_from2(learnableblock, state_dict, image_size, patch_size)
            learnableblock.load_state_dict(new_state_dict)
    return learnableblock

def load_from2(learnableblock, sam_dict, image_size, patch_size): # load the positional embedding
    samus_dict = learnableblock.state_dict()
    dict_trained = {k: v for k, v in sam_dict.items() if k in samus_dict}
    token_size = int(image_size//patch_size)
    # pos_embed = dict_trained['image_encoder.pos_embed']
    # pos_embed = pos_embed.permute(0, 3, 1, 2)  # [b, c, h, w]
    # pos_embed = F.interpolate(pos_embed, (token_size, token_size), mode='bilinear', align_corners=False)
    # pos_embed = pos_embed.permute(0, 2, 3, 1)  # [b, h, w, c]
    # dict_trained['image_encoder.pos_embed'] = pos_embed
    rel_pos_keys = [k for k in dict_trained.keys() if 'rel_pos' in k]
    global_rel_pos_keys = [k for k in rel_pos_keys if '2' in k or '5' in  k or '8' in k or '11' in k]
    for k in global_rel_pos_keys:
        rel_pos_params = dict_trained[k]
        h, w = rel_pos_params.shape
        rel_pos_params = rel_pos_params.unsqueeze(0).unsqueeze(0)
        rel_pos_params = F.interpolate(rel_pos_params, (token_size * 2 - 1, w), mode='bilinear', align_corners=False)
        dict_trained[k] = rel_pos_params[0, 0, ...]
    samus_dict.update(dict_trained)
    return samus_dict