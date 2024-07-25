from models.segment_anything.build_sam import sam_model_registry
from models.segment_anything_samus.build_learnable_block import learnableblockOffset_model_registry, learnableblock_model_registry, learnableblockZoom_model_registry, learnableblockMD_model_registry
from models.segment_anything_samus.build_sam_us import samus_model_registry

def get_model(modelname="SAM", args=None, opt=None):
    if modelname == "SAM":
        model = sam_model_registry['vit_b'](checkpoint=args.sam_ckpt)
    elif modelname == "SAMUS":
        model = samus_model_registry['vit_b'](args=args, checkpoint=args.sam_ckpt)
    elif modelname == "LearableBlock":
        model = learnableblock_model_registry['vit_b'](args=args, checkpoint=args.sam_ckpt)
    elif modelname == "LearableBlockZoom":
        model = learnableblockZoom_model_registry['vit_b'](args=args, checkpoint=args.sam_ckpt)
    elif modelname == "LearableBlockOffset":
        model = learnableblockOffset_model_registry['vit_b'](args=args, checkpoint=args.sam_ckpt)
    elif modelname == "LearableBlockMD":
        model = learnableblockMD_model_registry['vit_b'](args=args, checkpoint=args.sam_ckpt)
    elif modelname == "LearableBlockFiltering":
        model = learnableblock_model_registry['vit_b_filtering'](args=args, checkpoint=args.sam_ckpt)
    else:
        raise RuntimeError("Could not find the model:", modelname)
    return model
