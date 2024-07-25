from ast import arg
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
import argparse
from pickle import FALSE, TRUE
from statistics import mode
from tkinter import image_names
import torch
import torchvision
from torch import nn, Tensor
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torch.optim as optim
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
import time
import random
from utils.config import get_config
from utils.evaluation import get_eval
from importlib import import_module
from utils.data_us import JointTransform2D, ImageToImage2D, ImageToImage2DLB
from torch.nn.modules.loss import CrossEntropyLoss
from monai.losses import DiceCELoss
from einops import rearrange
from models.model_dict import get_model
from utils.data_us import JointTransform2D, ImageToImage2D
from utils.loss_functions.sam_loss import get_criterion
from utils.generate_prompts import get_click_prompt
from thop import profile
from utils.coco_eval import CocoEvaluator
from tqdm import tqdm
from utils.misc import reduce_dict, MetricLogger, SmoothedValue
from utils.loss_functions.setcriterion import SetCriterion
from utils.matcher import HungarianMatcher
from typing import Optional, List, Dict
import torch.nn.functional as F
import utils.box_ops as box_ops
from mean_average_precision import MetricBuilder
from pprint import pprint

def tensor_from_tensor_dict(tensor_dicts: List[Dict]):
    dict = {}
    for key in tensor_dicts[0]:
        batch_shape = [len(tensor_dicts)] + list(tensor_dicts[0][key].shape)
        dtype = tensor_dicts[0][key].dtype
        device = tensor_dicts[0][key].device
        tensor = torch.zeros(batch_shape, dtype=dtype, device=device)
        for idx in range(len(tensor_dicts)):
            tensor[idx] = tensor_dicts[idx][key]
        dict[key] = tensor
    return dict

def tensor_from_tensor_list(tensor_list: List[Tensor]):
    if tensor_list[0].ndim == 3:
        batch_shape = [len(tensor_list)] + [1, 256, 256]
        b, c, h, w = batch_shape
        dtype = tensor_list[0].dtype
        device = tensor_list[0].device
        tensor = torch.zeros(batch_shape, dtype=dtype, device=device)
        for idx in range(len(tensor_list)):
            tensor[idx] = tensor_list[idx]
    return tensor

def collate_fn(batch):
    batch = list(zip(*batch))
    # image는 tensor로 묶어주기
    batch[0] = tensor_from_tensor_list(batch[0])
    batch[1] = tensor_from_tensor_dict(batch[1])
    return tuple(batch)

def post_process(match_outputs, orig_sizes):
    # pred
    pboxes, pcls = match_outputs['pred_boxes'], match_outputs['pred_cls']
    img_h, img_w = orig_sizes.unbind(1)
    scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1)
    pboxes = box_ops.box_cxcywh_to_xyxy(pboxes * scale_fct)
    pscores, plabels = pcls
    # target
    tboxes, tcls = match_outputs['target_boxes'], match_outputs['target_cls']
    tboxes = box_ops.box_cxcywh_to_xyxy(tboxes * scale_fct)
    # preds    = torch.concat([box_ops.box_cxcywh_to_xyxy(pboxes), plabels[:,None], pscores[:,None]], dim=1)       # [N_boxes, 6]
    # targets  = torch.concat([box_ops.box_cxcywh_to_xyxy(tboxes), tcls[:, None], torch.zeros([int(tboxes.shape[0]), 2]).to(tboxes.device)], dim=1)
    return  {'pboxes' : pboxes, 'pscores' : pscores, 'plabels': plabels, 'tboxes': tboxes, 'tcls': tcls} #preds, targets

from torchmetrics.detection import MeanAveragePrecision
import cv2
GT_COLOR = (255, 255, 255)
MATCHING_COLOR_LIST = [(255,0,0), (0,0,255)]   # blue, red
MATCHING_COLOR_LIST_pre = [(237,149,100), (71,99,255)]   # blue, red
UNMATCHING_COLORS = [(143,188,143),(152,251,152),(144,238,144),(34,139,34),(50,205,50),(0,100,0)]
def vis(imgs, prompts, selected_prompts, before_matching_boxes, preds, boxess, orig_sizes, names, VIS_SAVE_ROOT):
    os.makedirs(VIS_SAVE_ROOT, exist_ok=True)
    for i in range(imgs.shape[0]):
        img, pred, gt, orig_size, name, p_coords = imgs[i][0], preds[i]['boxes'], boxess[i]['boxes'], orig_sizes[i], names[i], prompts['pts'][i]
        all_boxes = before_matching_boxes[i]
        # 1 img 
        image = cv2.resize(np.array(img.cpu()) * 255, np.array(orig_size.cpu()).astype(np.uint16).tolist())
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        # 2 prompts(candiates)
        for _i, p_coord in enumerate(p_coords):
            cv2.circle(image, np.array(p_coord[0].cpu()).astype(np.uint16).tolist(), 7, UNMATCHING_COLORS[_i], -1, cv2.LINE_AA)
        # all boxes
        for bN in range(all_boxes.shape[0]):
            minx, miny, maxx, maxy = np.array(box_ops.box_cxcywh_to_xyxy(all_boxes[bN]).cpu() * orig_size[0].item()).astype(np.uint16)
            cv2.rectangle(image, [minx, miny],  [maxx, maxy], UNMATCHING_COLORS[bN], 2)
            
        # 3 bbox
        for bN in range(gt.shape[0]):
            minx, miny, maxx, maxy = np.array(gt[bN].cpu())
            pminx, pminy, pmaxx, pmaxy = np.array(pred[bN].cpu())
            cv2.rectangle(image, [int(minx), int(miny)],  [int(maxx), int(maxy)], GT_COLOR, 2)
            cv2.rectangle(image, [int(pminx), int(pminy)],  [int(pmaxx), int(pmaxy)], MATCHING_COLOR_LIST[bN], 2)
            cv2.circle(image, np.array(selected_prompts[bN][0].cpu()).astype(np.uint16).tolist(), 7, MATCHING_COLOR_LIST_pre[bN], -1, cv2.LINE_AA)  # candiate
            # cv2.circle(image, [int((pminx + pmaxx)/2), int((pminy + pmaxy)/2)], 7, MATCHING_COLOR_LIST[bN], -1, cv2.LINE_AA)
        cv2.imwrite(f'{VIS_SAVE_ROOT}/{name}.png', image)
        # print(i)
import matplotlib.pyplot as plt
@torch.no_grad()
def evaluate(test_loader, model, criterion, opt, args, VIS_SAVE_ROOT):
    model.eval()
    criterion.eval()
    device = opt.device
    max_slice_number = len(test_loader)
    losses_dict = {'ce' : 0., 'bbox': 0., 'giou': 0., 'total': 0.}
    gt_boxes, pred_boxes = [{'boxes': torch.empty([0, 4]).to(device), 'scores': torch.empty([0]).to(device), 'labels': torch.empty([0]).to(device)}],  [{'boxes': torch.empty([0, 4]).to(device), 'scores': torch.empty([0]).to(device), 'labels': torch.empty([0]).to(device)}] #torch.empty([0, 7]).to(device), torch.empty([0, 6]).to(device)
    mAP_dicts = {'map' : [], 'map_50': [], 'map_75': [], 'map_small': [], 'map_large': [], 'map_medium': []}
    metric = MeanAveragePrecision(iou_type="bbox")
    # start
    for idx, (imgs, prompts, targets) in enumerate(tqdm(test_loader)):
        # imgs      : [bs, 1, 256, 256]
        # targets   : {boxes : [[정규화된 cx, cy, w, h], .. ], labels: [1], orig_size: [562,562], .. , image_id} 
        # prompts   : {p_labels : [bs, N_prompt, 1], pts: [bs, N_propmt, 1, 2]}
        imgs = imgs.to(device)                  # [1, 1, 256, 256]
        targets = [{k: (v.to(device) if k!='image_id' else v) for k, v in t.items()} for t in targets]
        prompts = {key:prompts[key].to(device) for key in prompts}
        pt = (prompts['pts'], prompts['p_labels'])
        # -------------------------------------------------------- forward --------------------------------------------------------
        preds = model(imgs, pt)      # {'pred_logits' : [bs, N, 2], 'pred_boxes' : [bs, N, 4]}
        loss_dict, boxes_for_metrics = criterion(preds, targets)
        weight_dict = criterion.weight_dict
        losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)
        # ------------------------------------------------------- result 추출 -----------------------------------------------------
        o1, o2, _, o4 = prompts['pts'].shape
        orig_target_sizes = torch.zeros([o1, o2, o4]).to(targets[0]['orig_size'].device)
        for _o1 in range(o1):
            for _o2 in range(o2):
                orig_target_sizes[_o1, _o2] = targets[_o1]['orig_size']
        orig_target_sizes = orig_target_sizes[boxes_for_metrics['idx']]

        result = post_process(boxes_for_metrics, orig_target_sizes)
        
        pred_boxes  = [{'boxes': result['pboxes'].to(dtype=torch.int64), 'labels': result['plabels'], 'scores': result['pscores']}]
        gt_boxes    = [{'boxes': result['tboxes'].to(dtype=torch.int64), 'labels': result['tcls']}]
        
        metric.update(pred_boxes, gt_boxes)
    
        target_images_id = [t['image_id'] for t in targets]
        selected_prompts = prompts['pts'][boxes_for_metrics['idx']]
        vis(imgs, prompts, selected_prompts, preds['pred_boxes'], pred_boxes, gt_boxes, orig_target_sizes, target_images_id, VIS_SAVE_ROOT)
    mAP = metric.compute()
    print(' ---- inference finish ---- ')
    pprint(f"[result]\n{mAP}")
    
def main():

    #  =========================================== parameters setting ==================================================

    parser = argparse.ArgumentParser(description='Networks')
    parser.add_argument('--modelname', default='LearableBlock', type=str, help='type of model, e.g., SAM, SAMFull, SAMHead, MSA, SAMed, SAMUS...')
    parser.add_argument('-encoder_input_size', type=int, default=256, help='the image size of the encoder input, 1024 in SAM and MSA, 512 in SAMed, 256 in SAMUS') 
    parser.add_argument('-low_image_size', type=int, default=128, help='the image embedding size, 256 in SAM and MSA, 128 in SAMed and SAMUS') 
    parser.add_argument('--task', default='BreastCancer_US_Learnable256', help='task or dataset name')
    parser.add_argument('--vit_name', type=str, default='vit_b', help='select the vit model for the image encoder of sam')
    parser.add_argument('--sam_ckpt', type=str, default='checkpoints/sam_vit_b_01ec64.pth', help='Pretrained checkpoint of SAM')
    parser.add_argument('--batch_size', type=int, default=1, help='batch_size per gpu') # 8 # SAMed is 12 bs with 2n_gpu and lr is 0.005
    parser.add_argument('--n_gpu', type=int, default=1, help='total gpu')
    parser.add_argument('--base_lr', type=float, default=0.0001, help='segmentation network learning rate, 0.005 for SAMed, 0.0001 for MSA') #0.0006
    parser.add_argument('--warmup', type=bool, default=False, help='If activated, warp up the learning from a lower lr to the base_lr') # True
    parser.add_argument('--warmup_period', type=int, default=250, help='Warp up iterations, only valid whrn warmup is activated')
    parser.add_argument('--cluster_num', type=int, default=2)
    parser.add_argument('--num_transformerlayer', type=int, default=6)
    
    args = parser.parse_args()
    opt = get_config(args.task)  # please configure your hyper-parameter
    print("task", args.task, "checkpoints:", opt.load_path)
    opt.mode = "test"
    prompt_type = 'click' #'multi_bbox' #'click'#
    #args.modelname = 'Samus_multi_prompts'
    #opt.eval_mode = "eval_mask_SamUSMultiPrompts"
    #!!!!!!!!!!!!!!!!!
    #opt.classes=2
    opt.visual = True
    #opt.eval_mode = "patient"
    opt.modelname = args.modelname
    device = torch.device(opt.device)

    #  =============================================================== add the seed to make sure the results are reproducible ==============================================================
    seed_value = 300 # the number of seed
    np.random.seed(seed_value)  # set random seed for numpy
    random.seed(seed_value)  # set random seed for python
    os.environ['PYTHONHASHSEED'] = str(seed_value)  # avoid hash random
    torch.manual_seed(seed_value)  # set random seed for CPU
    torch.cuda.manual_seed(seed_value)  # set random seed for one GPU
    torch.cuda.manual_seed_all(seed_value)  # set random seed for all GPU
    torch.backends.cudnn.deterministic = True  # set random seed for convolution
    
    # (0) model
    # register the sam model
    args.sam_ckpt = './checkpoints/BreastCancer_US_256_Learnable_C2/07240743/LearableBlock_07241537_342_0.6457503.pth'
    VIS_SAVE_ROOT = f'./results/{args.sam_ckpt.split("/")[-1].replace(".pth", "")}/'
    model = get_model(args.modelname, args=args, opt=opt)
    model.to(device)
    # opt.load_path = './checkpoints/BreastCancer_US_Learnable_C6/07221203/LearableBlock_07221223_36_0.2578001.pth'
    # checkpoint = torch.load(opt.load_path)
    # #------when the load model is saved under multiple GPU
    # new_state_dict = {}
    # for k,v in checkpoint.items():
    #     if k[:7] == 'module.':
    #         new_state_dict[k[7:]] = v
    #     else:
    #         new_state_dict[k] = v
    # model.load_state_dict(new_state_dict)
    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Total_params: {}".format(pytorch_total_params))
    # (1) Loss
    losses = ["labels", "boxes", "cardinality"]
    giou_weight, l1_weight, no_object_weight = 2.0, 5.0, 0.1
    matcher = HungarianMatcher(cost_class=1, cost_bbox=l1_weight, cost_giou=giou_weight)
    weight_dict = {"loss_ce": 1, "loss_bbox": l1_weight, "loss_giou": giou_weight}
    criterion = SetCriterion(
        1, matcher=matcher, weight_dict=weight_dict, eos_coef=no_object_weight, losses=losses,
    )
    criterion = criterion.to(device)

    #* -------------------------------- setting -------------------------------------------------------
    #  =========================================================================== model and data preparation ============================================================================
    # (2) Data Loading
    tf_test         = JointTransform2D(img_size=args.encoder_input_size, low_img_size=args.low_image_size, ori_size=opt.img_size, crop=opt.crop, p_flip=0, color_jitter_params=None, long_mask=True)
    test_dataset    = ImageToImage2DLB(opt.data_path, opt.test_split, tf_test, img_size=args.encoder_input_size, cluster_num=args.cluster_num)  # return image, mask, and filename
    test_loader     = DataLoader(test_dataset, batch_size=1, shuffle=False,  num_workers=8, pin_memory=True, collate_fn=collate_fn)

    #  ========================================================================= begin to evaluate the model ============================================================================
    results = evaluate(
        test_loader, model, criterion, opt, args, VIS_SAVE_ROOT
    )

    
if __name__ == '__main__':
    main()
