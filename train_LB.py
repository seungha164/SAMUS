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
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import time
import random
from utils.config import get_config
from utils.evaluation import get_eval
from importlib import import_module
from mean_average_precision import MetricBuilder
from torch.nn.modules.loss import CrossEntropyLoss
from monai.losses import DiceCELoss
from einops import rearrange
from models.model_dict import get_model
from utils.data_us import JointTransform2D, ImageToImage2D, ImageToImage2DLB
from utils.loss_functions.sam_loss import get_criterion
from utils.generate_prompts import get_click_prompt
from utils.matcher import HungarianMatcher
from utils.loss_functions.setcriterion import SetCriterion
from typing import Optional, List, Dict
import utils.box_ops as box_ops
from torchmetrics.detection import MeanAveragePrecision
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

@torch.no_grad()
def evaluate(test_loader, model, criterion, opt, args):
    model.eval()
    criterion.eval()
    device = opt.device
    N = len(test_loader)
    losses_result_dict = {'ce' : 0., 'bbox': 0., 'giou': 0., 'total': 0.}
    gt_boxes, pred_boxes = [{'boxes': torch.empty([0, 4]).to(device), 'scores': torch.empty([0]).to(device), 'labels': torch.empty([0]).to(device)}],  [{'boxes': torch.empty([0, 4]).to(device), 'scores': torch.empty([0]).to(device), 'labels': torch.empty([0]).to(device)}] #torch.empty([0, 7]).to(device), torch.empty([0, 6]).to(device)
    mAP_dicts = {'map' : [], 'map_50': [], 'map_75': [], 'map_small': [], 'map_large': [], 'map_medium': []}
    metric = MeanAveragePrecision(iou_type="bbox")
    # start
    for idx, (imgs, prompts, targets) in enumerate(tqdm(test_loader)):
        imgs = imgs.to(device)
        targets = [{k: (v.to(device) if k!='image_id' else v) for k, v in t.items()} for t in targets]
        prompts = {key:prompts[key].to(device) for key in prompts}
        pt = (prompts['pts'], prompts['p_labels'])
        # -------------------------------------------------------- forward --------------------------------------------------------
        preds = model(imgs, pt)      # {'pred_logits' : [bs, N, 2], 'pred_boxes' : [bs, N, 4]}
        loss_dict, boxes_for_metrics = criterion(preds, targets)
        weight_dict = criterion.weight_dict
        losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)
        
        losses_result_dict['ce'] += loss_dict['loss_ce'].item()
        losses_result_dict['bbox'] += loss_dict['loss_bbox'].item()
        losses_result_dict['giou'] += loss_dict['loss_giou'].item()
        losses_result_dict['total'] += losses.item()
        o1, o2, _, o4 = prompts['pts'].shape
        orig_target_sizes = torch.zeros([o1, o2, o4]).to(targets[0]['orig_size'].device)
        for _o1 in range(o1):
            for _o2 in range(o2):
                orig_target_sizes[_o1, _o2] = targets[_o1]['orig_size']
        orig_target_sizes = orig_target_sizes[boxes_for_metrics['idx']]

        # rpreds, rtargets = post_process(boxes_for_metrics, orig_target_sizes)
        result = post_process(boxes_for_metrics, orig_target_sizes)
        pred_boxes  = [{'boxes': result['pboxes'].to(dtype=torch.int64), 'labels': result['plabels'], 'scores': result['pscores']}]
        gt_boxes    = [{'boxes': result['tboxes'].to(dtype=torch.int64), 'labels': result['tcls']}]
        
        metric.update(pred_boxes, gt_boxes)
    mAP = metric.compute()
    print(' ---- inference finish ---- ')
    return {
        'loss_ce'       : (losses_result_dict['ce'] / N), 
        'loss_bbox'     : (losses_result_dict['bbox'] / N),
        'loss_giou'     : (losses_result_dict['giou'] / N),
        'loss_total'    : (losses_result_dict['total'] / N),
        'map'           : mAP['map'],
        'map_50'        : mAP['map_50'],
        'map_75'        : mAP['map_75'],
        'map_small'     : mAP['map_small'],
        'map_medium'    : mAP['map_medium'],
        'map_large'     : mAP['map_large'],
    }


def main():

    #  ============================================================================= parameters setting ====================================================================================
    parser = argparse.ArgumentParser(description='Networks')
    parser.add_argument('--modelname', default='LearableBlock', type=str, help='type of model, e.g., SAM, SAMFull, MedSAM, MSA, SAMed, SAMUS...')
    parser.add_argument('-encoder_input_size', type=int, default=256, help='the image size of the encoder input, 1024 in SAM and MSA, 512 in SAMed, 256 in SAMUS')
    parser.add_argument('-low_image_size', type=int, default=128, help='the image embedding size, 256 in SAM and MSA, 128 in SAMed and SAMUS')
    parser.add_argument('--task', default='BreastCancer_US_Learnable256', help='task or dataset name')
    parser.add_argument('--vit_name', type=str, default='vit_b', help='select the vit model for the image encoder of sam')
    parser.add_argument('--sam_ckpt', type=str, default='checkpoints/sam_vit_b_01ec64.pth', help='Pretrained checkpoint of SAM')
    parser.add_argument('--batch_size', type=int, default=8, help='batch_size per gpu') # SAMed is 12 bs with 2n_gpu and lr is 0.005
    parser.add_argument('--n_gpu', type=int, default=1, help='total gpu')
    parser.add_argument('--base_lr', type=float, default=0.0005, help='segmentation network learning rate, 0.005 for SAMed, 0.0001 for MSA') #0.0006
    parser.add_argument('--warmup', type=bool, default=False, help='If activated, warp up the learning from a lower lr to the base_lr') 
    parser.add_argument('--warmup_period', type=int, default=250, help='Warp up iterations, only valid whrn warmup is activated')
    parser.add_argument('-keep_log', type=bool, default=True, help='keep the loss&lr&dice during training or not')
    parser.add_argument('--cluster_num', type=int, default=3)
    parser.add_argument('--num_transformerlayer', type=int, default=6)
    args = parser.parse_args()
    args.sam_ckpt = './checkpoints/BreastCancer_US_Learnable/LearableBlock_07171906_240_tensor(0.1458).pth'
    opt = get_config(args.task) 
    
    device = torch.device(opt.device)
    logtimestr = time.strftime('%m%d%H%M')  # initialize the tensorboard for record the training process
    if args.keep_log:
        
        boardpath = opt.tensorboard_path + args.modelname + opt.save_path_code + logtimestr
        if not os.path.isdir(boardpath):
            os.makedirs(boardpath)
        TensorWriter = SummaryWriter(boardpath)
    opt.save_path = f"{opt.save_path[:-1]}_C{args.cluster_num}/{logtimestr}/"
    #  =============================================================== add the seed to make sure the results are reproducible ==============================================================

    seed_value = 1234  # the number of seed
    np.random.seed(seed_value)  # set random seed for numpy
    random.seed(seed_value)  # set random seed for python
    os.environ['PYTHONHASHSEED'] = str(seed_value)  # avoid hash random
    torch.manual_seed(seed_value)  # set random seed for CPU
    torch.cuda.manual_seed(seed_value)  # set random seed for one GPU
    torch.cuda.manual_seed_all(seed_value)  # set random seed for all GPU
    torch.backends.cudnn.deterministic = True  # set random seed for convolution

    #  =========================================================================== model and data preparation ============================================================================
    
    # register the sam model
    model = get_model(args.modelname, args=args, opt=opt)
    opt.batch_size = args.batch_size * args.n_gpu

    tf_train = JointTransform2D(img_size=args.encoder_input_size, low_img_size=args.low_image_size, ori_size=opt.img_size, crop=opt.crop, p_flip=0.0, p_rota=0.5, p_scale=0.5, p_gaussn=0.0,
                                p_contr=0.5, p_gama=0.5, p_distor=0.0, color_jitter_params=None, long_mask=True)  # image reprocessing
    tf_val = JointTransform2D(img_size=args.encoder_input_size, low_img_size=args.low_image_size, ori_size=opt.img_size, crop=opt.crop, p_flip=0, color_jitter_params=None, long_mask=True)
    
    train_dataset = ImageToImage2DLB(opt.data_path, opt.train_split, tf_train, img_size=args.encoder_input_size, cluster_num=args.cluster_num)
    val_dataset = ImageToImage2DLB(opt.data_path, opt.val_split, tf_val, img_size=args.encoder_input_size, cluster_num=args.cluster_num)  # return image, mask, and filename
    
    trainloader = DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=True, num_workers=8, pin_memory=True, collate_fn=collate_fn)
    valloader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=8, pin_memory=True, collate_fn=collate_fn)

    model.to(device)
    if opt.pre_trained:
        checkpoint = torch.load(opt.load_path)
        new_state_dict = {}
        for k,v in checkpoint.items():
            if k[:7] == 'module.':
                new_state_dict[k[7:]] = v
            else:
                new_state_dict[k] = v
        model.load_state_dict(new_state_dict)
      
    if args.n_gpu > 1:
        model = nn.DataParallel(model)
    
    if args.warmup:
        b_lr = args.base_lr / args.warmup_period
        optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=b_lr, betas=(0.9, 0.999), weight_decay=0.1)
    else:
        b_lr = args.base_lr
        optimizer = optim.Adam(model.parameters(), lr=args.base_lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
    #! ----------------------------------------------------------------------------------------------------------------
    dec_layers = 6
    # Loss parameters:
    giou_weight = 2.0
    l1_weight = 5.0
    deep_supervision = False #cfg.MODEL.DETR.DEEP_SUPERVISION
    no_object_weight = 0.1
    # building criterion
    matcher = HungarianMatcher(cost_class=5.0, cost_bbox=l1_weight, cost_giou=giou_weight)
    weight_dict = {"loss_ce": 1, "loss_bbox": l1_weight, "loss_giou": giou_weight}
    if deep_supervision:
        aux_weight_dict = {}
        for i in range(dec_layers - 1):
            aux_weight_dict.update({k + f"_{i}": v for k, v in weight_dict.items()})
        weight_dict.update(aux_weight_dict)
    losses = ["labels", "boxes", "cardinality"]
    criterion = SetCriterion(
        1, matcher=matcher, weight_dict=weight_dict, eos_coef=no_object_weight, losses=losses,
    )
    criterion.empty_weight = criterion.empty_weight.to(device)
    
    # criterion = get_criterion(modelname=args.modelname, opt=opt)
    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Total_params: {}".format(pytorch_total_params))

    #  ========================================================================= begin to train the model ============================================================================
    iter_num = 0
    max_iterations = opt.epochs * len(trainloader)
    best_mAP50, loss_log, dice_log = 0.0, np.zeros(opt.epochs+1), np.zeros(opt.epochs+1)
    print("Start training")
    for epoch in range(opt.epochs):
        #  --------------------------------------------------------- training ---------------------------------------------------------
        model.train()
        train_losses = 0
        for batch_idx, (imgs, prompts, targets) in enumerate(tqdm(trainloader)):
            imgs = imgs.to(device)
            targets = [{k: (v.to(device) if k!='image_id' else v) for k, v in t.items()} for t in targets]
            prompts = {key:prompts[key].to(device) for key in prompts}
            pt = (prompts['pts'], prompts['p_labels'])
            
            # -------------------------------------------------------- forward --------------------------------------------------------
            pred = model(imgs, pt)      # {'pred_logits' : [bs, N, 2], 'pred_boxes' : [bs, N, 4]}
            loss_dict, _ = criterion(pred, targets)
            weight_dict = criterion.weight_dict
            losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)
            # -------------------------------------------------------- backward -------------------------------------------------------
            optimizer.zero_grad()
            losses.backward()
            optimizer.step()
            train_losses += losses.item()
            # ------------------------------------------- adjust the learning rate when needed-----------------------------------------
            if args.warmup and iter_num < args.warmup_period:
                lr_ = args.base_lr * ((iter_num + 1) / args.warmup_period)
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr_
            else:
                if args.warmup:
                    shift_iter = iter_num - args.warmup_period
                    assert shift_iter >= 0, f'Shift iter is {shift_iter}, smaller than zero'
                    lr_ = args.base_lr * (1.0 - shift_iter / max_iterations) ** 0.9  # learning rate adjustment depends on the max iterations
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = lr_
            iter_num = iter_num + 1
            # break
        #  -------------------------------------------------- log the train progress --------------------------------------------------
        print('epoch [{}/{}], train loss:{:.4f}'.format(epoch, opt.epochs, train_losses / (batch_idx + 1)))
        if args.keep_log:
            TensorWriter.add_scalar('train_loss', train_losses / (batch_idx + 1), epoch)
            TensorWriter.add_scalar('learning rate', optimizer.state_dict()['param_groups'][0]['lr'], epoch)
            loss_log[epoch] = train_losses / (batch_idx + 1)

        #  --------------------------------------------------------- evaluation ----------------------------------------------------------
        if epoch % opt.eval_freq == 0:
            model.eval()
            val_results = evaluate(valloader, model, criterion, opt, args)
            # mean_loss_ce, mean_loss_bbox, mean_loss_giou, val_losses, val_mAPs = 
            # dices, mean_dice, _, val_losses = get_eval(valloader, model, criterion=criterion, opt=opt, args=args)
            # mean_loss_ce, mean_loss_bbox, mean_loss_giou, val_losses, val_mAP = get_eval(valloader, model, criterion=criterion, opt=opt, args=args)
            print('epoch [{}/{}], val loss:{:.4f}'.format(epoch, opt.epochs, val_results['loss_total']))
            print('epoch [{}/{}], val loss-ce:{:.4f}, loss-bbox:{:.4f}, loss-giou:{:.4f}'.format(
                epoch, opt.epochs, val_results['loss_ce'], val_results['loss_bbox'], val_results['loss_giou']))
            print('epoch [{}/{}], val mAP:{:.4f}, val mAP50:{:.4f}, val mAP75:{:.4f}'.format(
                epoch, opt.epochs, val_results['map'], val_results['map_50'], val_results['map_75']))
            if args.keep_log:
                TensorWriter.add_scalar('val_mAP', val_results['map'], epoch)
                TensorWriter.add_scalar('val_mAP50', val_results['map_50'], epoch)
                TensorWriter.add_scalar('val_mAP75', val_results['map_75'], epoch)
                TensorWriter.add_scalar('val_mAP_small', val_results['map_small'], epoch)
                TensorWriter.add_scalar('val_mAP_large', val_results['map_large'], epoch)
                TensorWriter.add_scalar('val_mAP_medium', val_results['map_medium'], epoch)
                TensorWriter.add_scalar('val_loss', val_results['loss_total'], epoch)
                TensorWriter.add_scalar('mean_loss_ce', val_results['loss_ce'], epoch)
                TensorWriter.add_scalar('mean_loss_bbox', val_results['loss_bbox'], epoch)
                TensorWriter.add_scalar('mean_loss_giou', val_results['loss_giou'], epoch)
                # dice_log[epoch] = val_mAP
            # if val_losses > best_losses:
            #     best_losses = val_losses
            #     timestr = time.strftime('%m%d%H%M')
            #     if not os.path.isdir(opt.save_path):
            #         os.makedirs(opt.save_path)
            #     save_path = opt.save_path + args.modelname + opt.save_path_code + '%s' % timestr + '_' + str(epoch) + '_' + str(round(val_losses.item(), 7))
            #     torch.save(model.state_dict(), save_path + ".pth", _use_new_zipfile_serialization=False)
            if val_results['map_50'] > best_mAP50:
                best_mAP50 = val_results['map_50']
                timestr = time.strftime('%m%d%H%M')
                if not os.path.isdir(opt.save_path):
                    os.makedirs(opt.save_path)
                save_path = opt.save_path + args.modelname + opt.save_path_code + '%s' % timestr + '_' + str(epoch) + '_' + str(round(best_mAP50.item(), 7))  # + '_' + str(round(val_losses.item(), 7))
                torch.save(model.state_dict(), save_path + ".pth", _use_new_zipfile_serialization=False)
                
        if epoch % opt.save_freq == 0 or epoch == (opt.epochs-1):
            if not os.path.isdir(opt.save_path):
                os.makedirs(opt.save_path)
            save_path = opt.save_path + args.modelname + opt.save_path_code + '_' + str(epoch)
            torch.save(model.state_dict(), save_path + ".pth", _use_new_zipfile_serialization=False)
            if args.keep_log:
                with open(opt.tensorboard_path + args.modelname + opt.save_path_code + logtimestr + '/trainloss.txt', 'w') as f:
                    for i in range(len(loss_log)):
                        f.write(str(loss_log[i])+'\n')
                with open(opt.tensorboard_path + args.modelname + opt.save_path_code + logtimestr + '/mAP.txt', 'w') as f:
                    for i in range(len(dice_log)):
                        f.write(str(dice_log[i])+'\n')

if __name__ == '__main__':
    main()