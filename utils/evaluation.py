# this file is utilized to evaluate the models from different mode: 2D-slice level, 2D-patient level, 3D-patient level
from tkinter import image_names
from torch.autograd import Variable
from torch.utils.data import DataLoader
import os
import numpy as np
import torch
import torch.nn.functional as F
import utils.metrics as metrics
from hausdorff import hausdorff_distance
from utils.visualization import visual_segmentation, visual_segmentation_binary, visual_segmentation_sets, visual_segmentation_sets_with_pt
from einops import rearrange
from utils.generate_prompts import get_click_prompt
import time
import pandas as pd
import utils.box_ops as box_ops
from mean_average_precision import MetricBuilder

def fix_bn(m):
    classname = m.__class__.__name__
    if classname.find('BatchNorm') != -1:
        m.eval()

def obtain_patien_id(filename):
    if "-" in filename: # filename = "xx-xx-xx_xxx"
        filename = filename.split('-')[-1]
    # filename = xxxxxxx or filename = xx_xxx
    if "_" in filename:
        patientid = filename.split("_")[0]
    else:
        patientid = filename[:3]
    return patientid

def eval_mask_slice(valloader, model, criterion, opt, args):
    model.eval()
    val_losses, mean_dice = 0, 0
    dices = np.zeros(opt.classes)
    hds = np.zeros(opt.classes)
    ious, accs, ses, sps = np.zeros(opt.classes), np.zeros(opt.classes), np.zeros(opt.classes), np.zeros(opt.classes)
    eval_number = 0
    sum_time = 0
    for batch_idx, (datapack) in enumerate(valloader):
        imgs = Variable(datapack['image'].to(dtype = torch.float32, device=opt.device))
        masks = Variable(datapack['low_mask'].to(dtype = torch.float32, device=opt.device))
        label = Variable(datapack['label'].to(dtype = torch.float32, device=opt.device))

        pt = get_click_prompt(datapack, opt)

        with torch.no_grad():
            start_time = time.time()
            pred = model(imgs, pt)
            sum_time =  sum_time + (time.time()-start_time)

        val_loss = criterion(pred, masks)
        val_losses += val_loss.item()

        if args.modelname == 'MSA' or args.modelname == 'SAM':
            gt = masks.detach().cpu().numpy()
        else:
            gt = label.detach().cpu().numpy()
        gt = gt[:, 0, :, :]
        predict = torch.sigmoid(pred['masks'])
        predict = predict.detach().cpu().numpy()  # (b, c, h, w)
        seg = predict[:, 0, :, :] > 0.5  # (b, h, w)
        b, h, w = seg.shape
        for j in range(0, b):
            pred_i = np.zeros((1, h, w))
            pred_i[seg[j:j+1, :, :] == 1] = 255
            gt_i = np.zeros((1, h, w))
            gt_i[gt[j:j+1, :, :] == 1] = 255
            dice_i = metrics.dice_coefficient(pred_i, gt_i)
            #print("name:", name[j], "coord:", coords_torch[j], "dice:", dice_i)
            dices[1] += dice_i
            iou, acc, se, sp = metrics.sespiou_coefficient2(pred_i, gt_i, all=False)
            ious[1] += iou
            accs[1] += acc
            ses[1] += se
            sps[1] += sp
            hds[1] += hausdorff_distance(pred_i[0, :, :], gt_i[0, :, :], distance="manhattan")
            del pred_i, gt_i
        eval_number = eval_number + b
    dices = dices / eval_number
    hds = hds / eval_number
    ious, accs, ses, sps = ious/eval_number, accs/eval_number, ses/eval_number, sps/eval_number
    val_losses = val_losses / (batch_idx + 1)
    mean_dice = np.mean(dices[1:])
    mean_hdis = np.mean(hds[1:])
    mean_iou, mean_acc, mean_se, mean_sp = np.mean(ious[1:]), np.mean(accs[1:]), np.mean(ses[1:]), np.mean(sps[1:])
    print("test speed", eval_number/sum_time)
    if opt.mode == "train":
        return dices, mean_dice, mean_hdis, val_losses
    else:
        return mean_dice, mean_iou, mean_acc, mean_se, mean_sp


def eval_mask_slice2(valloader, model, criterion, opt, args):
    model.eval()
    val_losses, mean_dice = 0, 0
    max_slice_number = opt.batch_size * (len(valloader) + 1)
    dices = np.zeros((max_slice_number, opt.classes))
    hds = np.zeros((max_slice_number, opt.classes))
    ious, accs, ses, sps = np.zeros((max_slice_number, opt.classes)), np.zeros((max_slice_number, opt.classes)), np.zeros((max_slice_number, opt.classes)), np.zeros((max_slice_number, opt.classes))
    eval_number = 0
    sum_time = 0
    for batch_idx, (datapack) in enumerate(valloader):
        imgs = Variable(datapack['image'].to(dtype = torch.float32, device=opt.device))
        masks = Variable(datapack['low_mask'].to(dtype = torch.float32, device=opt.device))
        label = Variable(datapack['label'].to(dtype = torch.float32, device=opt.device))
        class_id = datapack['class_id']
        image_filename = datapack['image_name']

        pt = get_click_prompt(datapack, opt)

        with torch.no_grad():
            start_time = time.time()
            pred = model(imgs, pt)
            sum_time =  sum_time + (time.time()-start_time)

        val_loss = criterion(pred, masks)
        val_losses += val_loss.item()

        if args.modelname == 'MSA' or args.modelname == 'SAM':
            gt = masks.detach().cpu().numpy()
        else:
            gt = label.detach().cpu().numpy()
        gt = gt[:, 0, :, :]
        predict = torch.sigmoid(pred['masks'])
        predict = predict.detach().cpu().numpy()  # (b, c, h, w)
        seg = predict[:, 0, :, :] > 0.5  # (b, h, w)
        b, h, w = seg.shape
        for j in range(0, b):
            pred_i = np.zeros((1, h, w))
            pred_i[seg[j:j+1, :, :] == 1] = 255
            gt_i = np.zeros((1, h, w))
            gt_i[gt[j:j+1, :, :] == 1] = 255
            dice_i = metrics.dice_coefficient(pred_i, gt_i)
            #print("name:", name[j], "coord:", coords_torch[j], "dice:", dice_i)
            dices[eval_number+j, 1] += dice_i
            iou, acc, se, sp = metrics.sespiou_coefficient2(pred_i, gt_i, all=False)
            ious[eval_number+j, 1] += iou
            accs[eval_number+j, 1] += acc
            ses[eval_number+j, 1] += se
            sps[eval_number+j, 1] += sp
            hds[eval_number+j, 1] += hausdorff_distance(pred_i[0, :, :], gt_i[0, :, :], distance="manhattan")
            del pred_i, gt_i
            if opt.visual:
                visual_segmentation_sets_with_pt(seg[j:j+1, :, :], image_filename[j], opt, pt[0][j, :, :])
        eval_number = eval_number + b
    dices = dices[:eval_number, :] 
    hds = hds[:eval_number, :] 
    ious, accs, ses, sps = ious[:eval_number, :], accs[:eval_number, :], ses[:eval_number, :], sps[:eval_number, :]
    val_losses = val_losses / (batch_idx + 1)

    dice_mean = np.mean(dices, axis=0)
    dices_std = np.std(dices, axis=0)
    hd_mean = np.mean(hds, axis=0)
    hd_std = np.std(hds, axis=0)

    mean_dice = np.mean(dice_mean[1:])
    mean_hdis = np.mean(hd_mean[1:])
    print("test speed", eval_number/sum_time)
    if opt.mode == "train":
        return dices, mean_dice, mean_hdis, val_losses
    else:
        # data = pd.DataFrame(dices*100)
        # writer = pd.ExcelWriter('./result/' + args.task + '/PT10-' + opt.modelname + '.xlsx')
        # data.to_excel(writer, 'page_1', float_format='%.2f')
        # writer._save()

        dice_mean = np.mean(dices*100, axis=0)
        dices_std = np.std(dices*100, axis=0)
        hd_mean = np.mean(hds, axis=0)
        hd_std = np.std(hds, axis=0)
        iou_mean = np.mean(ious*100, axis=0)
        iou_std = np.std(ious*100, axis=0)
        acc_mean = np.mean(accs*100, axis=0)
        acc_std = np.std(accs*100, axis=0)
        se_mean = np.mean(ses*100, axis=0)
        se_std = np.std(ses*100, axis=0)
        sp_mean = np.mean(sps*100, axis=0)
        sp_std = np.std(sps*100, axis=0)
        return dice_mean, hd_mean, iou_mean, acc_mean, se_mean, sp_mean, dices_std, hd_std, iou_std, acc_std, se_std, sp_std

def eval_camus_patient(valloader, model, criterion, opt, args):
    model.eval()
    val_losses, mean_dice = 0, 0
    dices = np.zeros(opt.classes)
    patientnumber = 6000  # maxnum patient number
    flag = np.zeros(patientnumber)  # record the patients
    tps, fps = np.zeros((patientnumber, opt.classes)), np.zeros((patientnumber, opt.classes))
    tns, fns = np.zeros((patientnumber, opt.classes)), np.zeros((patientnumber, opt.classes))
    hds = np.zeros((patientnumber, opt.classes))
    for batch_idx, (datapack) in enumerate(valloader):
        imgs = Variable(datapack['image'].to(dtype = torch.float32, device=opt.device))
        masks = Variable(datapack['low_mask'].to(dtype = torch.float32, device=opt.device))
        label = Variable(datapack['label'].to(dtype = torch.float32, device=opt.device))
        image_filename = datapack['image_name']
        class_id = datapack['class_id']

        pt = get_click_prompt(datapack, opt)
        bbox = torch.as_tensor(datapack['bbox'], dtype=torch.float32, device=opt.device)

        with torch.no_grad():
            pred = model(imgs, pt, bbox)

        val_loss = criterion(pred, masks)
        val_losses += val_loss.item()

        if args.modelname == 'MSA' or args.modelname == 'SAM':
            gt = masks.detach().cpu().numpy()
        else:
            gt = label.detach().cpu().numpy()
        gt = gt[:, 0, :, :]

        # predict = torch.sigmoid(pred['masks'])
        # predict = predict.detach().cpu().numpy()  # (b, c, h, w)
        # seg = predict[:, 0, :, :] > 0.5  # (b, h, w)

        predict = F.softmax(pred['masks'], dim=1)
        pred = predict.detach().cpu().numpy()  # (b, c, h, w)
        seg = np.argmax(pred, axis=1)

        b, h, w = seg.shape
        for j in range(0, b):
            patient_number = int(image_filename[j][:4]) # xxxx_2CH_xxx
            antrum = int(image_filename[j][5])
            if antrum == 2:
                patientid = patient_number
            elif antrum == 3:
                patientid = 2000 + patient_number
            else:
                patientid = 4000 + patient_number
            flag[patientid] = flag[patientid] + 1
            pred_i = np.zeros((1, h, w))
            pred_i[seg[j:j+1, :, :] == 1] = 255
            gt_i = np.zeros((1, h, w))
            gt_i[gt[j:j+1, :, :] == 1] = 255
            tp, fp, tn, fn = metrics.get_matrix(pred_i, gt_i)
            hd = hausdorff_distance(pred_i[0, :, :], gt_i[0, :, :], distance="manhattan")
            hds[patientid, class_id[j]] += hd
            tps[patientid, class_id[j]] += tp
            fps[patientid, class_id[j]] += fp
            tns[patientid, class_id[j]] += tn
            fns[patientid, class_id[j]] += fn
            if opt.visual:
                visual_segmentation(seg[j:j+1, :, :], image_filename[j], opt)
    tps = tps[flag > 0, :]
    fps = fps[flag > 0, :]
    tns = tns[flag > 0, :]
    fns = fns[flag > 0, :]
    hds = hds[flag > 0, :] / (flag[flag>0][:, None]/(opt.classes-1))
    patient_dices = (2 * tps + 1e-5) / (2 * tps + fps + fns + 1e-5)  # p c
    dices = np.mean(patient_dices, axis=0)  # c
    hdis = np.mean(hds, axis=0)
    val_losses = val_losses / (batch_idx + 1)
    mean_dice = np.mean(dices[1:], axis=0)
    mean_hdis = np.mean(hdis[1:], axis=0)
    #return dices, mean_dice, val_losses
    if opt.mode=="train":
        return dices, mean_dice, mean_hdis, val_losses
    else:
        dice_mean = np.mean(patient_dices, axis=0)
        dices_std = np.std(patient_dices, axis=0)
        hd_mean = np.mean(hds, axis=0)
        hd_std = np.std(hds, axis=0)
        smooth = 0.00001
        iou = (tps + smooth) / (fps + tps + fns + smooth) # p c
        iou_mean = np.mean(iou, axis=0) 
        iou_std = np.std(iou, axis=0)
        acc = (tps + tns + smooth)/(tps + fps + fns + tns + smooth)
        acc_mean = np.mean(acc, axis=0)
        acc_std = np.std(acc, axis=0)
        se = (tps + smooth) / (tps + fns + smooth)
        se_mean = np.mean(se, axis=0)
        se_std = np.std(se, axis=0)
        sp = (tns + smooth) / (fps + tns + smooth)
        sp_mean = np.mean(sp, axis=0)
        sp_std = np.std(sp, axis=0)
        return dice_mean, hd_mean, iou_mean, acc_mean, se_mean, sp_mean, dices_std, hd_std, iou_std, acc_std, se_std, sp_std


def eval_patient(valloader, model, criterion, opt, args):
    model.eval()
    val_losses, mean_dice = 0, 0
    dices = np.zeros(opt.classes)
    patientnumber = 5000  # maxnum patient number
    flag = np.zeros(patientnumber)  # record the patients
    tps, fps = np.zeros((patientnumber, opt.classes)), np.zeros((patientnumber, opt.classes))
    tns, fns = np.zeros((patientnumber, opt.classes)), np.zeros((patientnumber, opt.classes))
    hds = np.zeros((patientnumber, opt.classes))
    for batch_idx, (datapack) in enumerate(valloader):
        imgs = Variable(datapack['image'].to(dtype = torch.float32, device=opt.device))
        masks = Variable(datapack['low_mask'].to(dtype = torch.float32, device=opt.device))
        label = Variable(datapack['label'].to(dtype = torch.float32, device=opt.device))
        image_filename = datapack['image_name']
        class_id = datapack['class_id']

        pt = get_click_prompt(datapack, opt)
        bbox = torch.as_tensor(datapack['bbox'], dtype=torch.float32, device=opt.device)

        with torch.no_grad():
            pred = model(imgs, pt, bbox)

        val_loss = criterion(pred, masks)
        val_losses += val_loss.item()

        if args.modelname == 'MSA' or args.modelname == 'SAM':
            gt = masks.detach().cpu().numpy()
        else:
            gt = label.detach().cpu().numpy()
        gt = gt[:, 0, :, :]

        predict = torch.sigmoid(pred['masks'])
        predict = predict.detach().cpu().numpy()  # (b, c, h, w)
        seg = predict[:, 0, :, :] > 0.5  # (b, h, w)
        

        # predict = F.softmax(pred['masks'], dim=1)
        # pred = predict.detach().cpu().numpy()  # (b, c, h, w)
        # seg = np.argmax(pred, axis=1)

        b, h, w = seg.shape
        for j in range(0, b):
            patientid = int(obtain_patien_id(image_filename[j]))
            flag[patientid] = flag[patientid] + 1
            pred_i = np.zeros((1, h, w))
            pred_i[seg[j:j+1, :, :] == 1] = 255
            gt_i = np.zeros((1, h, w))
            gt_i[gt[j:j+1, :, :] == 1] = 255
            tp, fp, tn, fn = metrics.get_matrix(pred_i, gt_i)
            hd = hausdorff_distance(pred_i[0, :, :], gt_i[0, :, :], distance="manhattan")
            hds[patientid, class_id[j]] += hd
            tps[patientid, class_id[j]] += tp
            fps[patientid, class_id[j]] += fp
            tns[patientid, class_id[j]] += tn
            fns[patientid, class_id[j]] += fn
            if opt.visual:
                visual_segmentation(seg[j:j+1, :, :], image_filename[j], opt)
    tps = tps[flag > 0, :]
    fps = fps[flag > 0, :]
    tns = tns[flag > 0, :]
    fns = fns[flag > 0, :]
    hds = hds[flag > 0, :] / (flag[flag>0][:, None]/(opt.classes-1))
    patient_dices = (2 * tps + 1e-5) / (2 * tps + fps + fns + 1e-5)  # p c
    dices = np.mean(patient_dices, axis=0)  # c
    hdis = np.mean(hds, axis=0)
    val_losses = val_losses / (batch_idx + 1)
    mean_dice = np.mean(dices[1:], axis=0)
    mean_hdis = np.mean(hdis[1:], axis=0)
    #return dices, mean_dice, val_losses
    if opt.mode=="train":
        return dices, mean_dice, mean_hdis, val_losses
    else:
        dice_mean = np.mean(patient_dices, axis=0)
        dices_std = np.std(patient_dices, axis=0)
        hd_mean = np.mean(hds, axis=0)
        hd_std = np.std(hds, axis=0)
        smooth = 0.00001
        iou = (tps + smooth) / (fps + tps + fns + smooth) # p c
        iou_mean = np.mean(iou, axis=0) 
        iou_std = np.std(iou, axis=0)
        acc = (tps + tns + smooth)/(tps + fps + fns + tns + smooth)
        acc_mean = np.mean(acc, axis=0)
        acc_std = np.std(acc, axis=0)
        se = (tps + smooth) / (tps + fns + smooth)
        se_mean = np.mean(se, axis=0)
        se_std = np.std(se, axis=0)
        sp = (tns + smooth) / (fps + tns + smooth)
        sp_mean = np.mean(sp, axis=0)
        sp_std = np.std(sp, axis=0)
        return dice_mean, hd_mean, iou_mean, acc_mean, se_mean, sp_mean, dices_std, hd_std, iou_std, acc_std, se_std, sp_std


def eval_slice(valloader, model, criterion, opt, args):
    model.eval()
    val_losses, mean_dice = 0, 0
    max_slice_number = opt.batch_size * (len(valloader) + 1)
    dices = np.zeros((max_slice_number, opt.classes))
    hds = np.zeros((max_slice_number, opt.classes))
    ious, accs, ses, sps = np.zeros((max_slice_number, opt.classes)), np.zeros((max_slice_number, opt.classes)), np.zeros((max_slice_number, opt.classes)), np.zeros((max_slice_number, opt.classes))
    eval_number = 0
    sum_time = 0
    for batch_idx, (datapack) in enumerate(valloader):
        imgs = datapack['image'].to(dtype = torch.float32, device=opt.device)
        masks = datapack['low_mask'].to(dtype = torch.float32, device=opt.device)
        label = datapack['label'].to(dtype = torch.float32, device=opt.device)
        pt = get_click_prompt(datapack, opt)
        image_filename = datapack['image_name']

        with torch.no_grad():
            start_time = time.time()
            pred = model(imgs, pt)
            sum_time =  sum_time + (time.time()-start_time)

        val_loss = criterion(pred, masks)
        val_losses += val_loss.item()

        if args.modelname == 'MSA' or args.modelname == 'SAM':
            gt = masks.detach().cpu().numpy()
        else:
            gt = label.detach().cpu().numpy()
        gt = gt[:, 0, :, :]
        predict_masks = pred['masks']
        predict_masks = torch.softmax(predict_masks, dim=1)
        pred = predict_masks.detach().cpu().numpy()  # (b, c, h, w)
        seg = np.argmax(pred, axis=1)  # (b, h, w)
        b, h, w = seg.shape
        for j in range(0, b):
            pred_i = np.zeros((1, h, w))
            pred_i[seg[j:j+1, :, :] == 1] = 255
            gt_i = np.zeros((1, h, w))
            gt_i[gt[j:j+1, :, :] == 1] = 255
            dices[eval_number+j, 1] += metrics.dice_coefficient(pred_i, gt_i)
            iou, acc, se, sp = metrics.sespiou_coefficient2(pred_i, gt_i, all=False)
            ious[eval_number+j, 1] += iou
            accs[eval_number+j, 1] += acc
            ses[eval_number+j, 1] += se
            sps[eval_number+j, 1] += sp
            hds[eval_number+j, 1] += hausdorff_distance(pred_i[0, :, :], gt_i[0, :, :], distance="manhattan")
            del pred_i, gt_i
            if opt.visual:
                visual_segmentation_sets_with_pt(seg[j:j+1, :, :], image_filename[j], opt, pt[0][j, :, :])
        eval_number = eval_number + b
    dices = dices[:eval_number, :] 
    hds = hds[:eval_number, :] 
    ious, accs, ses, sps = ious[:eval_number, :], accs[:eval_number, :], ses[:eval_number, :], sps[:eval_number, :]
    val_losses = val_losses / (batch_idx + 1)

    dice_mean = np.mean(dices, axis=0)
    dices_std = np.std(dices, axis=0)
    hd_mean = np.mean(hds, axis=0)
    hd_std = np.std(hds, axis=0)

    mean_dice = np.mean(dice_mean[1:])
    mean_hdis = np.mean(hd_mean[1:])
    print("test speed", eval_number/sum_time)
    if opt.mode == "train":
        return dices, mean_dice, mean_hdis, val_losses
    else:
        # data = pd.DataFrame(dices*100)
        # writer = pd.ExcelWriter('./result/' + args.task + '/PT10-' + opt.modelname + '.xlsx')
        # data.to_excel(writer, 'page_1', float_format='%.2f')
        # writer._save()

        dice_mean = np.mean(dices*100, axis=0)
        dices_std = np.std(dices*100, axis=0)
        hd_mean = np.mean(hds, axis=0)
        hd_std = np.std(hds, axis=0)
        iou_mean = np.mean(ious*100, axis=0)
        iou_std = np.std(ious*100, axis=0)
        acc_mean = np.mean(accs*100, axis=0)
        acc_std = np.std(accs*100, axis=0)
        se_mean = np.mean(ses*100, axis=0)
        se_std = np.std(ses*100, axis=0)
        sp_mean = np.mean(sps*100, axis=0)
        sp_std = np.std(sps*100, axis=0)
        return dice_mean, hd_mean, iou_mean, acc_mean, se_mean, sp_mean, dices_std, hd_std, iou_std, acc_std, se_std, sp_std
from einops import rearrange
from tqdm import tqdm

def post_process(match_output):
    preds, targets = match_output['pred_boxes'], match_output['target_boxes']
    # prob = F.softmax(out_logits, -1)
    
    
def eval_learnable_block(valloader, model, criterion, opt, args, tartet_size = 256):
    model.eval()
    criterion.eval()
    max_slice_number = len(valloader) #opt.batch_size * (len(valloader) + 1)
    losses_ce = 0. #np.zeros((len(valloader)))
    losses_bbox = 0. #np.zeros((len(valloader)))
    losses_giou = 0. #np.zeros((len(valloader)))
    val_losses = 0.
    gt_boxes, pred_boxes = torch.empty([0, 7]).to(opt.device), torch.empty([0, 6]).to(opt.device)
    for batch_idx, (imgs, prompts, targets) in enumerate(tqdm(valloader)):
        imgs = imgs.to(opt.device)
        targets = [{k: v.to(opt.device) for k, v in t.items()} for t in targets]
        prompts = {key:prompts[key].to(opt.device) for key in prompts}
        pt = (prompts['pts'], prompts['p_labels'])
        # -------------------------------------------------------- forward --------------------------------------------------------
        with torch.no_grad():
            pred = model(imgs, pt)      # {'pred_logits' : [bs, N, 2], 'pred_boxes' : [bs, N, 4]}
        loss_dict, boxes_for_metrics = criterion(pred, targets)
        weight_dict = criterion.weight_dict
        losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)
        losses_ce += loss_dict['loss_ce'].item()
        losses_bbox += loss_dict['loss_bbox'].item()
        losses_giou += loss_dict['loss_giou'].item()
        val_losses += losses.item()
        #** result화
        
        # out_logits, out_bbox = pred['pred_logits'], pred['pred_boxes']
        post_process(boxes_for_metrics)
        
        # orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
        # img_h, img_w = orig_target_sizes.unbind(1)
        # scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1)
        # # boxes = box_ops.box_cxcywh_to_xyxy(out_bbox)
        # boxes = out_bbox * scale_fct[:, None, :]
        # scores, labels = prob[..., :-1].max(-1)
        
        # # boxes = boxes * tartet_size
        
        # pred = torch.concat([boxes.flatten(0,1), labels.flatten(0,1)[:,None], scores.flatten(0,1)[:,None]], dim=1)
        
        # pred_boxes = torch.concat([pred_boxes, pred], dim=0)
        # gt_boxes = torch.concat([gt_boxes, boxes_for_metrics['target_boxes']], dim=0)
        
        # gt = torch.concat([torch.concat([target['boxes'], target['labels'][:, None]], dim=1) for target in targets], dim = 0)
    metric_fn = MetricBuilder.build_evaluation_metric("map_2d", async_mode=True, num_classes=1)
    metric_fn.add(np.array(pred_boxes.cpu().detach()), np.array(gt_boxes.cpu().detach()))
    coco_matric = metric_fn.value(iou_thresholds=np.arange(0.5, 1.0, 0.05), recall_thresholds=np.arange(0., 1.01, 0.01), mpolicy='soft')['mAP']
    return (losses_ce / max_slice_number), (losses_bbox / max_slice_number), (losses_bbox / max_slice_number), (val_losses / max_slice_number), coco_matric
    
def eval_camus_samed(valloader, model, criterion, opt, args):
    model.eval()
    val_losses, mean_dice = 0, 0
    classes = 4
    dices = np.zeros(classes)
    patientnumber = 6000  # maxnum patient number
    flag = np.zeros(patientnumber)  # record the patients
    tps, fps = np.zeros((patientnumber, classes)), np.zeros((patientnumber, classes))
    tns, fns = np.zeros((patientnumber, classes)), np.zeros((patientnumber, classes))
    hds = np.zeros((patientnumber, classes))
    eval_number = 0
    sum_time = 0
    for batch_idx, (datapack) in enumerate(valloader):
        imgs = datapack['image'].to(dtype = torch.float32, device=opt.device)
        masks = datapack['low_mask'].to(dtype = torch.float32, device=opt.device)
        label = datapack['label'].to(dtype = torch.float32, device=opt.device)
        image_filename = datapack['image_name']
        class_id = datapack['class_id']
        
        pt = get_click_prompt(datapack, opt)
        bbox = torch.as_tensor(datapack['bbox'], dtype=torch.float32, device=opt.device)

        with torch.no_grad():
            start_time = time.time()
            pred = model(imgs, pt, bbox)
            sum_time =  sum_time + (time.time()-start_time)

        val_loss = criterion(pred, masks)
        val_losses += val_loss.item()

        if args.modelname == 'MSA' or args.modelname == 'SAM':
            gt = masks.detach().cpu().numpy()
        else:
            gt = label.detach().cpu().numpy()
        gt = gt[:, 0, :, :]
        predict_masks = pred['masks']
        predict_masks = torch.softmax(predict_masks, dim=1)
        pred = predict_masks.detach().cpu().numpy()  # (b, c, h, w)
        seg = np.argmax(pred, axis=1)  # (b, h, w)
        b, h, w = seg.shape
        for j in range(0, b):
            patient_number = int(image_filename[j][:4]) # xxxx_2CH_xxx
            antrum = int(image_filename[j][5])
            if antrum == 2:
                patientid = patient_number
            elif antrum ==3:
                patientid = 2000 + patient_number
            else:
                patientid = 4000 + patient_number
            flag[patientid] = flag[patientid] + 1
            pred_i = np.zeros((1, h, w))
            pred_i[seg[j:j+1, :, :] == 1] = 255
            gt_i = np.zeros((1, h, w))
            gt_i[gt[j:j+1, :, :] == 1] = 255
            tp, fp, tn, fn = metrics.get_matrix(pred_i, gt_i)
            hd = hausdorff_distance(pred_i[0, :, :], gt_i[0, :, :], distance="manhattan")
            hds[patientid, class_id[j]] += hd
            tps[patientid, class_id[j]] += tp
            fps[patientid, class_id[j]] += fp
            tns[patientid, class_id[j]] += tn
            fns[patientid, class_id[j]] += fn
            if opt.visual:
                visual_segmentation(seg[j:j+1, :, :], image_filename[j], opt)
        eval_number = eval_number + b
    tps = tps[flag > 0, :]
    fps = fps[flag > 0, :]
    tns = tns[flag > 0, :]
    fns = fns[flag > 0, :]
    hds = hds[flag > 0, :] / (flag[flag>0][:, None]/(opt.classes-1))
    patient_dices = (2 * tps + 1e-5) / (2 * tps + fps + fns + 1e-5)  # p c
    dices = np.mean(patient_dices, axis=0)  # c
    hdis = np.mean(hds, axis=0)
    val_losses = val_losses / (batch_idx + 1)
    mean_dice = np.mean(dices[1:], axis=0)
    mean_hdis = np.mean(hdis[1:], axis=0)
    #return dices, mean_dice, val_losses
    print("test speed", eval_number/sum_time)
    if opt.mode=="train":
        return dices, mean_dice, mean_hdis, val_losses
    else:
        dice_mean = np.mean(patient_dices, axis=0)
        dices_std = np.std(patient_dices, axis=0)
        hd_mean = np.mean(hds, axis=0)
        hd_std = np.std(hds, axis=0)
        smooth = 0.00001
        iou = (tps + smooth) / (fps + tps + fns + smooth) # p c
        iou_mean = np.mean(iou, axis=0) 
        iou_std = np.std(iou, axis=0)
        acc = (tps + tns + smooth)/(tps + fps + fns + tns + smooth)
        acc_mean = np.mean(acc, axis=0)
        acc_std = np.std(acc, axis=0)
        se = (tps + smooth) / (tps + fns + smooth)
        se_mean = np.mean(se, axis=0)
        se_std = np.std(se, axis=0)
        sp = (tns + smooth) / (fps + tns + smooth)
        sp_mean = np.mean(sp, axis=0)
        sp_std = np.std(sp, axis=0)
        return dice_mean, hd_mean, iou_mean, acc_mean, se_mean, sp_mean, dices_std, hd_std, iou_std, acc_std, se_std, sp_std

def get_eval(valloader, model, criterion, opt, args):
    if args.modelname == "SAMed":
        if opt.eval_mode == "camusmulti":
            opt.eval_mode = "camus_samed"
        else:
            opt.eval_mode = "slice"
    elif args.modelname == 'LearableBlock':
        opt.eval_mode = 'learnable'
    
    if opt.eval_mode == 'learnable':
        return eval_learnable_block(valloader, model, criterion, opt, args)    
    if opt.eval_mode == "mask_slice":
        return eval_mask_slice2(valloader, model, criterion, opt, args)
    elif opt.eval_mode == "slice":
        return eval_slice(valloader, model, criterion, opt, args)
    elif opt.eval_mode == "camusmulti":
        return eval_camus_patient(valloader, model, criterion, opt, args)
    elif opt.eval_mode == "patient":
        return eval_patient(valloader, model, criterion, opt, args)
    elif opt.eval_mode == "camus_samed":
        return eval_camus_samed(valloader, model, criterion, opt, args)
    else:
        raise RuntimeError("Could not find the eval mode:", opt.eval_mode)