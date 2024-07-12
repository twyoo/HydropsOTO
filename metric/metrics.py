import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class avg_metrics(object):
    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def dice_cal(seg, gt, ratio=0.5):
    seg = seg.flatten()
    gt = gt.flatten()
    seg[seg > ratio] = np.float32(1)
    seg[seg < ratio] = np.float32(0)
   
    denominator = float(gt.sum() + seg.sum())
    if denominator == 0:
        return 0.0  
    dice = float(2 * (gt * seg).sum()) / denominator
    
    return dice

def iou_cal(seg, gt, ratio=0.5):
    seg = seg.flatten()
    gt = gt.flatten()
    seg[seg > ratio] = np.float32(1)
    seg[seg < ratio] = np.float32(0)
    
    denominator = float(gt.sum() + seg.sum())
    if denominator == 0:
        return 0.0  
    iou = float((gt * seg).sum()) / (float(gt.sum() + seg.sum()) - float((gt * seg).sum()))
        
    return iou

def pixel_cal(mask_image, space):
    pixels = np.sum(mask_image)
    
    return pixels

def volume_cal(mask_image, space):
    voxel = np.prod(space)               
    volume = voxel*np.sum(mask_image)
    
    return volume


def calculate_dice(pred, target, C): 
    target = target.long()
    pred = pred.long()
    
    N = pred.shape[0]
    target_mask = target.data.new(N, C).fill_(0)   
    target_mask.scatter_(1, target, 1.)    
    
    pred_mask = pred.data.new(N, C).fill_(0)
    pred_mask.scatter_(1, pred, 1.) 

    intersection= pred_mask * target_mask
    summ = pred_mask + target_mask

    intersection = intersection.sum(0).type(torch.float32)
    summ = summ.sum(0).type(torch.float32)
    
    summ += 1e-5 
    dice = 2 * intersection / summ

    return dice, intersection, summ    

def calculate_iou(pred, target, C): 
    target = target.long()
    pred = pred.long()
    
    N = pred.shape[0]
    target_mask = target.data.new(N, C).fill_(0)
    target_mask.scatter_(1, target, 1.) 

    pred_mask = pred.data.new(N, C).fill_(0)
    pred_mask.scatter_(1, pred, 1.) 

    intersection= pred_mask * target_mask
    summ = pred_mask + target_mask

    intersection = intersection.sum(0).type(torch.float32)
    summ = summ.sum(0).type(torch.float32)
    
    summ += 1e-5 
    iou = intersection / (summ - intersection)
    
    return iou, intersection, summ