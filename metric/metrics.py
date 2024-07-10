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

