import os
from glob import glob
import numpy as np
import pickle
import logging
import random

import torch
import torch.nn as nn
import torch.nn.functional as F


def dump_file_pkl(data, file_name):
    with open(file_name, 'wb') as file:
        pickle.dump(data, file)

def load_file_pkl(file_name):
    with open(file_name, 'rb') as file:
        data = pickle.load(file)    
    return data

def create_check_dir(path):
    try:
        if not os.path.exists(path):
            os.makedirs(path)
    except OSError:
       print ('Error: Creating directory. ' +  path)

def create_concatDictList(imageList, maskList, ratio):  
    data_dicts = [
        {'image': image_name, 'mask': mask_name} 
        for image_name, mask_name in zip(imageList,  maskList)
    ]
    
    cut = int(ratio * len(data_dicts))
    train_dictlist = data_dicts[:cut]
    test_dictlist = data_dicts[cut:]

    return train_dictlist, test_dictlist

def create_DictList(imageList, maskList, ratio):
    data_dicts = [
        {'image': image_name, 'mask': mask_name} 
        for image_name, mask_name in zip(imageList,  maskList)
    ]

    cut = int(ratio * len(data_dicts))
    train_dictlist = data_dicts[:cut]
    test_dictlist = data_dicts[cut:]

    return train_dictlist, test_dictlist

def create_imageDictList(mrcimageList, mi2imageList): 
        
    image_dicts = [
        {'mrcimage': mrcimage_name, 'mi2image': mi2image_name} 
        for mrcimage_name, mi2image_name in zip(mrcimageList, mi2imageList)
    ]

    return image_dicts 

def create_maskDictList(mrcmaskList, mi2maskList): 
        
    mask_dicts = [
        {'mrcmask': mrcmask_name, 'mi2mask': mi2mask_name} 
        for mrcmask_name, mi2mask_name in zip(mrcmaskList, mi2maskList)
    ]

    return mask_dicts 

def split_idx(half_win, size, i):
    start_idx = half_win * i
    end_idx = start_idx + half_win*2

    if end_idx > size:
        start_idx = size - half_win*2
        end_idx = size

    return start_idx, end_idx

def inference_sliding_window(model, image, args):
    model.eval()

    B, C, D, H, W = image.shape

    win_d, win_h, win_w = args.window_size  

    flag = False
    if D < win_d or H < win_h or W < win_w:
        flag = True
        diff_D = max(0, win_d-D)
        diff_H = max(0, win_h-H)
        diff_W = max(0, win_w-W)

        image = F.pad(image, (0, diff_W, 0, diff_H, 0, diff_D))
        
        origin_D, origin_H, origin_W = D, H, W
        B, C, D, H, W = image.shape

    half_win_d = win_d // 2     ## 16 // 2 = 8
    half_win_h = win_h // 2     ## 192 // 2 = 96
    half_win_w = win_w // 2     ## 192 // 2 = 96

    pred_output = torch.zeros((B, args.classes, D, H, W)).to(image.device)    

    counter = torch.zeros((B, 1, D, H, W)).to(image.device)
    one_count = torch.ones((B, 1, win_d, win_h, win_w)).to(image.device)

    with torch.no_grad():
        for i in range(D // half_win_d):
            for j in range(H // half_win_h):
                for k in range(W // half_win_w):
                    
                    d_start_idx, d_end_idx = split_idx(half_win_d, D, i)
                    h_start_idx, h_end_idx = split_idx(half_win_h, H, j)
                    w_start_idx, w_end_idx = split_idx(half_win_w, W, k)

                    input_tensor = image[:, :, d_start_idx:d_end_idx, h_start_idx:h_end_idx, w_start_idx:w_end_idx]
                    
                    pred = model(input_tensor)

                    if isinstance(pred, tuple) or isinstance(pred, list):
                        pred = pred[0]

                    pred = F.softmax(pred, dim=1)  
                    pred_output[:, :, d_start_idx:d_end_idx, h_start_idx:h_end_idx, w_start_idx:w_end_idx] += pred
                    counter[:, :, d_start_idx:d_end_idx, h_start_idx:h_end_idx, w_start_idx:w_end_idx] += one_count

    pred_output /= counter
    
    if flag:
        pred_output = pred_output[:, :, :origin_D, :origin_H, :origin_W]

    return pred_output

def check_dir_createFolder(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)     
    except OSError:
        print ('Error: Creating directory. ' +  directory)

class averageMeter(object):
    def __init__(self, name, fmt=":f"):
        self.name = name
        self.fmt = fmt
        self.reset()

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

    def __str__(self): 
        fmtstr = "{name} {val" + self.fmt + "} ({avg" + self.fmt + "})"
        return fmtstr.format(**self.__dict__)

class progressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        logging.info("\t".join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1)) 
        fmt = "{:" + str(num_digits) + "d}"
        return "[" + fmt + "/" + fmt.format(num_batches) + "]"
    
def exp_lr_scheduler_with_warmup(optimizer, init_lr, epoch, warmup_epoch, max_epoch):
    
    if epoch >= 0 and epoch <= warmup_epoch and warmup_epoch != 0:
        lr = init_lr * 2.718 ** (10*(float(epoch) / float(warmup_epoch) - 1.))
        if epoch == warmup_epoch:
            lr = init_lr
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        return lr

    else:
        lr = init_lr * (1 - epoch / max_epoch)**0.9
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

    return lr

def is_master(args):
    return args.rank % args.ngpus_per_node == 0

def configure_logger(rank, log_path=None):
    LOG_FORMAT = "[%(levelname)s] %(asctime)s %(filename)s:%(lineno)s %(message)s"
    LOG_DATEFMT = "%Y-%m-%d %H:%M:%S"

    if log_path:
        log_dir = os.path.dirname(log_path)
        os.makedirs(log_dir, exist_ok=True)

    level = logging.INFO if rank in {-1, 0} else logging.WARNING     
    handlers = [logging.StreamHandler()]
    if rank in {0, -1} and log_path:
        handlers.append(logging.FileHandler(log_path, "w"))   

    logging.basicConfig(
        level=level,
        format=LOG_FORMAT,
        datefmt=LOG_DATEFMT,
        handlers=handlers,
        force=True,
    )

def save_configure(args):
    if hasattr(args, "distributed"):
        if (args.distributed and is_master(args)) or (not args.distributed):
            with open(f"{args.checkpoint_path}/config.txt", 'w') as f:
                for name in args.__dict__:
                    f.write(f"{name}: {getattr(args, name)}\n")
    else:
        with open(f"{args.checkpoint_path}/config.txt", 'w') as f:
            for name in args.__dict__:
                f.write(f"{name}: {getattr(args, name)}\n")
                
def fix_seed(random_seed):  
    random.seed(random_seed)      
    np.random.seed(random_seed)    
    torch.manual_seed(random_seed) 
    torch.cuda.manual_seed_all(random_seed)  

    if hasattr(torch, 'set_deterministic'):   
        torch.set_deterministic(True)        
    
    torch.backends.cudnn.benchmark = False     
    torch.backends.cudnn.deterministic = True  

ALPHA = 0.9  
BETA = 0.1         
GAMMA = 2 
class FocalTverskyLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(FocalTverskyLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1, alpha=ALPHA, beta=BETA, gamma=GAMMA):
        
        inputs = F.sigmoid(inputs)       
        inputs = inputs.view(-1)
        targets = targets.contiguous().view(-1, )
        
        TP = (inputs * targets).sum()    
        FP = ((1-targets) * inputs).sum()
        FN = (targets * (1-inputs)).sum()
        
        Tversky = (TP + smooth) / (TP + alpha*FN + beta*FP + smooth)  
        FocalTversky = (1 - Tversky)**gamma
                       
        return FocalTversky
                