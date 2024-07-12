import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

import os
import sys
import random
import time
import yaml
import logging
import argparse
import shutil
import warnings
import numpy as np
from tqdm import tqdm

from datetime import datetime, timedelta
from model.oto3DUnet import oto3DUnet
from utils.utils import (
    fix_seed, FocalTverskyLoss,
    check_dir_createFolder, inference_sliding_window, 
    averageMeter, progressMeter, configure_logger, save_configure
)
from metric.metrics import calculate_dice, calculate_iou

warnings.filterwarnings("ignore", category=UserWarning)  

from monai.transforms import (
    EnsureChannelFirstd,
    Compose,
    LoadImaged,
    ScaleIntensityRanged,
    Rand3DElasticd,
    RandAffined,
    RandGaussianNoised,
    RandAdjustContrastd,
    RandFlipd,
)

class otoDataSet(Dataset):
    def __init__(self, args, phase='train', k_fold=5, k=0, seed=0):
        self.phase = phase
        self.args = args
        self.rand_affine_dict = RandAffined(
                keys=["image", "label"],
                mode=("bilinear", "nearest"),
                prob=1.0,                        
                translate_range=(10, 10, 6),      
                rotate_range=(np.pi / 18, np.pi / 18, np.pi / 36), 
                scale_range=(0.15, 0.15, 0.15),
                padding_mode="border",
            )

        self.rand_elastic_dict = Rand3DElasticd(
                keys=["image", "label"],
                mode=("bilinear", "nearest"),
                prob=1.0,
                sigma_range=(5, 8),
                magnitude_range=(50, 100),
                padding_mode="border",
            )

        self.rand_transform_dict = Compose([
                LoadImaged(keys=("image", "label"), image_only=False),
                EnsureChannelFirstd(keys=["image", "label"]),
                RandGaussianNoised(keys=["image"], prob=0.5, mean=0.0, std=0.1), 
                RandAdjustContrastd(keys=["image"], prob=0.5, gamma=(0.9, 1.1)),   
                self.rand_affine_dict,
                self.rand_elastic_dict,
                RandFlipd(keys=["image", "label"], spatial_axis=[0, 1], prob=0.5), 
                ScaleIntensityRanged(keys=["image"], a_min=0, a_max=255, b_min=0.0, b_max=1.0, clip=True),
            ])
        
        self.rand_transform_dict_test = Compose([
                LoadImaged(keys=("image", "label"), image_only=False),
                EnsureChannelFirstd(keys=["image", "label"]),
                ScaleIntensityRanged(keys=["image"], a_min=0, a_max=255, b_min=0.0, b_max=1.0, clip=True),
            ])
        
        assert phase in ['train', 'valid']   
        
        with open(os.path.join('./data/target_dirlist', 'dataset.yaml'), 'r') as f:
            dir_name_list = yaml.load(f, Loader=yaml.SafeLoader)
        
        random.Random(seed).shuffle(dir_name_list)

        length = len(dir_name_list)  
        test_name_list = dir_name_list[k*(length//k_fold) : (k+1)*(length//k_fold)]  
        train_name_list = list(set(dir_name_list) - set(test_name_list))    
        logging.info(f'Length of train data: {len(train_name_list)}')
        logging.info(f'Length of test data: {len(test_name_list)}')
        
        if phase == 'train':
            img_name_list = train_name_list
            logging.info(f'Start loading {self.phase} data : {len(train_name_list)}')
        else:
            img_name_list = test_name_list
            logging.info(f'Start loading {self.phase} data : {len(test_name_list)}')

        logging.info(f'Start loading {self.phase} data')

        self.np_mrcImg_list = []
        self.np_mrcMsk_list = []
        self.np_mi2Img_list = []
        self.np_mi2Msk_list = []
        self.spacing_list = []
        
        for idx, directory in enumerate(img_name_list):
            np_mrcImg = os.path.join(directory, '22 MRC/cropImage.npy')
            np_mrcMsk = os.path.join(directory, '22 MRC/cropMask.npy')
            np_mi2Img = os.path.join(directory, '26 MI2/cropImage.npy')
            np_mi2Msk = os.path.join(directory, '26 MI2/cropMask.npy')
            self.np_mrcImg_list.append(np_mrcImg)
            self.np_mrcMsk_list.append(np_mrcMsk)
            self.np_mi2Img_list.append(np_mi2Img)
            self.np_mi2Msk_list.append(np_mi2Msk)
            
            self.spacing_list.append(args.spacing)
             
        logging.info(f"Load done, length of {phase} dataset: {len(self.np_mrcImg_list)}")   
 
    def __len__(self):    
        if self.phase == 'train':
            return len(self.np_mrcImg_list) * self.args.len_dataset  
        else:
            return len(self.np_mrcImg_list)    
            
    def __getitem__(self, idx):
        idx = idx % len(self.np_mrcImg_list)     
        
        mrcImg = self.np_mrcImg_list[idx] 
        mrcMsk = self.np_mrcMsk_list[idx] 
        mi2Img = self.np_mi2Img_list[idx] 
        mi2Msk = self.np_mi2Msk_list[idx] 
        
        data_mrc_dicts = {"image": mrcImg, "label": mrcMsk}
        data_mi2_dicts = {"image": mi2Img, "label": mi2Msk}
        
        if self.phase == 'train':
            transformed_dict_mrc = self.rand_transform_dict(data_mrc_dicts) 
            aug_img_mrc, aug_lbl_mrc = transformed_dict_mrc["image"], transformed_dict_mrc["label"]
            transformed_dict_mi2 = self.rand_transform_dict({"image": mi2Img, "label": mi2Msk}) 
            aug_img_mi2, aug_lbl_mi2 = transformed_dict_mi2["image"], transformed_dict_mi2["label"]
        else:
            transformed_dict_mrc = self.rand_transform_dict_test(data_mi2_dicts) 
            aug_img_mrc, aug_lbl_mrc = transformed_dict_mrc["image"], transformed_dict_mrc["label"]
            transformed_dict_mi2 = self.rand_transform_dict_test({"image": mi2Img, "label": mi2Msk}) 
            aug_img_mi2, aug_lbl_mi2 = transformed_dict_mi2["image"], transformed_dict_mi2["label"]
        
        concatCH2_img = np.concatenate([aug_img_mrc, aug_img_mi2])  
        concatCH2_lab = np.concatenate([aug_lbl_mrc, aug_lbl_mi2])  
        concatCH2_img = concatCH2_img.transpose(0,3,1,2) 
        concatCH2_lab = concatCH2_lab.transpose(0,3,1,2) 
        
        data = concatCH2_img.astype(np.float32)
        mask = concatCH2_lab.astype(np.float32)
        
        image_tensor = torch.from_numpy(data).float()        
        label_tensor = torch.from_numpy(mask).to(torch.int8) 
        
        if self.phase == 'train':
            return image_tensor, label_tensor
        else:
            return image_tensor, label_tensor, np.array(self.spacing_list[idx])
      
def validation(model, dataloader, args, fold_idx, epoch):
    model.eval()

    mrcdice_list = []
    mrcIOU_list = []
        
    mi2dice_list = []
    mi2IOU_list = []
    
    for i in range(args.classes-1): 
        mrcdice_list.append([])    
        
    for i in range(args.classes-1): 
        mi2dice_list.append([])          
        mi2IOU_list.append([])
    
    logging.info("Evaluating")
    
    check_dir_createFolder(args.validation_dir + f'/{fold_idx}/{epoch}') 
    print(f"check_dir_createFolder {fold_idx}/{epoch}")
    
    fn_class = lambda x: 1.0 * (x > 0.5)
    
    idx = 0
    with torch.no_grad():
        iterator = tqdm(dataloader)   
        for (images, labels, spacing) in iterator:   
            images, labels = images.cuda(), labels.cuda()

            pred = inference_sliding_window(model, images, args) 
            image_np = images.squeeze().detach().cpu().numpy()
            label_np = labels.squeeze().detach().cpu().numpy()
            label_pred_np = pred.squeeze().detach().cpu().numpy()

            label_pred_np = fn_class(label_pred_np)
            mrcOutput = label_pred_np[0,:,:,:]
            mi2Output = label_pred_np[1,:,:,:]
            mrcLabel = label_np[0,:,:,:]    
            mi2Label = label_np[1,:,:,:]
        
            np.save(os.path.join(args.validation_dir, f'{fold_idx}/{epoch}', f'input_np_{idx}.npy'), image_np)
            np.save(os.path.join(args.validation_dir, f'{fold_idx}/{epoch}', f'label_np_{idx}.npy'), label_np)
            np.save(os.path.join(args.validation_dir, f'{fold_idx}/{epoch}', f'label_pred_np_{idx}.npy'), label_pred_np)
            idx += 1

            torch_mrcOutput = torch.from_numpy(mrcOutput)
            torch_mrcLabel =  torch.from_numpy(mrcLabel)
            torch_mi2Output = torch.from_numpy(mi2Output)
            torch_mi2Label =  torch.from_numpy(mi2Label)
            
            mrc = torch_mrcLabel.flatten()
            mi2 = torch_mi2Label.flatten()
            if mrc.sum() <= 0 or mi2.sum() <= 0:
                raise ValueError("mrclabel.sum() or mi2label.sum() : Zero......")
            
            mrcdice, _, _ = calculate_dice(torch_mrcOutput.view(-1, 1), torch_mrcLabel.view(-1, 1), args.classes)
            mi2dice, _, _ = calculate_dice(torch_mi2Output.view(-1, 1), torch_mi2Label.view(-1, 1), args.classes)
            mrcIOU, _, _ = calculate_iou(torch_mrcOutput.view(-1, 1), torch_mrcLabel.view(-1, 1), args.classes)
            mi2IOU, _, _ = calculate_iou(torch_mi2Output.view(-1, 1), torch_mi2Label.view(-1, 1), args.classes)
           
            mrcdice = mrcdice.cpu().numpy()[1:]  
            mi2dice = mi2dice.cpu().numpy()[1:]  

            mrcIOU = mrcIOU.cpu().numpy()[1:]
            mi2IOU = mi2IOU.cpu().numpy()[1:]
            
            unique_cls = torch.unique(labels)   
            for cls in range(0, args.classes-1):
                if cls+1 in unique_cls:              
                    mrcdice_list[cls].append(mrcdice[cls])

                    mrcIOU_list[cls].append(mrcIOU[cls])
                    
                    mi2dice_list[cls].append(mi2dice[cls])
                    mi2IOU_list[cls].append(mi2IOU[cls])
                       
    mrcout_dice, mrcout_IOU = [], []
    mi2out_dice, mi2out_IOU = [], []
    
    for cls in range(0, args.classes-1):
        mrcout_dice.append(np.array(mrcdice_list[cls]).mean())
        mrcout_IOU.append(np.array(mrcIOU_list[cls]).mean())
       
        mi2out_dice.append(np.array(mi2dice_list[cls]).mean())
        mi2out_IOU.append(np.array(mi2IOU_list[cls]).mean())
   
    return np.array(mrcout_dice), np.array(mrcout_IOU), np.array(mi2out_dice), np.array(mi2out_IOU) 

      
def parse_opt():
    parser = argparse.ArgumentParser(description='HydropsOTO Medical Image Segmentation') 
    parser.add_argument('--dataset', type=str, default='oto', help='dataset name')
    parser.add_argument('--model', type=str, default='3DU-Net', help='model name')
    parser.add_argument('--dimension', type=str, default='2d', help='2d model or 3d model')
    parser.add_argument('--pretrain', action='store_true', help='if use pretrained weight for init')
    parser.add_argument('--batch_size', type=int, default=8, help='batch size')
    parser.add_argument('--resume', action='store_true', help='if resume training from checkpoint')
    parser.add_argument('--load', type=str, default=False, help='load pretrained model')
    parser.add_argument('--config_path', type=str, default='./config', help='config path')
    parser.add_argument('--config_file', type=str, default='config_oto.yaml', help='config file')    
    parser.add_argument('--expresult_path', type=str, default='./exp', help='experiment result path')
    parser.add_argument('--checkpoint_path', type=str, default='./exp/checkpoint', help='checkpoint path')
    parser.add_argument('--log_path', type=str, default='./log', help='log path')
    parser.add_argument('--unique_name', type=str, default='test', help='unique experiment name')
    parser.add_argument('--gpu', type=str, default='0')
    parser.add_argument('--save_opt', default=True, action='store_false', help='save option')

    args = parser.parse_args()
    
    args.num_workers = 4 * args.batch_size                                   
    config_path = os.path.join(args.config_path, args.config_file)
    if not os.path.exists(args.config_path):
        raise ValueError("The specified configuration doesn't exist: %s"%config_path)
    else:
        print('Loading configurations from %s'%config_path)
        with open(config_path, 'r') as f:
            config = yaml.load(f, Loader=yaml.SafeLoader)  
                                
        for key, value in config.items():
            setattr(args, key, value)       
    
    return args

def train_model(model, args, fold_idx):    
    phase = 'train'
    trainset = otoDataSet(args, phase=phase, k_fold=args.k_fold, k=fold_idx, seed=args.split_seed)
    train_loader = DataLoader(      
        trainset, 
        batch_size=args.batch_size,
        shuffle=True, 
        pin_memory=(args.aug_device != 'gpu'), 
        num_workers=args.num_workers,           
        persistent_workers=(args.num_workers>0)
    )
    validset =  otoDataSet(args, phase='valid', k_fold=args.k_fold, k=fold_idx)   
    valid_loader = DataLoader(validset, batch_size=1, 
                                pin_memory=True, 
                                shuffle=False, 
                                drop_last=True,
                                num_workers=args.num_workers) 
    
    logging.info(f"Created Dataset and DataLoader")

    writer = SummaryWriter(f"{args.log_path}/{args.unique_name}/fold_{fold_idx}")  ## log_path='./log/'

    criterion = FocalTverskyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.base_lr, betas=args.betas, weight_decay=args.weight_decay, eps=1e-5) 
    
    mrcbest_Dice = np.zeros(args.classes)   
    mrcbest_IOU = np.zeros(args.classes)
    
    mi2best_Dice = np.zeros(args.classes)
    mi2best_IOU = np.zeros(args.classes)
    
    for epoch in range(args.start_epoch, args.num_epochs):
        logging.info(f"Starting epoch {epoch+1}/{args.num_epochs}")
        exp_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.97)  
        ##exp_scheduler = exp_lr_scheduler_with_warmup(optimizer, init_lr=args.base_lr, epoch=epoch, warmup_epoch=5, max_epoch=args.num_epochs)  
        logging.info(f"Current lr: {exp_scheduler:.4e}")
        
        batch_time = averageMeter("Time:", ":4.2f")
        epoch_loss = averageMeter("Loss:", ":.6f")
        progress = progressMeter(
            len(train_loader) if args.dimension=='2d' else args.iter_per_epoch, 
            [batch_time, epoch_loss], 
            prefix=f"   Fold_idx: [{fold_idx}]   Epoch: [{epoch+1}]",
        )
        
        model.train() 
  
        tic = time.time()
        iter_num_per_epoch = 0
        for idx, (inputs, labels) in enumerate(train_loader):
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            step = idx + epoch * len(train_loader)
            optimizer.zero_grad()
            
            result = model(inputs)
            loss = 0 
            if isinstance(result, tuple) or isinstance(result, list):
                for j in range(len(result)):
                    loss += args.aux_weight[j] * (criterion(result[j], labels.squeeze(1))) 
            else:
                labels = labels.squeeze()
                result = result.squeeze()
                loss = criterion(result, labels)  
                        
            if phase == 'train':
                loss.backward()
                optimizer.step()
            
            epoch_loss.update(loss.item(), inputs.shape[0])
            batch_time.update(time.time() - tic)
            tic = time.time()
            
            if idx % args.print_freq == 0:   
                progress.display(idx)

            if args.dimension == '3d':
                iter_num_per_epoch += 1
                if iter_num_per_epoch > args.iter_per_epoch:     
                    break

            writer.add_scalar('Train/Loss', epoch_loss.avg, epoch+1)
            
        torch.save({
            'epoch': epoch+1,
            'model_state_dict': model.state_dict() if not args.torch_compile else model._orig_mod.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, f"{args.checkpoint_path}/{args.dataset}/{args.unique_name}/fold_{fold_idx}_latest.pth")
   
        if (epoch+1) % args.val_freq == 0:   
            mrcdice_list_test, mrcIOU_list_test, mi2dice_list_test, mi2IOU_list_test \
                         = validation(model, valid_loader, args, fold_idx, epoch)
           
            for cls in range(0, args.classes-1):
                print(f'mrcdice_list_test : {mrcdice_list_test[cls]:6.4f}')
                print(f'mrcdice_list_test.mean() : {mrcdice_list_test[cls].mean():6.4f}')
                print(f'mi2dice_list_test : {mi2dice_list_test[cls]:6.4f}')
                print(f'mi2dice_list_test.mean() : {mi2dice_list_test[cls].mean():6.4f}')
            
            if (mrcdice_list_test.mean() + mi2dice_list_test.mean()) > (mrcbest_Dice.mean() + mi2best_Dice.mean()):
                mrcbest_Dice = mrcdice_list_test
                mrcbest_IOU = mrcIOU_list_test
                
                mi2best_Dice = mi2dice_list_test
                mi2best_IOU = mi2IOU_list_test
                               
                torch.save({
                    'epoch': epoch+1,
                    'model_state_dict': model.state_dict() if not args.torch_compile else model._orig_mod.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }, f"{args.checkpoint_path}/{args.dataset}/{args.unique_name}/fold_{fold_idx}_best.pth")
            
            logging.info("Evaluation Done")
            logging.info(f"Dice: {mrcdice_list_test.mean():.4f}/Best Dice: {mrcbest_Dice.mean():.4f}")
    
        writer.add_scalar('LR', exp_scheduler, epoch+1)
    
    return mrcbest_Dice, mrcbest_IOU, mi2best_Dice, mi2best_IOU

   
if __name__ == "__main__":

    args = parse_opt() 
    args.data_root = 'DATA ROOT PATH' ## './data/dicom2022Crop'
 
    if args.random_seed is not None:          
        fix_seed(args.random_seed)

    if args.save_opt:
        run_id = datetime.now().strftime("%Y%m%d-%H%M")
        args.unique_name = os.path.join(args.task_name + "_" + run_id)  
        print(args.unique_name)
        
        expsave_path = os.path.join(args.checkpoint_path, args.dataset, args.unique_name)
        check_dir_createFolder(expsave_path)
    
        shutil.copyfile(  
            __file__, os.path.join(expsave_path, run_id + "_" + os.path.basename(__file__))
        )

    check_dir_createFolder(args.checkpoint_path)
    args.log_path = args.log_path + '%s/'%args.dataset    
    check_dir_createFolder(args.log_path)
    args.validation_save_path = 'validation_result'
    args.validation_dir = f"{args.expresult_path}/{args.dataset}/{args.unique_name}/{args.validation_save_path}"
    check_dir_createFolder(args.validation_dir)

    mrcDice_list, mrcIOU_list = [], []
    mi2Dice_list, mi2IOU_list = [], []
    
    start = time.time()
    for fold_idx in range(args.k_fold):   
        ts = time.time() ##
       
        configure_logger(0, args.checkpoint_path+f"/fold_{fold_idx}.txt") 
        save_configure(args)  
        
        logging.info(
            f"\nDataset: {args.dataset},\n" + f"Model: {args.model},\n" + f"Dimension: {args.dimension}"
        )
        
        model = oto3DUnet()
        if torch.cuda.is_available():
            print("torch.cuda.device_count : ", torch.cuda.device_count())
            device_ids = list(range(torch.cuda.device_count()))
            model = nn.DataParallel(model, device_ids=device_ids)
            device = torch.device('cuda')
        else:
            device = torch.device("cpu")        
        model.to(device)   
        logging.info(f"Created Model")    
        
        
        mrcbest_Dice, mrcbest_IOU, mi2best_Dice, mi2best_IOU = train_model(model, args, fold_idx=fold_idx) 

        logging.info(f"Training and evaluation on Fold {fold_idx} is done")
        te = time.time()   
        sec1 = te -ts
        fold_idx_time = timedelta(seconds=sec1)
        logging.info(f"Training and valuation on Fold {fold_idx} Time : {fold_idx_time}")   ##
        
        mrcDice_list.append(mrcbest_Dice)
        mrcIOU_list.append(mrcbest_IOU)
        
        mi2Dice_list.append(mi2best_Dice)
        mi2IOU_list.append(mi2best_IOU)  
        
    end = time.time()
    sec = end - start
    result = timedelta(seconds=sec)
    logging.info(f'{args.num_epochs} per Fold, {args.iter_per_epoch} per Epoch')
    logging.info(f'Total Training & Validation Time on Fold 1~{fold_idx+1}: {result}')    

    mrctotal_Dice = np.vstack(mrcDice_list)   
    mrctotal_IOU = np.vstack(mrcIOU_list)
    
    mi2total_Dice = np.vstack(mi2Dice_list)   
    mi2total_IOU = np.vstack(mi2IOU_list)
    
    with open(f"{args.checkpoint_path}/{args.dataset}/{args.unique_name}/cross_validation.txt",  'w') as f:
        np.set_printoptions(precision=4, suppress=True)   
        f.write('mrcDice\n')
        for i in range(args.k_fold):
            f.write(f"Fold {i}: {mrcDice_list[i]}\n")
        f.write(f"Each Class Dice Avg: {np.mean(mrctotal_Dice, axis=0)}\n")
        f.write(f"Each Class Dice Std: {np.std(mrctotal_Dice, axis=0)}\n")
        f.write("\n")
        
        f.write('mrcIOU\n')
        for i in range(args.k_fold):
            f.write(f"Fold {i}: {mrcIOU_list[i]}\n")
        f.write(f"Each Class IOU Avg: {np.mean(mrctotal_IOU, axis=0)}\n")
        f.write(f"Each Class IOU Std: {np.std(mrctotal_IOU, axis=0)}\n")
        f.write("\n\n")
        
        f.write('mi2Dice\n')
        for i in range(args.k_fold):
            f.write(f"Fold {i}: {mi2Dice_list[i]}\n")
        f.write(f"Each Class Dice Avg: {np.mean(mi2total_Dice, axis=0)}\n")
        f.write(f"Each Class Dice Std: {np.std(mi2total_Dice, axis=0)}\n")
        f.write("\n")
        
        f.write('mi2IOU\n')
        for i in range(args.k_fold):
            f.write(f"Fold {i}: {mi2IOU_list[i]}\n")
        f.write(f"Each Class IOU Avg: {np.mean(mi2total_IOU, axis=0)}\n")
        f.write(f"Each Class IOU Std: {np.std(mi2total_IOU, axis=0)}\n")

    sys.exit(0)
