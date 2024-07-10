import os
from glob import glob
import numpy as np
import pickle


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
