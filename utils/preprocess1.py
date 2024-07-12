import os
from glob import glob

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

import SimpleITK as sitk
from utils.utils import create_check_dir
from metric.metrics import avg_metrics, volume_cal

import warnings
warnings.filterwarnings("ignore")

## MRC and Mi2 mask(label) data mask value
mrclandmark_value = 2   # MRC : landmark 2, pixel values 1, 0 in MRC mask 
mrcmask_value = 1
mi2mask_value = 5       # HYDROPS-Mi2 : pixel values 5, 0
mrclandmark_count = 6   # rightside 6 + leftside 6 = 12 in MRC mask 

roi_sizexy = 32   
roi_sizez = 12    


def normalization_sklearn(volume: np.ndarray, index: int):
    scaler =  MinMaxScaler(feature_range=(0, 255)) #StandardScaler() 
    print(volume.shape, volume.shape[-1])
    rvolume = scaler.fit_transform(volume.reshape(-1, volume.shape[-1])).reshape(volume.shape)

    return rvolume 

def crop_coordinate(mrcArrDicom, mi2ArrDicom, mrcarrDicom, mrcmask_data, mi2arrDicom, mi2mask_data, lmrep_coord, repslice):
    y1, x1, y2, x2 = lmrep_coord   
    
    centery = int((y1+y2)/2)
    centerx = int((x1+x2)/2)
   
    cy1 = centery - roi_sizexy       
    cy2 = centery + roi_sizexy - 1   
   
    cx1 = centerx - roi_sizexy
    cx2 = centerx + roi_sizexy - 1   
   
    cz1 = repslice - roi_sizez       
    cz2 = repslice + roi_sizez - 1

    print(cx1, cx2, cy1, cy2, cz1, cz2)    
    
    mrcArrimageCrop = mrcArrDicom[cy1:cy2+1, cx1:cx2+1, cz1:cz2+1]   
    mrcimageCrop = mrcarrDicom[cy1:cy2+1, cx1:cx2+1, cz1:cz2+1]   
    mrcmaskCrop = mrcmask_data[cy1:cy2+1, cx1:cx2+1, cz1:cz2+1]   
    
    mi2ArrimageCrop = mi2ArrDicom[cy1:cy2+1, cx1:cx2+1, cz1:cz2+1]   
    mi2imageCrop = mi2arrDicom[cy1:cy2+1, cx1:cx2+1, cz1:cz2+1]   
    mi2maskCrop = mi2mask_data[cy1:cy2+1, cx1:cx2+1, cz1:cz2+1]   
    
    print(mrcimageCrop.shape, mrcmaskCrop.shape,  mi2imageCrop.shape, mi2maskCrop.shape)
    
    return mrcArrimageCrop, mi2ArrimageCrop, mrcimageCrop, mrcmaskCrop, mi2imageCrop, mi2maskCrop 

def save_dicomCropnpy(mrcimage, mi2image, savepath):
    create_check_dir(savepath)
    
    mrcoutpath_images = os.path.join(savepath, '22 MRC')
    mi2outpath_images = os.path.join(savepath, '26 MI2')
    
    create_check_dir(mrcoutpath_images)
    create_check_dir(mi2outpath_images)
    
    np.save(os.path.join(mrcoutpath_images, 'cropImage'), mrcimage)
    np.save(os.path.join(mi2outpath_images, 'cropImage'), mi2image)

def save_Cropnpy(mrcimage, mrcmask, mi2image, mi2mask, savepath):
    create_check_dir(savepath)
    
    mrcoutpath_images = os.path.join(savepath, '22 MRC')
    mrcoutpath_masks = os.path.join(savepath, '22 MRC')
    mi2outpath_images = os.path.join(savepath, '26 MI2')
    mi2outpath_masks = os.path.join(savepath, '26 MI2')
    
    create_check_dir(mrcoutpath_images)
    create_check_dir(mrcoutpath_masks)
    create_check_dir(mi2outpath_images)
    create_check_dir(mi2outpath_masks)
   
    np.save(os.path.join(mrcoutpath_images, 'cropImage'), mrcimage)
    np.save(os.path.join(mrcoutpath_masks, 'cropMask'), mrcmask)
    np.save(os.path.join(mi2outpath_images, 'cropImage'), mi2image)
    np.save(os.path.join(mi2outpath_masks, 'cropMask'), mi2mask)

def load_dicom_sitk(dicom_path):
    series_ids = sitk.ImageSeriesReader.GetGDCMSeriesIDs(dicom_path)
    series_file_names = sitk.ImageSeriesReader.GetGDCMSeriesFileNames(dicom_path, series_ids[0])
    
    series_reader = sitk.ImageSeriesReader()
    series_reader.SetFileNames(series_file_names)
    dicom_sitk = series_reader.Execute()
    sitk_dicomImage = series_reader.Execute()

    Spacing = list(dicom_sitk.GetSpacing())
    Spacing_npy = np.round(Spacing, 4)
    
    sitk_dicomArray = sitk.GetArrayFromImage(sitk_dicomImage)
    print(sitk_dicomArray.shape) 
    
    sitk_dicomArrayT = np.transpose(sitk_dicomArray, (1,2,0))
    print(sitk_dicomArrayT.shape) 
    
    return sitk_dicomArrayT, Spacing_npy
    
def load_nifti_sitk(nifti_path):
    
    sitk_niftiImage = sitk.ReadImage(nifti_path)
    sitk_niftiArray = sitk.GetArrayFromImage(sitk_niftiImage)
    print(sitk_niftiArray.shape) 
    
    sitk_niftiArrayT = np.transpose(sitk_niftiArray, (1,2,0))
    print(sitk_niftiArrayT.shape) 
    
    return sitk_niftiArrayT
    
def has_meniere(mask):
    
    unique, counts = np.unique(mask, return_counts=True)
    dict_mask = dict(zip(unique, counts))             
    
    mrcmask = (mask==mrcmask_value).sum()   
    mi2mask = (mask==mi2mask_value).sum()    
    
    if mrcmask >= mi2mask:
        return mrcmask
    else:
        return mi2mask

def has_Cropmeniere(mask):
    
    unique, counts = np.unique(mask, return_counts=True)
    dict_mask = dict(zip(unique, counts))                
    
    return (mask==255).sum()    

def get_index_meniere_slice(masks):
    masks_idx = []    
  
    if len(masks.shape) == 3:    
        for i in range(0, masks.shape[2]):        
          if has_meniere(masks[:,:,i]):   
              masks_idx.append(i)
              
    elif len(masks.shape) == 2:   
        if has_meniere(masks[:,:,i]):    
            masks_idx.append(i)
            
    if not masks_idx:
        raise ValueError("There must be at least one slice with meniere")
    else:      

        return masks_idx 
              
def get_index_meniere_Cropslice(masks):
    masks_idx = []    
  
    if len(masks.shape) == 3:      
        for i in range(0, masks.shape[2]):        
          if has_Cropmeniere(masks[:,:,i]):   
              masks_idx.append(i)
              
    elif len(masks.shape) == 2:    
        if has_Cropmeniere(masks[:,:,i]):   
            masks_idx.append(i)
            
    if not masks_idx:
        raise ValueError("There must be at least one slice with meniere")
    else:      
        return masks_idx 
    
def landmark_representative_slice(mask, meniere_side, index_meniere):
    halfline = mask.shape
    
    if meniere_side == 'right':
        for idx in index_meniere:
            if ((mask[:,:int(halfline[1]/2),idx] == mrclandmark_value).sum()) == 4:
                repre_slice = idx
                mrclandmark_coord = np.argwhere(mask[:,:int(halfline[1]/2)+1,idx] == mrclandmark_value)   # return array, 

    if meniere_side == 'left':
        for idx in index_meniere:
            if ((mask[:,int(halfline[1]/2)+1:,idx] == mrclandmark_value).sum()) == 4:
                repre_slice = idx
                mrclandmark_coord = np.argwhere(mask[:,int(halfline[1]/2)+1:,idx] == mrclandmark_value)   # return array, 
                  
    return repre_slice


def landmark_representative_coord(meniere_side, repslice, rl_landmark):
    rl_lmlist = rl_landmark.tolist()
    
    yxlist = []
    for (y, x, z) in rl_lmlist:
        if z == repslice:
            yxlist.append([y, x])
           
    yxarray = np.array(yxlist)
    
    yaxis = yxarray[:,0]
    xaxis = yxarray[:,1]
    
    y1 = np.min(yaxis)
    y2 = np.max(yaxis)
    x1 = np.min(xaxis)
    x2 = np.max(xaxis)
    
    return [y1, x1, y2, x2]
          
def check_meniereSlice(image, mask, folderside):
    
    nbslice = []
    nbslice_meniere = []
    
    if image.shape[0] != mask.shape[0] or image.shape[1] != mask.shape[1]:      
        print("size different between mask and image")
    elif image.shape[2] != mask.shape[2]:     
        print("slice number different between mask and image")
    else:
        index_meniere = get_index_meniere_slice(mask)
        slice_image = image.shape[2] 
        slice_image_with_menieres = len(index_meniere)

        nbslice.append(slice_image - slice_image_with_menieres)
        nbslice_meniere.append(slice_image_with_menieres)      

        print("{} : {}*{} : {} total slices, {} slices with menieres)".format('image', \
                                image.shape[0], image.shape[1], slice_image, slice_image_with_menieres))        
            
    mrclandmark_coord = np.argwhere(mask == mrclandmark_value)   
    
    if len(mrclandmark_coord) != mrclandmark_count * 2:
        raise Exception('landmark ValueError')
    
    mrclandmark_coord = mrclandmark_coord[mrclandmark_coord[:,1].argsort()]
    
    right_landmark = mrclandmark_coord[:6]
    left_landmark = mrclandmark_coord[-6:]
    
    halfline = mask.shape
    rpixelc = (mask[:,:int(halfline[1]/2),:] == 1).sum()   
    lpixelc = (mask[:,int(halfline[1]/2)+1:,:] == 1).sum()

    if rpixelc > lpixelc:
        meniere_side = 'right'   
    else:
        meniere_side = 'left'

    if  meniere_side == 'right' and folderside == 'Rt':
        pass
    elif meniere_side == 'left' and folderside == 'Lt':
        pass
    else:
        raise Exception('Side Error')
        
    rep_slice = landmark_representative_slice(mask, meniere_side, index_meniere)
    
    if meniere_side == 'right':
        lmrep_coord = landmark_representative_coord(meniere_side, rep_slice, right_landmark)
    else:
        lmrep_coord = landmark_representative_coord(meniere_side, rep_slice, left_landmark)
     
    mask = np.where(mask == mrclandmark_value, 0, mask)  
    print('min max after mask_3d delete mrclandmark_value :', np.min(mask), np.max(mask))
    
    return mask, meniere_side, rep_slice, lmrep_coord, index_meniere

        
if __name__ == '__main__':
    ## Set the path of the original data, crop data, and dicom crop data

    cropdata_path = './data/dicom2022Crop' # 'CROP DATA PATH'
    dicomcropdata_path = 'DICOM CROP DATA PATH'
    origindata_path = 'ORIGIN DATA PATH'
    
    create_check_dir(cropdata_path)
    create_check_dir(dicomcropdata_path)
    
    directory_list = sorted(glob(os.path.join(origindata_path, '*')))

    df = pd.DataFrame(columns=['name', 'side', 'rep_slice', 'crop rep_slice', 'Spacing',\
                               'GT MRC Volume', 'Pred MRC Volume', 'GT MI2 Volume',\
                                    'Pred MI2 Volume', 'GT MI2/MRC', 'Pred MI2/MRC'])

    train_mrcmsk_volume = avg_metrics()
    train_mi2msk_volume = avg_metrics()

    space = [0.5, 0.5, 1.0]
    right_count = 0
    left_count = 0
    
    for idx, directory in enumerate(directory_list):
        if 'Rt' in directory:
            folderside = 'Rt'
            right_count += 1
        elif 'Lt' in directory:
            folderside = 'Lt'
            left_count += 1
            
        mrcPathDicom = os.path.join(directory, '22 MRC') 
        mrcfileMask = os.path.join(directory, '22.nii.gz')
   
        mi2PathDicom = os.path.join(directory, '26 MI2') 
        mi2fileMask = os.path.join(directory, '26.nii.gz')

        mrcArrayDicom, mrcSpacing = load_dicom_sitk(mrcPathDicom)
        mrcarrDicom = normalization_sklearn(mrcArrayDicom, 104)   
        
        mi2ArrayDicom, mi2Spacing = load_dicom_sitk(mi2PathDicom)
        mi2arrDicom = normalization_sklearn(mi2ArrayDicom, 104)   
       
        if np.array_equal(mrcSpacing, mi2Spacing):
            pass
        else:
            raise Exception("Space same error")    
      
        mrcarrNifti = load_nifti_sitk(mrcfileMask)
        mi2arrNifti = load_nifti_sitk(mi2fileMask)
    
        #return mask, meniere_side, rep_slice, lmrep_coord
        mrcmask_data, meniere_side, rep_slice, lmrep_coord, index_meniere = \
            check_meniereSlice(mrcarrDicom, mrcarrNifti, folderside) 
        print(mrcmask_data, meniere_side, rep_slice, lmrep_coord, index_meniere)
        
        mrcArrimageCrop, mi2ArrimageCrop, mrcimage_Crop, mrcmask_Crop, mi2image_Crop, mi2mask_Crop = \
            crop_coordinate(mrcArrayDicom, mi2ArrayDicom, mrcarrDicom, mrcmask_data, mi2arrDicom, mi2arrNifti, lmrep_coord, rep_slice)
        
        mrcmask_Crop = np.where(mrcmask_Crop == mrcmask_value, 1, 0) 
        mi2mask_Crop = np.where(mi2mask_Crop == mi2mask_value, 1, 0) 
       
        dirsplit = directory.rsplit('/', maxsplit=1)   
        crop_rep_slice = roi_sizez  # roi_sizez == 12 
                                  
        train_mrcmsk_volume.update(volume_cal(mrcmask_Crop, space))
        train_mi2msk_volume.update(volume_cal(mi2mask_Crop, space))

        df.loc[len(df)] = [dirsplit[1], meniere_side, rep_slice, crop_rep_slice, mrcSpacing,\
                           train_mrcmsk_volume.val, 0, train_mi2msk_volume.val, 0,\
                               train_mi2msk_volume.val/train_mrcmsk_volume.val, 0]
        
        save_path1 = directory.replace(origindata_path, cropdata_path)
        save_Cropnpy(mrcimage_Crop, mrcmask_Crop, mi2image_Crop, mi2mask_Crop, save_path1)
    
        save_path2 = directory.replace(origindata_path, dicomcropdata_path)
        save_dicomCropnpy(mrcArrimageCrop, mi2ArrimageCrop, save_path2)
        
    df.loc[len(df), 'name':'side'] = ['Right', right_count]
    df.loc[len(df), 'name':'side'] = ['Left', left_count]
    
    df.to_csv('meniere_repsliceFrame_csvfile.csv')    
