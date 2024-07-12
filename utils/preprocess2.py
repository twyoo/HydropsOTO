import os
from glob import glob
import natsort
import yaml

from utils import create_check_dir


if __name__ == '__main__':

    src_path = './data/dicom2022Crop'
    tgt_path = './data/target_dir'
    
    create_check_dir(src_path)
    create_check_dir(tgt_path)

    crop_directory_list = natsort.natsorted(glob(os.path.join(src_path, '*')))
    assert len(crop_directory_list) == 130, 'dataset length problem'

    name_list = natsort.natsorted(crop_directory_list, reverse=False)
  
    if not os.path.exists(tgt_path+'list'):
        os.makedirs('%slist'%(tgt_path))
        print('%slist'%(tgt_path))
    print(name_list)
    
    with open("%slist/dataset.yaml"%tgt_path, "w",encoding="utf-8") as f:
        yaml.dump(name_list, f)
        
        