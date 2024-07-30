import torch

import torch.utils.data as data
import torchvision.transforms as transforms
import random
import os
import os.path as osp

#from resize_natural_3_dataset import ResizeNatural3Dataset
from .IBCLN_data_loader import IBCLN_data_loader_ibcln, IBCLN_data_loader_ibcln_TEST
from .data_util import create_list_paired
from .ERRNET_dataset import CEILTestDataset
from .RR4K_dataset_loader import RR4K_dataset
from .reflect_dataset import CEILTestDataset
def create_dataset(
    args,
    data_path,
    mode='train',
):
    def _list_image_files_recursively(data_dir):
        file_list = []
        for home, dirs, files in os.walk(data_dir):
            for filename in files:
                if filename.endswith('gt.jpg'):
                    file_list.append(os.path.join(home, filename))
        file_list.sort()
        return file_list
    
    if args.DATA_TYPE == 'IBCLN':
        #IBCLN_files = sorted([file for file in os.listdir(data_path + '/source') if file.endswith('.png')])
        
        if mode == 'train':
            dataset = IBCLN_data_loader_ibcln(args=args,root_path=data_path,mode=mode)
        elif mode == 'test':
            transform_flag = args.data_transform
            blend_list, gt_list = create_list_paired(data_path)
            dataset = IBCLN_data_loader_ibcln_TEST(blended_list=blend_list,fake_trans_list=blend_list,trans_list=gt_list,transform=transform_flag,train_flag=True)
        elif mode == 'debug':
            dataset = None
        else:
            raise 'wrong Mode!'
    elif args.DATA_TYPE =='RR4K':
        if mode == 'train':
            data_dir = osp.join(data_path,'train')
            dataset = RR4K_dataset(datadir=data_dir,args=args,size=args.dataset_size,enable_transforms=True)
        elif mode == 'test':
            transform_flag = args.data_transform
            blend_list, gt_list = create_list_paired(data_path)
            #data_dir = osp.join(data_path,'test')
            #print(blend_list[:10],gt_list[:10])
            #dataset = RR4K_dataset_test(blended_list=blend_list,fake_trans_list=blend_list,trans_list=gt_list,test_size_h=args.GT_SIZE,transform=transform_flag,train_flag=True)
            dataset = CEILTestDataset(datadir=data_path,GT_height=512,enable_transforms=transform_flag,if_align=False,transformers=None,round_factor=32)
    
    else:
        print('Unrecognized data_type!')
        raise NotImplementedError
    data_loader = data.DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.WORKER, drop_last=True
    )

    return data_loader
