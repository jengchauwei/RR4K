import os
from torch.utils.data.dataset import Dataset
import random
import torch
import math
from PIL import Image
import torchvision.transforms as transforms
import torchvision.transforms.functional as F

to_tensor = transforms.ToTensor()


def __scale_width(img, target_width):
    ow, oh = img.size
    if (ow == target_width):
        return img
    w = target_width
    h = int(target_width * oh / ow)
    h = math.ceil(h / 2.) * 2  # round up to even
    return img.resize((w, h), Image.BICUBIC)

def __scale_height(img, target_height):
    ow, oh = img.size
    if (oh == target_height):
        return img
    h = target_height
    w = int(target_height * ow / oh)
    w = math.ceil(w / 2.) * 2  # round up to even
    return img.resize((w, h), Image.BICUBIC)

def paired_data_transforms(img_1, img_2, unaligned_transforms=False):
    def get_params(img, output_size):
        w, h = img.size
        th, tw = output_size
        if w == tw and h == th:
            return 0, 0, h, w

        i = random.randint(0, h - th)
        j = random.randint(0, w - tw)
        return i, j, th, tw
    
    # target_size = int(random.randint(224+10, 448) / 2.) * 2
    target_size = int(random.randint(224, 448) / 2.) * 2
    # target_size = int(random.randint(256, 480) / 2.) * 2
    ow, oh = img_1.size
    if ow >= oh:
        img_1 = __scale_height(img_1, target_size)
        img_2 = __scale_height(img_2, target_size)
    else:
        img_1 = __scale_width(img_1, target_size)
        img_2 = __scale_width(img_2, target_size)

    if random.random() < 0.5:
        img_1 = F.hflip(img_1)
        img_2 = F.hflip(img_2)

    i, j, h, w = get_params(img_1, (224,224))
    # i, j, h, w = get_params(img_1, (256,256))
    img_1 = F.crop(img_1, i, j, h, w)
    
    if unaligned_transforms:
        # print('random shift')
        i_shift = random.randint(-10, 10)
        j_shift = random.randint(-10, 10)
        i += i_shift
        j += j_shift

    img_2 = F.crop(img_2, i, j, h, w)
    
    return img_1,img_2   

class CEILTestDataset(Dataset):
    def __init__(self, datadir, fns=None, size=None, GT_height=512,enable_transforms=False, transformers=None, round_factor=32, flag=None,if_align =False):
        super(CEILTestDataset, self).__init__()
        self.size = size
        self.datadir = datadir
        self.fns = fns or os.listdir(os.path.join(datadir, 'blend'))
        self.enable_transforms = enable_transforms
        self.transforms = transformers
        self.round_factor = round_factor
        self.flag = flag
        self.if_align = if_align
        self.GT_height = GT_height
        
        if size is not None:
            self.fns = self.fns[:size]
    def align(self, x1, x2):
        h, w = x1.height, x1.width
        h, w = h // self.round_factor * self.round_factor, w // self.round_factor * self.round_factor
        x1 = x1.resize((w, h),Image.BICUBIC)
        x2 = x2.resize((w, h),Image.BICUBIC)
        return x1, x2
    def __getitem__(self, index):
        fn = self.fns[index]
        
        t_img = Image.open(os.path.join(self.datadir, 'transmission', fn)).convert('RGB')
        m_img = Image.open(os.path.join(self.datadir, 'blend', fn)).convert('RGB')
        if self.enable_transforms:
            if t_img.size[0] > t_img.size[1]:
                newh = self.GT_height
                neww = (round((newh / t_img.size[1]) * t_img.size[0])//self.round_factor)*self.round_factor
            if t_img.size[0] < t_img.size[1]:
                neww = self.GT_height
                newh = (round((neww / t_img.size[0]) * t_img.size[1])//self.round_factor)*self.round_factor  # times of 2
            t_img = t_img.resize((neww,newh),Image.BICUBIC)
            m_img = m_img.resize((neww,newh),Image.BICUBIC)

        if self.if_align:
            t_img, m_img = self.align(t_img, m_img)
        
        #print(t_img.size)

        if self.transforms is not None:
            M = self.transforms(m_img)
        else:
            M = to_tensor(m_img)
            
        B = to_tensor(t_img)
        dic =  {'Blend': M, 'GT': B, 'name': fn, 'real':True, 'target_r': B} # fake reflection gt 
        if self.flag is not None:
            dic.update(self.flag)
        return dic
    
    def __len__(self):
        if self.size is not None:
            return min(len(self.fns), self.size)
        else:
            return len(self.fns)
