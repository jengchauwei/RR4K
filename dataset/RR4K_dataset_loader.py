

from PIL import Image
import random
import math
import numpy as np
import cv2
import scipy.stats as st
import os
import os.path as osp
from torch.utils.data.dataset import Dataset
import torchvision.transforms as transforms
import torchvision.transforms.functional as F
try:
    import accimage
except ImportError:
    accimage = None
from scipy.signal import convolve2d

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]
def read_fns(filename):
    with open(filename) as f:
        fns = f.readlines()
        fns = [fn.strip() for fn in fns]
    return fns

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


def paired_data_transforms(img_1, img_2, crop_size=256,resize_range = (256,1080),unaligned_transforms=False):
    def get_params(img, output_size):
        w, h = img.size
        th, tw = output_size
        if w == tw and h == th:
            return 0, 0, h, w

        i = random.randint(0, h - th)
        j = random.randint(0, w - tw)
        return i, j, th, tw
    
    # target_size = int(random.randint(224+10, 448) / 2.) * 2
    target_size = int(random.randint(resize_range[0], resize_range[1]) / 2.) * 2
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

    i, j, h, w = get_params(img_1, (crop_size,crop_size))
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

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def make_dataset(dir, fns=None):
    images = []
    assert os.path.isdir(dir), '%s is not a valid directory' % dir

    if fns is None:
        for root, _, fnames in sorted(os.walk(dir)):
            for fname in fnames:
                if is_image_file(fname):                
                    path = os.path.join(root, fname)
                    images.append(path)
    else:
        for fname in fns:
            if is_image_file(fname):
                path = os.path.join(dir, fname)
                images.append(path)

    return images


class RR4K_dataset(Dataset):
    def __init__(self, datadir,args, fns=None, size=None, enable_transforms=False, unaligned_transforms=False, round_factor=1, flag=None):
        super(RR4K_dataset, self).__init__()
        self.opt = args
        self.size = size
        self.datadir = datadir
        self.fns = fns or os.listdir(osp.join(datadir, 'reflection'))
        
        self.enable_transforms = enable_transforms
        self.unaligned_transforms = unaligned_transforms
        self.round_factor = round_factor
        self.flag = flag
        
        if size is not None:
            self.fns = self.fns[:size]

    def __getitem__(self, index):
        idx = index%len(self.fns)
        fn = self.fns[idx]
        
        t_img = Image.open(osp.join(self.datadir, 'transmission', fn)).convert('RGB')
        m_img = Image.open(osp.join(self.datadir, 'reflection', fn)).convert('RGB')
        
        if self.enable_transforms:
            t_img, m_img = paired_data_transforms(t_img, m_img,crop_size=self.opt.crop_size,resize_range=(self.opt.resize_min,self.opt.resize_max))

        B = transforms.ToTensor()(t_img)
        M = transforms.ToTensor()(m_img)

        dic =  {'Blend': M, 'GT': B, 'name': fn, 'real':True} # fake reflection gt 
        if self.flag is not None:
            dic.update(self.flag)
        return dic

    def __len__(self):
        if self.size is not None:
            return max(len(self.fns), self.size)
        else:
            return len(self.fns)


class RR4K_dataset_test(Dataset):
    def __init__(self,blended_list,fake_trans_list,trans_list,test_size_h=512,transform=False,if_GT=True, train_flag=False):
        self.to_tensor = transforms.ToTensor()            
        self.blended_list = blended_list
        self.test_size_h = test_size_h
        self.trans_list = trans_list
        self.transform = transform
        self.if_GT = if_GT
        self.train_flag = train_flag
        if not train_flag: # already get result, save as image files
            self.fake_img_list = fake_trans_list
        else:
            self.fake_img_list = None
    def __getitem__(self, index):
        blended = Image.open(self.blended_list[index])
        basename, _ = os.path.splitext(osp.basename(self.blended_list[index]))
        trans = blended

        fake_trans = blended
        if self.if_GT:
            trans = Image.open(self.trans_list[index])
            
        if not self.train_flag:
            #fake_trans = cv2.imread(self.fake_img_list[index])
            fake_trans = Image.open(self.fake_img_list[index])

        if self.transform == True:
            """
            if trans.shape[0] > trans.shape[1]:
                neww = 300
                newh = (round((neww / trans.shape[1]) * trans.shape[0])//2)*2 #
            if trans.shape[0] < trans.shape[1]:
                newh = 300
                neww = (round((newh / trans.shape[0]) * trans.shape[1])//2)*2 # times of 2
            """
            trans_np = np.array(trans)
            
            if trans_np.shape[0] > trans_np.shape[1]:
                neww = self.test_size_h
                newh = (round((neww / trans_np.shape[1]) * trans_np.shape[0])//2)*2 #
            if trans_np.shape[0] < trans_np.shape[1]:
                newh = self.test_size_h
                neww = (round((newh / trans_np.shape[0]) * trans_np.shape[1])//2)*2 # times of 2
            
                
            blended = cv2.resize(np.float32(blended), (neww, newh), cv2.INTER_CUBIC)/255.0
            trans = cv2.resize(np.float32(trans), (neww, newh), cv2.INTER_CUBIC)/255.0
            fake_trans = cv2.resize(np.float32(fake_trans), (neww, newh), cv2.INTER_CUBIC)/255.0
        
        blended = self.to_tensor(blended)
        trans = self.to_tensor(trans)
        
    
        fake_trans = self.to_tensor(fake_trans)
        
        
        # b_g_r to r_g_b
            
        return {'Blend':blended,'Ts':fake_trans,'GT':trans,'name':basename}
    def __len__(self):
        return len(self.blended_list)
    