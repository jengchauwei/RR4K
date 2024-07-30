
import torchvision.transforms as transforms
from torch.utils.data.dataset import Dataset
import random
from PIL import Image
import os
from .image_folder import make_dataset
from .base_dataset import BaseDataset, get_transform
import numpy as np
import torch
import cv2
import os.path as osp
import torch
import numpy as np
class RandomCrop(object):
    """Crop randomly the image in a sample.

    Args:
        output_size (tuple or int): Desired output size. If int, square crop is made.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, sample):
        image, landmarks = sample['Blend'], sample['Ts']
        h, w = image.shape[:2]
        min_a = min(h, w)
        self.output_size = (min_a * 7 // 10, min_a * 7 // 10)
        new_h, new_w = self.output_size

        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)
        image = image[top: top + new_h,
                      left: left + new_w]

        landmarks = landmarks[top: top + new_h,
                      left: left + new_w]
        return {'Blend': image, 'Ts': landmarks}

    
def default_loader(path_set=[]):
    imgs = []
    for path in path_set:
         img = Image.open(path).convert('RGB')
         img = default_toTensor(img)
         imgs.append(img)
    
    return imgs
    
def crop_loader(crop_size, x, y, path_set=[]):
    imgs = []
    for path in path_set:
         img = Image.open(path).convert('RGB')
         img = img.crop((x, y, x + crop_size, y + crop_size))
         img = default_toTensor(img)
         imgs.append(img)
    
    return imgs
    
def resize_loader(resize_size, path_set=[]):
    imgs = []
    for path in path_set:
         img = Image.open(path).convert('RGB')
         img = img.resize((resize_size,resize_size),Image.BICUBIC)
         img = default_toTensor(img)
         imgs.append(img)
    
    return imgs

def default_toTensor(img):
    t_list = [transforms.ToTensor()]
    composed_transform = transforms.Compose(t_list)
    return composed_transform(img)



def augment(*args, hflip=True, rot=True):
    hflip = hflip and random.random() < 0.5
    vflip = rot and random.random() < 0.5
    rot90 = rot and random.random() < 0.5

    def _augment(img):
        if hflip: img = img[:, ::-1, :]
        if vflip: img = img[::-1, :, :]
        if rot90: img = img.transpose(1, 0, 2)
        
        return img

    return [_augment(a) for a in args]


class R4KR_data_loader(BaseDataset):
    """
       this synthesis function was copy from ERRNet-> transform.py 
    """
    def __init__(self, args, root_path, mode='train' ):
        self.mode = mode
        self.args = args
        self.natural_dir_ts = os.path.join(root_path, 'natural_' + 'T')
        self.natural_dir_blend = os.path.join(root_path, 'natural_' + 'I')
        self.natural_ts_paths = sorted(make_dataset(self.natural_dir_ts, args.max_dataset_size))  # load images from '/path/to/data/trainA1'
        self.natural_blend_paths = sorted(make_dataset(self.natural_dir_blend, args.max_dataset_size))  # load images from '/path/to/data/trainB'
        self.natural_size = len(self.natural_ts_paths)  # get the size of dataset A1

        self.crop = RandomCrop(args.load_size)
        self.dir_ts = os.path.join(root_path, mode,'transmission') #transimision layer
        self.dir_ref = os.path.join(root_path, mode,'reflection') #reflection layer
        self.dir_blend = os.path.join(root_path, mode,'blend') # mixed ,output layer
        
        self.ts_paths = sorted(make_dataset(self.dir_ts, args.max_dataset_size))  # load images from '/path/to/data/trainA1'
        self.ref_paths = sorted(make_dataset(self.dir_ref, args.max_dataset_size)) # load images from '/path/to/data/trainA2'

        if not self.mode == 'train':
            self.blend_paths = sorted(make_dataset(self.dir_blend, args.max_dataset_size))  # load images from '/path/to/data/trainB'
            self.blend_size = len(self.blend_paths)  # get the size of dataset B

        self.ts_size = len(self.ts_paths)  # get the size of dataset transimision
        self.ref_size = len(self.ref_paths)  # get the  size of dataset reflection

        input_nc =self.args.input_nc  # get the number of channels of input image
        output_nc = self.args.output_nc  # get the number of channels of output image
        self.transform_A = get_transform(self.args, grayscale=(input_nc == 1))
        self.transform_B = get_transform(self.args, grayscale=(output_nc == 1))
        print(self.transform_A)

        
    def __getitem__(self, index):
        is_natural = True
        if self.mode == 'train':
            if is_natural:
                natural_index = index % self.natural_size
                ts_path = self.natural_ts_paths[natural_index]  # make sure index is within then range
                blend_path = self.natural_blend_paths[natural_index]

                ts_img = np.asarray(Image.open(ts_path).convert('RGB')) #transimision layer
                ref_img = Image.fromarray(np.zeros_like(ts_img))        #reflection layer
                blend_img = np.asarray(Image.open(blend_path).convert('RGB'))   #blended image
                imgs = self.crop({'Blend': blend_img, 'Ts': ts_img})
                ts_img, blend_img = Image.fromarray(imgs['Ts']), Image.fromarray(imgs['Blend'])
                is_natural_int = 1 
                #print(np.array(ts_img).shape)  
            else:
                ts_path = self.ts_paths[index % self.ts_size]  # make sure index is within then range
                index_ref = random.randint(0, self.ref_size - 1)
                ref_path = self.ref_paths[index_ref]
                blend_path = ''

                ts_img = Image.open(ts_path).convert('RGB')
                ref_img = Image.open(ref_path).convert('RGB')
                blend_img = Image.fromarray(np.zeros_like(ts_img))
                is_natural_int = 0
        else:  # test
            blend_path = self.blend_paths[index]
            blend_img = Image.open(blend_path).convert('RGB')
            if index < len(self.ts_paths):
                ts_path = self.ts_paths[index]
                ts_img = Image.open(ts_path).convert('RGB')
            else:
                ts_img = Image.fromarray(np.zeros_like(blend_img))
            ref_img = None
            is_natural_int = 1
    

        w, h = ts_img.size
        neww = w // 4 * 4
        newh = h // 4 * 4
        resize = transforms.Resize([newh, neww])
        ts_img = resize(ts_img)
        ref_img = resize(ref_img) if ref_img else None
        blend_img = resize(blend_img)

        Ts = self.transform_A(ts_img)
        Ref = self.transform_A(ref_img) if ref_img else None
        Blend = self.transform_B(blend_img)

        
        # compositi synthesis reflection image
        if not is_natural_int:
            if torch.mean(Ts) * 1 / 2 > torch.mean(Ref):
                return {'GT': torch.zeros_like(Ts), 'Blend':torch.zeros_like(Ts),'Ref':torch.zeros_like(Ts), 'train_flag': False}

                #trans_layer, ref_blur_layer, blend_layer,alpha = self.syn_model(Ref, Ts, self.k_sz)
                #return {'GT': Ref, 'Blend':blend_layer,'Ref':ref_blur_layer,'train_flag':True}
                #return {'GT': torch.zeros_like(Ts), 'Blend':torch.zeros_like(Ts),'Ref':torch.zeros_like(Ts),'train_flag': True}
            else:
                trans_layer, ref_blur_layer, blend_layer,alpha = self.syn_model(Ts, Ref, self.k_sz)
                if Ts.max() < 0.15 or ref_blur_layer.max() < 0.15 or blend_layer.max() < 0.1:
                    return {'GT': torch.zeros_like(Ts), 'Blend':torch.zeros_like(Ts),'Ref':torch.zeros_like(Ts), 'train_flag': False}
            #print('synthesis',np.array(trans_layer).shape,np.array(blend_layer).shape,np.array(ref_blur_layer).shape)
                return {'GT': Ts, 'Blend':blend_layer,'Ref':ref_blur_layer,'train_flag':True}
            
        else:
            #print('natural',np.array(Ts).shape,np.array(Blend).shape,np.array(Ref).shape)
            return {'GT': Ts, 'Blend':Blend, 'Ref':Ref, 'train_flag':True}
        
        
                
        #if A2 is not None:
        #    return {'T': A1, 'T2': T2, 'T4': T4, 'R': A2, 'I': B, 'B_paths': B_path, 'isNatural': is_natural_int}
        #else:
        #    return {'T': A1, 'T2': T2, 'T4': T4, 'I': B, 'B_paths': B_path, 'isNatural': is_natural_int}

    def __len__(self):
        #return super().__len__()
        """Return the total number of images."""
        if self.args.dataset_size == 0 or self.mode == 'test':
            length = max(self.ts_size, self.ref_size, self.blend_size)
        else:
            length = self.args.dataset_size
        return length
    
