from .common import SSIM, PSNR, tensor2img
from skimage.metrics import peak_signal_noise_ratio as ski_psnr
from skimage.metrics import structural_similarity as ski_ssim
from utils.matlab_ssim import MATLAB_SSIM
import lpips
import torch
import numpy as np
from math import log10

from .index import tensor2im, quality_assess
class create_metrics():
    def __init__(self, args, device):
        self.data_type = args.DATA_TYPE
        self.lpips_fn = lpips.LPIPS(net='alex').cuda()
        
    def compute(self, out_img, gt):
        if self.data_type == 'IBCLN':
            res_psnr, res_ssim = self.ibcln_psnr_ssim(out_img, gt)
        elif self.data_type == 'RR4K':
            res_psnr, res_ssim = self.ibcln_psnr_ssim(out_img, gt)
        elif self.data_type == 'ERRNET':
            res_psnr, res_ssim = self.ibcln_psnr_ssim(out_img, gt)
        else: 
            print('Unrecognized data_type for evaluation!')
            raise NotImplementedError
        pre = torch.clamp(out_img, min=0, max=1)
        tar = torch.clamp(gt, min=0, max=1)

        # calculate LPIPS
        res_lpips = self.lpips_fn.forward(pre, tar, normalize=True).item()
        return res_lpips, res_psnr, res_ssim

    def ibcln_psnr_ssim(self,out_img, gt):
        out_img = tensor2im(out_img)
        gt = tensor2im(gt)
        res, psnr, ssim = quality_assess(out_img, gt)

        return psnr, ssim