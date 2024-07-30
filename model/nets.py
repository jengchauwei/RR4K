"""
Implementation of ESDNet for image demoireing
"""


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch import Tensor
from typing import List
from .BSRN_arch import BSConvU
import math

from .dcn import DeformableConv2d
from .LKDN_arch import LKDB,BSConvU,default_init_weights,Attention
#from structs import 
class my_model(nn.Module):
    def __init__(self,
                 en_feature_num,
                 en_inter_num,
                 de_feature_num,
                 de_inter_num,
                 dmm_number=1,
                 ):
        super(my_model, self).__init__()
        self.encoder = Encoder(feature_num=en_feature_num, inter_num=en_inter_num, dmm_number=dmm_number)
        self.decoder = Decoder(en_num=en_feature_num, feature_num=de_feature_num, inter_num=de_inter_num,
                               dmm_number=dmm_number)
    

    def forward(self, x):
        y_1, y_2, y_3 = self.encoder(x)
        out_1, out_2, out_3 = self.decoder(y_1, y_2, y_3)

        return out_1, out_2, out_3

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0.0, 0.02)
                if m.bias is not None:
                    m.bias.data.normal_(0.0, 0.02)
            if isinstance(m, nn.ConvTranspose2d):
                m.weight.data.normal_(0.0, 0.02)


class Decoder(nn.Module):
    def __init__(self, en_num, feature_num, inter_num, dmm_number):
        super(Decoder, self).__init__()
        self.preconv_3 = conv_relu(4 * en_num, feature_num, 3, padding=1)
        self.decoder_3 = Decoder_Level(feature_num, inter_num, dmm_number)
        self.preconv_2 = conv_relu(2 * en_num + feature_num, feature_num, 3, padding=1)
        self.decoder_2 = Decoder_Level(feature_num, inter_num, dmm_number)
        self.preconv_1 = conv_relu(en_num + feature_num, feature_num, 3, padding=1)
        self.decoder_1 = Decoder_Level(feature_num, inter_num, dmm_number)

    def forward(self, y_1, y_2, y_3):
        x_3 = y_3
        x_3 = self.preconv_3(x_3)
        out_3, feat_3 = self.decoder_3(x_3)

        x_2 = torch.cat([y_2, feat_3], dim=1)
        x_2 = self.preconv_2(x_2)
        out_2, feat_2 = self.decoder_2(x_2)

        x_1 = torch.cat([y_1, feat_2], dim=1)
        x_1 = self.preconv_1(x_1)
        out_1 = self.decoder_1(x_1, feat=False)

        return out_1, out_2, out_3

class Encoder(nn.Module):
    def __init__(self, feature_num, inter_num, dmm_number):
        super(Encoder, self).__init__()
        
        self.conv_first = nn.Sequential(
            nn.Conv2d(12, feature_num, kernel_size=5, stride=1, padding=2, bias=True),
            nn.ReLU(inplace=True)
        )
        
        """
        self.dcn_conv_first =  nn.Sequential(
            DeformableConv2d(in_channels=12, out_channels= feature_num, kernel_size=5, dilation=1, padding=2),
            nn.ReLU(inplace=True)
            )
        """
        self.encoder_1 = Encoder_Level(feature_num, inter_num, level=1, dmm_number=dmm_number)
        self.encoder_2 = Encoder_Level(2 * feature_num, inter_num, level=2, dmm_number=dmm_number)
        self.encoder_3 = Encoder_Level(4 * feature_num, inter_num, level=3, dmm_number=dmm_number)

    def forward(self, x):
        x = F.pixel_unshuffle(x, 2)
        x = self.conv_first(x)

        out_feature_1, down_feature_1 = self.encoder_1(x)
        out_feature_2, down_feature_2 = self.encoder_2(down_feature_1)
        out_feature_3 = self.encoder_3(down_feature_2)

        return out_feature_1, out_feature_2, out_feature_3


class Encoder_Level(nn.Module):
    def __init__(self, feature_num, inter_num, level, dmm_number):
        super(Encoder_Level, self).__init__()
        self.rdb = RLKDB(in_channel=feature_num, d_list=(1, 2, 1), inter_num=inter_num)
        self.dmm_blocks = nn.ModuleList()
        for _ in range(dmm_number):
            dmm_block = DMM(in_channel=feature_num, d_list=(1, 2, 3, 2, 1), inter_num=inter_num)
            self.dmm_blocks.append(dmm_block)

        if level < 3:
            self.down = nn.Sequential(
                nn.Conv2d(feature_num, 2 * feature_num, kernel_size=3, stride=2, padding=1, bias=True),
                nn.ReLU(inplace=True)
            )
        self.level = level

    def forward(self, x):
        out_feature = self.rdb(x)
        for dmm_block in self.dmm_blocks:
            out_feature = dmm_block(out_feature)
        if self.level < 3:
            down_feature = self.down(out_feature)
            return out_feature, down_feature
        return out_feature


class Decoder_Level(nn.Module):
    def __init__(self, feature_num, inter_num, dmm_number):
        super(Decoder_Level, self).__init__()
        self.rdb = RLKDB(feature_num, (1, 2, 1), inter_num)
        self.dmm_blocks = nn.ModuleList()
        for _ in range(dmm_number):
            dmm_block = DMM(in_channel=feature_num, d_list=(1, 2, 3, 2, 1), inter_num=inter_num)
            self.dmm_blocks.append(dmm_block)
        self.conv = conv(in_channel=feature_num, out_channel=12, kernel_size=3, padding=1)

    def forward(self, x, feat=True):
        x = self.rdb(x)
        for dmm_block in self.dmm_blocks:
            x = dmm_block(x)
        out = self.conv(x)
        out = F.pixel_shuffle(out, 2)

        if feat:
            feature = F.interpolate(x, scale_factor=2, mode='bilinear')
            return out, feature
        else:
            return out


class DB(nn.Module):
    def __init__(self, in_channel, d_list, inter_num):
        super(DB, self).__init__()
        self.d_list = d_list
        self.conv_layers = nn.ModuleList()
        c = in_channel
        for i in range(len(d_list)):
            dense_conv = conv_relu(in_channel=c, out_channel=inter_num, kernel_size=3, dilation_rate=d_list[i],
                                   padding=d_list[i])
            self.conv_layers.append(dense_conv)
            c = c + inter_num
        self.conv_post = conv(in_channel=c, out_channel=in_channel, kernel_size=1)

    def forward(self, x):
        t = x
        for conv_layer in self.conv_layers:
            _t = conv_layer(t)
            t = torch.cat([_t, t], dim=1)
        t = self.conv_post(t)
        return t

class DFEB(nn.Module): 
    # Deformable Feature Enhancement Block
    def __init__(self, in_channel, d_list, inter_num):
        super(DFEB, self).__init__()
        self.d_list = d_list
        self.conv_layers = nn.ModuleList()
        c = in_channel
        for i in range(len(d_list)):
            if i in (0,1,2):
                dense_conv = conv_relu(in_channel=c, out_channel=inter_num, kernel_size=3, dilation_rate=d_list[i], padding=d_list[i])
            if i == 3:
                dense_conv = nn.Sequential(
                DeformableConv2d(in_channels=c, out_channels= inter_num, kernel_size=3,  dilation=d_list[i],padding= d_list[i]))
            if i == 4:
                dense_conv = nn.Sequential(
                DeformableConv2d(in_channels=c, out_channels= inter_num, kernel_size=3,  dilation=d_list[i],padding= d_list[i]),
                nn.ReLU(inplace=True)
                )
            
            self.conv_layers.append(dense_conv)
            c = c + inter_num
            
        self.conv_post = conv(in_channel=c, out_channel=in_channel, kernel_size=1)

    def forward(self, x):
        t = x
        for conv_layer in self.conv_layers:
            _t = conv_layer(t)
            t = torch.cat([_t, t], dim=1)
        t = self.conv_post(t)
        return t
class DMM(nn.Module):
    def __init__(self, in_channel, d_list, inter_num):
        super(DMM, self).__init__()
        self.basic_block = DFEB(in_channel=in_channel, d_list=d_list, inter_num=inter_num)
        self.basic_block_2 = DFEB(in_channel=in_channel, d_list=d_list, inter_num=inter_num) 
        self.basic_block_4 = DFEB(in_channel=in_channel, d_list=d_list, inter_num=inter_num)
        
        self.fusion = ECA_CSAF(3 * in_channel)
    def forward(self, x):
        x_0 = x
        x_2 = F.interpolate(x, scale_factor=0.5, mode='bilinear')
        x_4 = F.interpolate(x, scale_factor=0.25, mode='bilinear')

        y_0 = self.basic_block(x_0)
        y_2 = self.basic_block_2(x_2)
        y_4 = self.basic_block_4(x_4)

        y_2 = F.interpolate(y_2, scale_factor=2, mode='bilinear')
        y_4 = F.interpolate(y_4, scale_factor=4, mode='bilinear')

        y = self.fusion(y_0, y_2, y_4)
        y = x + y

        return y


class ECA_CSAF(nn.Module):
    """Constructs a ECA module.

    Args:
        channel: Number of channels of the input feature map
        k_size: Adaptive selection of kernel size
    """
    def __init__(self, in_chnls, ratio=4, k_size = 3 ):
        t = int(abs((math.log(in_chnls,2)+1)/2))
        k_size = t if t%2 else t+1
        super(ECA_CSAF, self).__init__()
        self.squeeze = nn.AdaptiveAvgPool2d((1, 1))
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x0, x2, x4):
        out0 = self.squeeze(x0) #  step: GAP
        out2 = self.squeeze(x2) # step: GAP
        out4 = self.squeeze(x4) # step: GAP
        out = torch.cat([out0, out2, out4], dim=1)
        
        #out = self.squeeze(out)
        out = self.conv(out.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        out = self.sigmoid(out)
        w0, w2, w4 = torch.chunk(out, 3, dim=1)
        
        x = x0 * w0 + x2 * w2 + x4 * w4
        
        return x 
        

class RLKDB(nn.Module):
    def __init__(self, in_channel,d_list,inter_num,num_block =4):
        super(RLKDB,self).__init__()
        self.d_list = d_list
        self.conv_layers = nn.ModuleList()
          
        self.conv = BSConvU
        self.B1 = LKDB(in_channels=in_channel, out_channels=in_channel, atten_channels=inter_num, conv=self.conv)
        self.B2 = LKDB(in_channels=in_channel, out_channels=in_channel, atten_channels=inter_num, conv=self.conv)
        self.B3 = LKDB(in_channels=in_channel, out_channels=in_channel, atten_channels=inter_num, conv=self.conv)
        #self.B4 = LKDB(in_channels=in_channel, out_channels=in_channel, atten_channels=inter_num, conv=self.conv)
        #self.B5 = LKDB(in_channels=in_channel, out_channels=in_channel, atten_channels=inter_num, conv=self.conv)
        self.gelu = nn.GELU()
        
        self.c1 = nn.Conv2d(in_channel * num_block, in_channel, 1 )
        #self.conv_post = nn.Conv2d(in_channels=c, out_channels=in_channel,kernel_size=1)
        
    def forward(self, x):
        t = x
        out_B1 = self.B1(t)
        out_B2 = self.B2(out_B1)
        out_B3 = self.B3(out_B2)
        #out_B4 = self.B4(out_B3)
        #out_B5 = self.B5(out_B4)
        trunk = torch.cat([out_B3,out_B2,out_B1, t], dim=1)
        out_B = self.c1(trunk)
        out_B = self.gelu(out_B)
        
        return out_B + x

class conv(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, dilation_rate=1, padding=0, stride=1):
        super(conv, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=kernel_size, stride=stride,
                              padding=padding, bias=True, dilation=dilation_rate)

    def forward(self, x_input):
        out = self.conv(x_input)
        return out


class conv_relu(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, dilation_rate=1, padding=0, stride=1):
        super(conv_relu, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=kernel_size, stride=stride,
                      padding=padding, bias=True, dilation=dilation_rate),
            nn.ReLU(inplace=True)
        )

    def forward(self, x_input):
        out = self.conv(x_input)
        return out

class dcn_conv_relu(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, dilation_rate=1, padding=0, stride=1):
        super(dcn_conv_relu, self).__init__()
        """
        self.dcn_conv_first =  nn.Sequential(
            DeformableConv2d(in_channels=12, out_channels= feature_num, kernel_size=5, dilation=1, padding=2),
            nn.ReLU(inplace=True)
            )
        """
        self.conv =  nn.Sequential(
            DeformableConv2d(in_channels=in_channel, out_channels= out_channel, kernel_size= kernel_size, stride=stride,
                             padding= padding, bias= True, dilation= dilation_rate),
            nn.ReLU(inplace=True)
        )
    def forward(self, x_input):
        out = self.conv(x_input)
        return out
     