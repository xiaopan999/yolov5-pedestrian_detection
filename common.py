# YOLOv5 common modules

import math
from copy import copy
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import torch
import torch.nn as nn
from PIL import Image
from torch.cuda import amp

from utils.datasets import letterbox
from utils.general import non_max_suppression, make_divisible, scale_coords, increment_path, xyxy2xywh, save_one_box, box_iou
from utils.plots import colors, plot_one_box
from utils.torch_utils import time_synchronized

# class PositionEmbeddingLearned(nn.Module):
#     """
#     Absolute pos embedding, learned.
#     """
#
#     def __init__(self, num_embeddings, num_pos_feats):
#         super().__init__()
#         self.row_embed = nn.Embedding(num_embeddings=num_embeddings, embedding_dim=num_pos_feats)
#         self.col_embed = nn.Embedding(num_embeddings=num_embeddings, embedding_dim=num_pos_feats)
#         self.reset_parameters()
#
#     def reset_parameters(self):
#         nn.init.uniform_(self.row_embed.weight)
#         nn.init.uniform_(self.col_embed.weight)
#
#     def forward(self, x):
#         h, w = x.shape[-2:]
#         i = torch.arange(w, device=x.device)
#         j = torch.arange(h, device=x.device)
#
#         x_emb = self.col_embed(i)
#         y_emb = self.row_embed(j)
#
#         pos = torch.cat([
#             x_emb.unsqueeze(0).repeat(h, 1, 1),
#             y_emb.unsqueeze(1).repeat(1, w, 1),
#         ], dim=-1).permute(2, 0, 1).unsqueeze(0).repeat(x.shape[0], 1, 1, 1)
#         return pos
# PE = PositionEmbeddingLearned(256, 128)
# x = torch.randn((2, 256, 80, 80))
# out = PE(x)

def autopad(k, p=None):  # kernel, padding
    # Pad to 'same'
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p


def DWConv(c1, c2, k=1, s=1, act=True):
    # Depthwise convolution
    return Conv(c1, c2, k, s, g=math.gcd(c1, c2), act=act)


class Conv(nn.Module):
    # Standard convolution
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True, r=1):  # ch_in, ch_out, kernel, stride, padding, groups
        super(Conv, self).__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False, dilation=r)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def fuseforward(self, x):
        return self.act(self.conv(x))


class TransformerLayer(nn.Module):
    # Transformer layer https://arxiv.org/abs/2010.11929 (LayerNorm layers removed for better performance)
    def __init__(self, c, num_heads):
        super().__init__()
        self.q = nn.Linear(c, c, bias=False)
        self.k = nn.Linear(c, c, bias=False)
        self.v = nn.Linear(c, c, bias=False)
        self.ma = nn.MultiheadAttention(embed_dim=c, num_heads=num_heads)
        self.fc1 = nn.Linear(c, c, bias=False)
        self.fc2 = nn.Linear(c, c, bias=False)

    def forward(self, x):
        x = self.ma(self.q(x), self.k(x), self.v(x))[0] + x
        x = self.fc2(self.fc1(x)) + x
        return x


class TransformerBlock(nn.Module):
    # Vision Transformer https://arxiv.org/abs/2010.11929
    def __init__(self, c1, c2, num_heads, num_layers):
        super().__init__()
        self.conv = None
        if c1 != c2:
            self.conv = Conv(c1, c2)
        self.linear = nn.Linear(c2, c2)  # learnable position embedding
        self.tr = nn.Sequential(*[TransformerLayer(c2, num_heads) for _ in range(num_layers)])
        self.c2 = c2

    def forward(self, x):
        if self.conv is not None:
            x = self.conv(x)
            x = self.conv(x)
        b, _, w, h = x.shape
        p = x.flatten(2)
        p = p.unsqueeze(0)
        p = p.transpose(0, 3)
        p = p.squeeze(3)
        e = self.linear(p)
        x = p + e

        x = self.tr(x)
        x = x.unsqueeze(3)
        x = x.transpose(0, 3)
        x = x.reshape(b, self.c2, w, h)
        return x

class RFPBottleneck(nn.Module):
    # Standard bottleneck
    def __init__(self, c1, c2, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, shortcut, groups, expansion
        super(RFPBottleneck, self).__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_, c2, 3, 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        x1 = x[0]
        x2 = x[1]
        return x1 + self.cv2(self.cv1(x1)) + x2 if self.add else self.cv2(self.cv1(x1))

class Bottleneck(nn.Module):
    # Standard bottleneck
    def __init__(self, c1, c2, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, shortcut, groups, expansion
        super(Bottleneck, self).__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_, c2, 3, 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))

#res2block -> bottle2block --> as for backbone(resnet part)
class Bottle2neck(nn.Module):
    # Standard bottleneck
    def __init__(self, c1, c2, shortcut=True, g=1, e=0.5, shuffle=False):  # ch_in, ch_out, shortcut, groups, expansion
        super(Bottle2neck, self).__init__()
        self.add = shortcut and c1 == c2
        self.channel_shuffle = shuffle#use channel shuffle?
        c_ = int(c2 * e)  # hidden channels

        if self.add:
            self.cv1 = Conv(c1, c_, 1, 1)
            self.cv_de = Conv(c_, c2 // 4, 1, 1)
            self.arf_one = Conv(c2 // 4, c2 // 4, 3, 1)
            self.arf_two = Conv(c2 // 4, c2 // 4, 3, 1)
            self.arf_three = Conv(c2 // 4, c2 // 4, 3, 1)
            self.arf_four = Conv(c2 // 4, c2 // 4, 3, 1)
            # convs = []
            # self.width = int(math.floor(c1 * (26.0 / 64.0)))
            # self.cv1 = Conv(c1, self.width * 4, 1, 1)
            # for i in range(3):
            #     convs.append(Conv(self.width, self.width, 3, 1, p=1))
            # self.convs = nn.ModuleList(convs)
            # self.cv2 = Conv(self.width * 4, c2, 1, 1, g=g)
        else:
            self.cv1 = Conv(c1, c_, 1, 1)
            self.cv2 = Conv(c_, c2, 3, 1, g=g)

    def forward(self, x):

        if self.add:
            residual = x
            cv1_out = self.cv1(x)
            re_x = self.cv_de(cv1_out)
            out1 = self.arf_one(re_x)  # out1
            out2 = self.arf_two(out1)  # out2
            out3 = self.arf_three(out2)  # out3
            out4 = self.arf_four(out3)  # out4
            out = torch.cat([out1, out2, out3, out4], dim=1)
            out += residual

            # residual = x
            # out = self.cv1(residual)
            # spx = torch.split(out, self.width, 1)  # 4 part
            # for i in range(3):
            #     if i == 0:
            #         sp = spx[i]
            #     else:
            #         sp = sp + spx[i]
            #     sp = self.convs[i](sp)
            #     if i == 0:
            #         out = sp
            #     else:
            #         out = torch.cat((out, sp), 1)
            # out = torch.cat((out, spx[3]), 1)  # 4 part concate
            # #use channel shuffle?
            # if self.channel_shuffle:
            # #     '''Channel shuffle: [N,C,H,W] -> [N,g,C/g,H,W] -> [N,C/g,g,H,w] -> [N,C,H,W]'''
            #     N, C, H, W = out.size()
            #     g = 4
            #     out = out.view(N, g, int(C // g), H, W).permute(0, 2, 1, 3, 4).contiguous().view(N, C, H, W)
            #
            # out = self.cv2(out)
            # out += residual
        else:
            out = self.cv2(self.cv1(x))

        return out


#-------------------------------RFPN_part__v6
class RFPNV6(nn.Module):
    def __init__(self, c1, c2):
        super(RFPNV6, self).__init__()
        #--------------------load pretrain_weights
        self.conv0 = Conv(c1, 64, k=6, s=2, p=2)#(b, 3, 640, 640)->(b, 64, 320, 320)
        self.conv1_b = Conv(64, 128, 3, 2)#(b, 64, 320, 320)->(b, 128, 160, 160)
        self.c3_1_b = C3(128, 128, n=3)#(b, 128, 160, 160)->(b, 128, 160, 160),short_cut=True,n=3
        self.conv2_b = Conv(128, 256, 3, 2)#(b, 128, 160, 160)->(b, 256, 80, 80)
        self.c3_2_b = C3(256, 256, n=6)#(b, 256, 80, 80)->(b, 256, 80, 80),short_cut=True,n=9
        self.conv3_b = Conv(256, 512, 3, 2)#(b, 256, 80, 80)->(b, 512, 40, 40)
        self.c3_3_b = C3(512, 512, n=9)  # (b, 512, 40, 80)->(b, 512, 40, 40),short_cut=True,n=9
        self.conv4_b = Conv(512, 1024, 3, 2)  # (b, 512, 40, 40)->(b, 1024, 20, 20)
        self.c3_4_b = C3(1024, 1024, n=3)  # (b, 1024, 20, 20)->(b, 1024, 20, 20),short_cut=True,n=3
        self.SPPF = SPPF(1024, 1024)
        #----------------------------------


        #-------------------RFPN_stage_1
        self.c3_1_n = C3(1024, 1024, n=3, shortcut=False)  # (b, 1024, 20, 20)->(b, 1024, 20, 20),short_cut=False,n=3

        self.upsample1 = nn.Upsample(size=None, scale_factor=2, mode='nearest')#(b, 1024, 20, 20)->(b, 1024, 40, 40)
        # self.g_l1 = MS_CAM(512, 4)
        self.concat1 = Concat()#(b, 1024, 40, 40)-(b, 512, 40, 40)->(b, 1536, 40, 40)
        self.conv1_n = Conv(1536, 512, 1, 1)#(b, 1536, 40, 40)->(b, 512, 40, 40)
        self.c3_2_n = C3(512, 512, n=3, shortcut=False)# (b, 512, 40, 40)->(b, 512, 40, 40),short_cut=False,n=3

        self.upsample2 = nn.Upsample(size=None, scale_factor=2, mode='nearest')  # (b, 512, 40, 40)->(b, 512, 80, 80)
        # self.g_l2 = MS_CAM(256, 4)
        self.concat2 = Concat()  # (b, 512, 80, 80)-(b, 256, 80, 80)->(b, 768, 80, 80)
        self.conv2_n = Conv(768, 256, 1, 1)  # (b, 768, 80, 80)->(b, 256, 80, 80)
        self.c3_3_n = C3(256, 256, n=3, shortcut=False)  # (b, 256, 80, 80)->(b, 256, 80, 80),short_cut=False,n=3

        #--------------------------backbone_stage_2 load pretrain weights
        # self.c3_4_b = C3(256, 256, n=9)  # (b, 256, 80, 80)->(b, 256, 80, 80),short_cut=True,n=9
        # self.conv5_b = Conv(256, 512, 3, 2)#(b, 256, 80, 80)->(b, 512, 40, 40)
        # self.c3_5_b = C3(512, 512, n=9)  # (b, 512, 40, 80)->(b, 512, 40, 40),short_cut=True,n=9
        # self.conv6_b = Conv(512, 1024, 3, 2)  # (b, 512, 40, 40)->(b, 1024, 20, 20)
        # self.SPP2 = SPP(1024, 1024)
        # self.c3_6_b = C3(1024, 1024, n=6)  # (b, 1024, 20, 20)->(b, 1024, 20, 20),short_cut=True,n=3

        #--------------------------RFPN_stage_2
        # self.c3_4_n = C3(1024, 1024, n=3, shortcut=False)  # (b, 1024, 20, 20)->(b, 1024, 20, 20),short_cut=False,n=3
        #
        # self.upsample3 = nn.Upsample(size=None, scale_factor=2, mode='nearest')  # (b, 1024, 20, 20)->(b, 1024, 40, 40)
        # # self.g_l3 = MS_CAM(512, 4)
        # self.concat3 = Concat()  # (b, 1024, 40, 40)-(b, 512, 40, 40)->(b, 1536, 40, 40)
        # self.conv3_n = Conv(1536, 512, 1, 1)  # (b, 1536, 40, 40)->(b, 512, 40, 40)
        # self.c3_5_n = C3(512, 512, n=3, shortcut=False)  # (b, 512, 40, 40)->(b, 512, 40, 40),short_cut=False,n=3
        #
        # self.upsample4 = nn.Upsample(size=None, scale_factor=2, mode='nearest')  # (b, 512, 40, 40)->(b, 512, 80, 80)
        # # self.g_l4 = MS_CAM(256, 4)
        # self.concat4 = Concat()  # (b, 512, 80, 80)-(b, 256, 80, 80)->(b, 768, 80, 80)
        # self.conv4_n = Conv(768, 256, 1, 1)  # (b, 768, 80, 80)->(b, 256, 80, 80)
        # self.c3_6_n = C3(256, 256, n=3, shortcut=False)  # (b, 256, 80, 80)->(b, 256, 80, 80),short_cut=False,n=3



    def forward(self, x):
        #backbone_1
        conv0 = self.conv0(x)#(1, 64, 320, 320)
        conv1_b = self.conv1_b(conv0)#(1, 128, 160, 160)
        c3_1_b = self.c3_1_b(conv1_b)#(1, 128, 160, 160)

        conv2_b = self.conv2_b(c3_1_b)#(1, 256, 80, 80)
        c3_2_b = self.c3_2_b(conv2_b)#(1, 256, 80, 80)

        conv3_b = self.conv3_b(c3_2_b)#(1, 512, 40, 40)
        c3_3_b = self.c3_3_b(conv3_b)#(1, 512, 40, 40)

        conv4_b = self.conv4_b(c3_3_b)#(1, 1024, 20, 20)
        c3_4_b = self.c3_4_b(conv4_b)#(1, 1024, 20, 20)
        sppf = self.SPPF(c3_4_b)  # (1, 1024, 20, 20)


        #neck_stage_1
        c3_1_n = self.c3_1_n(sppf)#(1, 1024, 20, 20)

        upsample1 = self.upsample1(c3_1_n)
        # c3_3_b_a = self.g_l1(c3_3_b)
        concat1 = self.concat1([upsample1, c3_3_b])
        conv1_n = self.conv1_n(concat1)
        c3_2_n = self.c3_2_n(conv1_n)#(1, 512, 40, 40)

        upsample2 = self.upsample2(c3_2_n)
        # c3_2_b_a = self.g_l2(c3_2_b)
        concat2 = self.concat2([upsample2, c3_2_b])
        conv2_n = self.conv2_n(concat2)
        c3_3_n = self.c3_3_n(conv2_n)#(1, 256, 80, 80)


        #backbone_2
        c3_2_b_2 = self.c3_2_b(conv2_b + c3_3_n)
        conv3_b_2 = self.conv3_b(c3_2_b_2)#(1, 512, 40, 40)
        c3_3_b_2 = self.c3_3_b(conv3_b_2 + c3_2_n)#(1, 512, 40, 40)

        conv4_b_2 = self.conv4_b(c3_3_b_2)#(1, 1024, 20, 20)
        c3_4_b_2 = self.c3_4_b(conv4_b_2 + c3_1_n)  # (1, 1024, 20, 20)
        spp2 = self.SPPF(c3_4_b_2)



        # neck_stage_2
        c3_4_n = self.c3_1_n(spp2)#(1, 1024, 20, 20)


        upsample3 = self.upsample1(c3_4_n)
        # c3_3_b_2_a = self.g_l3(c3_3_b_2)
        concat3 = self.concat1([upsample3, c3_3_b_2])
        conv3_n = self.conv1_n(concat3)
        c3_5_n = self.c3_2_n(conv3_n)#(1, 512, 40, 40)

        upsample4 = self.upsample2(c3_5_n)
        # c3_2_b_2_a = self.g_l4(c3_2_b_2)
        concat4 = self.concat2([upsample4, c3_2_b_2])
        conv4_n = self.conv2_n(concat4)
        c3_6_n = self.c3_3_n(conv4_n)#(1, 256, 80, 80)

        #first no consider the fused
        out = [c3_4_n, c3_5_n, c3_6_n]

        #stage_1 and stage_2 fused
        return out




#-------------------------------RFPN_part
class RFPN(nn.Module):
    def __init__(self, c1, c2):
        super(RFPN, self).__init__()
        #--------------------load pretrain_weights
        self.focus = Focus(c1, 64, 3)#(b, 3, 640, 640)->(b, 64, 320, 320)
        self.conv1_b = Conv(64, 128, 3, 2)#(b, 64, 320, 320)->(b, 128, 160, 160)
        self.c3_1_b = C3(128, 128, n=3)#(b, 128, 160, 160)->(b, 128, 160, 160),short_cut=True,n=3
        self.conv2_b = Conv(128, 256, 3, 2)#(b, 128, 160, 160)->(b, 256, 80, 80)
        self.c3_2_b = C3(256, 256, n=9, rfp=True)#(b, 256, 80, 80)->(b, 256, 80, 80),short_cut=True,n=9
        self.conv3_b = Conv(256, 512, 3, 2)#(b, 256, 80, 80)->(b, 512, 40, 40)
        self.c3_3_b = C3(512, 512, n=9, rfp=True)  # (b, 512, 40, 80)->(b, 512, 40, 40),short_cut=True,n=9
        self.conv4_b = Conv(512, 1024, 3, 2)  # (b, 512, 40, 40)->(b, 1024, 20, 20)
        self.SPP = SPP(1024, 1024)
        #----------------------------------
        self.c3_4_b = C3(1024, 1024, n=6, rfp=True)  # (b, 1024, 20, 20)->(b, 1024, 20, 20),short_cut=True,n=3

        #-------------------RFPN_stage_1
        self.c3_1_n = C3(1024, 1024, n=3, shortcut=False)  # (b, 1024, 20, 20)->(b, 1024, 20, 20),short_cut=False,n=3

        self.upsample1 = nn.Upsample(size=None, scale_factor=2, mode='nearest')#(b, 1024, 20, 20)->(b, 1024, 40, 40)
        self.g_l1 = MS_CAM(512, 4)
        self.concat1 = Concat()#(b, 1024, 40, 40)-(b, 512, 40, 40)->(b, 1536, 40, 40)
        self.conv1_n = Conv(1536, 512, 1, 1)#(b, 1536, 40, 40)->(b, 512, 40, 40)
        self.c3_2_n = C3(512, 512, n=3, shortcut=False)# (b, 512, 40, 40)->(b, 512, 40, 40),short_cut=False,n=3

        self.upsample2 = nn.Upsample(size=None, scale_factor=2, mode='nearest')  # (b, 512, 40, 40)->(b, 512, 80, 80)
        self.g_l2 = MS_CAM(256, 4)
        self.concat2 = Concat()  # (b, 512, 80, 80)-(b, 256, 80, 80)->(b, 768, 80, 80)
        self.conv2_n = Conv(768, 256, 1, 1)  # (b, 768, 80, 80)->(b, 256, 80, 80)
        self.c3_3_n = C3(256, 256, n=3, shortcut=False)  # (b, 256, 80, 80)->(b, 256, 80, 80),short_cut=False,n=3

        #--------------------------backbone_stage_2 load pretrain weights
        # self.c3_5_b = C3(256, 256, n=9)  # (b, 256, 80, 80)->(b, 256, 80, 80),short_cut=True,n=9
        # self.conv5_b = Conv(256, 512, 3, 2)#(b, 256, 80, 80)->(b, 512, 40, 40)
        # self.c3_6_b = C3(512, 512, n=9)  # (b, 512, 40, 80)->(b, 512, 40, 40),short_cut=True,n=9
        # self.conv6_b = Conv(512, 1024, 3, 2)  # (b, 512, 40, 40)->(b, 1024, 20, 20)
        # self.SPP2 = SPP(1024, 1024)
        # self.c3_7_b = C3(1024, 1024, n=6)  # (b, 1024, 20, 20)->(b, 1024, 20, 20),short_cut=True,n=3

        #--------------------------RFPN_stage_2
        # self.c3_4_n = C3(1024, 1024, n=3, shortcut=False)  # (b, 1024, 20, 20)->(b, 1024, 20, 20),short_cut=False,n=3
        #
        # self.upsample3 = nn.Upsample(size=None, scale_factor=2, mode='nearest')  # (b, 1024, 20, 20)->(b, 1024, 40, 40)
        self.g_l3 = MS_CAM(512, 4)
        # self.concat3 = Concat()  # (b, 1024, 40, 40)-(b, 512, 40, 40)->(b, 1536, 40, 40)
        # self.conv3_n = Conv(1536, 512, 1, 1)  # (b, 1536, 40, 40)->(b, 512, 40, 40)
        # self.c3_5_n = C3(512, 512, n=3, shortcut=False)  # (b, 512, 40, 40)->(b, 512, 40, 40),short_cut=False,n=3
        #
        # self.upsample4 = nn.Upsample(size=None, scale_factor=2, mode='nearest')  # (b, 512, 40, 40)->(b, 512, 80, 80)
        self.g_l4 = MS_CAM(256, 4)
        # self.concat4 = Concat()  # (b, 512, 80, 80)-(b, 256, 80, 80)->(b, 768, 80, 80)
        # self.conv4_n = Conv(768, 256, 1, 1)  # (b, 768, 80, 80)->(b, 256, 80, 80)
        # self.c3_6_n = C3(256, 256, n=3, shortcut=False)  # (b, 256, 80, 80)->(b, 256, 80, 80),short_cut=False,n=3



    def forward(self, x):
        #backbone_1
        focus = self.focus(x)#(1, 64, 320, 320)
        conv1_b = self.conv1_b(focus)#(1, 128, 160, 160)
        c3_1_b = self.c3_1_b(conv1_b)#(1, 128, 160, 160)

        conv2_b = self.conv2_b(c3_1_b)#(1, 256, 80, 80)
        stage_1_1_rfp = torch.zeros_like(conv2_b)
        c3_2_b = self.c3_2_b([conv2_b, stage_1_1_rfp])#(1, 256, 80, 80)

        conv3_b = self.conv3_b(c3_2_b)#(1, 512, 40, 40)
        stage_1_2_rfp = torch.zeros_like(conv3_b)
        c3_3_b = self.c3_3_b([conv3_b, stage_1_2_rfp])#(1, 512, 40, 40)

        conv4_b = self.conv4_b(c3_3_b)#(1, 1024, 20, 20)
        spp = self.SPP(conv4_b)  # (1, 1024, 20, 20)
        stage_1_3_rfp = torch.zeros_like(spp)
        c3_4_b = self.c3_4_b([spp, stage_1_3_rfp])#(1, 1024, 20, 20)


        #neck_stage_1
        c3_1_n = self.c3_1_n(c3_4_b)#(1, 1024, 20, 20)

        upsample1 = self.upsample1(c3_1_n)
        c3_3_b_a = self.g_l1(c3_3_b)
        concat1 = self.concat1([upsample1, c3_3_b_a])
        conv1_n = self.conv1_n(concat1)
        c3_2_n = self.c3_2_n(conv1_n)#(1, 512, 40, 40)

        upsample2 = self.upsample2(c3_2_n)
        c3_2_b_a = self.g_l2(c3_2_b)
        concat2 = self.concat2([upsample2, c3_2_b_a])
        conv2_n = self.conv2_n(concat2)
        c3_3_n = self.c3_3_n(conv2_n)#(1, 256, 80, 80)


        #backbone_2
        c3_2_b_2 = self.c3_2_b([conv2_b, c3_3_n])
        conv3_b_2 = self.conv3_b(c3_2_b_2)#(1, 512, 40, 40)
        c3_3_b_2 = self.c3_3_b([conv3_b_2, c3_2_n])#(1, 512, 40, 40)

        conv4_b_2 = self.conv4_b(c3_3_b_2)#(1, 1024, 20, 20)
        spp2 = self.SPP(conv4_b_2)
        c3_4_b_2 = self.c3_4_b([spp2, c3_1_n])#(1, 1024, 20, 20)


        # neck_stage_2
        c3_4_n = self.c3_1_n(c3_4_b_2)#(1, 1024, 20, 20)


        upsample3 = self.upsample1(c3_4_n)
        c3_3_b_2_a = self.g_l3(c3_3_b_2)
        concat3 = self.concat1([upsample3, c3_3_b_2_a])
        conv3_n = self.conv1_n(concat3)
        c3_5_n = self.c3_2_n(conv3_n)#(1, 512, 40, 40)

        upsample4 = self.upsample2(c3_5_n)
        c3_2_b_2_a = self.g_l4(c3_2_b_2)
        concat4 = self.concat2([upsample4, c3_2_b_2_a])
        conv4_n = self.conv2_n(concat4)
        c3_6_n = self.c3_3_n(conv4_n)#(1, 256, 80, 80)

        #first no consider the fused
        out = [c3_4_n, c3_5_n, c3_6_n]

        #stage_1 and stage_2 fused
        return out



#-------------------------------RFPN_part
class RFPNl2(nn.Module):
    def __init__(self, c1, c2):
        super(RFPNl2, self).__init__()
        #--------------------load pretrain_weights
        self.focus = Focus(c1, 64, 3)#(b, 3, 640, 640)->(b, 64, 320, 320)
        self.conv1_b = Conv(64, 128, 3, 2)#(b, 64, 320, 320)->(b, 128, 160, 160)
        self.c3_1_b = C3(128, 128, n=3)#(b, 128, 160, 160)->(b, 128, 160, 160),short_cut=True,n=3
        self.conv2_b = Conv(128, 256, 3, 2)#(b, 128, 160, 160)->(b, 256, 80, 80)
        self.c3_2_b = C3(256, 256, n=9)#(b, 256, 80, 80)->(b, 256, 80, 80),short_cut=True,n=9
        self.conv3_b = Conv(256, 512, 3, 2)#(b, 256, 80, 80)->(b, 512, 40, 40)
        self.c3_3_b = C3(512, 512, n=9)  # (b, 512, 40, 80)->(b, 512, 40, 40),short_cut=True,n=9
        self.conv4_b = Conv(512, 1024, 3, 2)  # (b, 512, 40, 40)->(b, 1024, 20, 20)
        self.SPP = SPP(1024, 1024)
        #----------------------------------
        self.c3_4_b = C3(1024, 1024, n=6)  # (b, 1024, 20, 20)->(b, 1024, 20, 20),short_cut=True,n=3

        #-------------------RFPN_stage_1
        self.c3_1_n = C3(1024, 1024, n=3, shortcut=False)  # (b, 1024, 20, 20)->(b, 1024, 20, 20),short_cut=False,n=3

        self.upsample1 = nn.Upsample(size=None, scale_factor=2, mode='nearest')#(b, 1024, 20, 20)->(b, 1024, 40, 40)
        self.g_l1 = MS_CAM(512, 4)
        self.concat1 = Concat()#(b, 1024, 40, 40)-(b, 512, 40, 40)->(b, 1536, 40, 40)
        self.conv1_n = Conv(1536, 512, 1, 1)#(b, 1536, 40, 40)->(b, 512, 40, 40)
        self.c3_2_n = C3(512, 512, n=3, shortcut=False)# (b, 512, 40, 40)->(b, 512, 40, 40),short_cut=False,n=3

        self.upsample2 = nn.Upsample(size=None, scale_factor=2, mode='nearest')  # (b, 512, 40, 40)->(b, 512, 80, 80)
        self.g_l2 = MS_CAM(256, 4)
        self.concat2 = Concat()  # (b, 512, 80, 80)-(b, 256, 80, 80)->(b, 768, 80, 80)
        self.conv2_n = Conv(768, 256, 1, 1)  # (b, 768, 80, 80)->(b, 256, 80, 80)
        self.c3_3_n = C3(256, 256, n=3, shortcut=False)  # (b, 256, 80, 80)->(b, 256, 80, 80),short_cut=False,n=3

        self.upsample3 = nn.Upsample(size=None, scale_factor=2, mode='nearest')  # (b, 256, 80, 80)->(b, 256, 160, 160)
        self.g_l3 = MS_CAM(128, 4)
        self.concat3 = Concat()  # (b, 256, 160, 160)-(b, 128, 160, 160)->(b, 384, 160, 160)
        self.conv3_n = Conv(384, 128, 1, 1)  # (b, 384, 160, 160)->(b, 128, 160, 160)
        self.c3_4_n = C3(128, 128, n=3, shortcut=False)  # (b, 128, 160, 160)->(b, 128, 160, 160),short_cut=False,n=3

        #--------------------------backbone_stage_2 load pretrain weights
        # self.c3_5_b = C3(256, 256, n=9)  # (b, 256, 80, 80)->(b, 256, 80, 80),short_cut=True,n=9
        # self.conv5_b = Conv(256, 512, 3, 2)#(b, 256, 80, 80)->(b, 512, 40, 40)
        # self.c3_6_b = C3(512, 512, n=9)  # (b, 512, 40, 80)->(b, 512, 40, 40),short_cut=True,n=9
        # self.conv6_b = Conv(512, 1024, 3, 2)  # (b, 512, 40, 40)->(b, 1024, 20, 20)
        # self.SPP2 = SPP(1024, 1024)
        # self.c3_7_b = C3(1024, 1024, n=6)  # (b, 1024, 20, 20)->(b, 1024, 20, 20),short_cut=True,n=3

        #--------------------------RFPN_stage_2
        # self.c3_4_n = C3(1024, 1024, n=3, shortcut=False)  # (b, 1024, 20, 20)->(b, 1024, 20, 20),short_cut=False,n=3
        #
        # self.upsample3 = nn.Upsample(size=None, scale_factor=2, mode='nearest')  # (b, 1024, 20, 20)->(b, 1024, 40, 40)
        self.g_l4 = MS_CAM(512, 4)
        # self.concat3 = Concat()  # (b, 1024, 40, 40)-(b, 512, 40, 40)->(b, 1536, 40, 40)
        # self.conv3_n = Conv(1536, 512, 1, 1)  # (b, 1536, 40, 40)->(b, 512, 40, 40)
        # self.c3_5_n = C3(512, 512, n=3, shortcut=False)  # (b, 512, 40, 40)->(b, 512, 40, 40),short_cut=False,n=3
        #
        # self.upsample4 = nn.Upsample(size=None, scale_factor=2, mode='nearest')  # (b, 512, 40, 40)->(b, 512, 80, 80)
        self.g_l5 = MS_CAM(256, 4)
        # self.concat4 = Concat()  # (b, 512, 80, 80)-(b, 256, 80, 80)->(b, 768, 80, 80)
        # self.conv4_n = Conv(768, 256, 1, 1)  # (b, 768, 80, 80)->(b, 256, 80, 80)
        # self.c3_6_n = C3(256, 256, n=3, shortcut=False)  # (b, 256, 80, 80)->(b, 256, 80, 80),short_cut=False,n=3
        self.g_l6 = MS_CAM(128, 4)


    def forward(self, x):
        #backbone_1
        focus = self.focus(x)#(1, 64, 320, 320)
        conv1_b = self.conv1_b(focus)#(1, 128, 160, 160)
        c3_1_b = self.c3_1_b(conv1_b)#(1, 128, 160, 160)

        conv2_b = self.conv2_b(c3_1_b)#(1, 256, 80, 80)
        c3_2_b = self.c3_2_b(conv2_b)#(1, 256, 80, 80)

        conv3_b = self.conv3_b(c3_2_b)#(1, 512, 40, 40)
        c3_3_b = self.c3_3_b(conv3_b)#(1, 512, 40, 40)

        conv4_b = self.conv4_b(c3_3_b)#(1, 1024, 20, 20)
        spp = self.SPP(conv4_b)  # (1, 1024, 20, 20)
        c3_4_b = self.c3_4_b(spp)#(1, 1024, 20, 20)


        #neck_stage_1
        c3_1_n = self.c3_1_n(c3_4_b)#(1, 1024, 20, 20)

        upsample1 = self.upsample1(c3_1_n)
        c3_3_b_a = self.g_l1(c3_3_b)
        concat1 = self.concat1([upsample1, c3_3_b_a])
        conv1_n = self.conv1_n(concat1)
        c3_2_n = self.c3_2_n(conv1_n)#(1, 512, 40, 40)

        upsample2 = self.upsample2(c3_2_n)
        c3_2_b_a = self.g_l2(c3_2_b)
        concat2 = self.concat2([upsample2, c3_2_b_a])
        conv2_n = self.conv2_n(concat2)
        c3_3_n = self.c3_3_n(conv2_n)#(1, 256, 80, 80)

        upsample3 = self.upsample3(c3_3_n)
        c3_1_b_a = self.g_l3(c3_1_b)
        concat3 = self.concat2([upsample3, c3_1_b_a])
        conv3_n = self.conv3_n(concat3)
        c3_4_n = self.c3_4_n(conv3_n)  # (1, 128, 160, 160)



        #backbone_2
        c3_1_b_2 = self.c3_1_b(conv1_b + c3_4_n)
        conv2_b_2 = self.conv2_b(c3_1_b_2)
        c3_2_b_2 = self.c3_2_b(conv2_b_2 + c3_3_n)
        conv3_b_2 = self.conv3_b(c3_2_b_2)#(1, 512, 40, 40)
        c3_3_b_2 = self.c3_3_b(conv3_b_2 + c3_2_n)#(1, 512, 40, 40)

        conv4_b_2 = self.conv4_b(c3_3_b_2)#(1, 1024, 20, 20)
        spp2 = self.SPP(conv4_b_2)
        c3_4_b_2 = self.c3_4_b(spp2 + c3_1_n)#(1, 1024, 20, 20)


        # neck_stage_2
        c3_5_n = self.c3_1_n(c3_4_b_2)#(1, 1024, 20, 20)


        upsample4 = self.upsample1(c3_5_n)
        c3_3_b_2_a = self.g_l4(c3_3_b_2)
        concat4 = self.concat1([upsample4, c3_3_b_2_a])
        conv4_n = self.conv1_n(concat4)
        c3_6_n = self.c3_2_n(conv4_n)#(1, 512, 40, 40)

        upsample5 = self.upsample2(c3_6_n)
        c3_2_b_2_a = self.g_l5(c3_2_b_2)
        concat5 = self.concat2([upsample5, c3_2_b_2_a])
        conv5_n = self.conv2_n(concat5)
        c3_7_n = self.c3_3_n(conv5_n)#(1, 256, 80, 80)

        upsample6 = self.upsample3(c3_7_n)
        c3_1_b_2_a = self.g_l6(c3_1_b_2)
        concat6 = self.concat3([upsample6, c3_1_b_2_a])
        conv6_n = self.conv3_n(concat6)
        c3_8_n = self.c3_4_n(conv6_n)  # (1, 256, 80, 80)



        #first no consider the fused
        out = [c3_5_n, c3_6_n, c3_7_n, c3_8_n]

        #stage_1 and stage_2 fused
        return out

class RFPN_P(nn.Module):
    def __init__(self, p, c):
        super(RFPN_P, self).__init__()
        self.p = p

    def forward(self, x):
        return x[self.p]




class BottleneckCSP(nn.Module):
    # CSP Bottleneck https://github.com/WongKinYiu/CrossStagePartialNetworks
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super(BottleneckCSP, self).__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = nn.Conv2d(c1, c_, 1, 1, bias=False)
        self.cv3 = nn.Conv2d(c_, c_, 1, 1, bias=False)
        self.cv4 = Conv(2 * c_, c2, 1, 1)
        self.bn = nn.BatchNorm2d(2 * c_)  # applied to cat(cv2, cv3)
        self.act = nn.LeakyReLU(0.1, inplace=True)
        self.m = nn.Sequential(*[Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)])

    def forward(self, x):
        y1 = self.cv3(self.m(self.cv1(x)))
        y2 = self.cv2(x)
        return self.cv4(self.act(self.bn(torch.cat((y1, y2), dim=1))))


class C3(nn.Module):
    # CSP Bottleneck with 3 convolutions
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super(C3, self).__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)  # act=FReLU(c2)
        self.m = nn.Sequential(*[Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)])
        # self.m = nn.Sequential(*[CrossConv(c_, c_, 3, 1, g, 1.0, shortcut) for _ in range(n)])

    def forward(self, x):
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), dim=1))



class C3TR(C3):
    # C3 module with TransformerBlock()
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)
        self.m = TransformerBlock(c_, c_, 4, n)

#mix transoformer---
class MixDWConv(nn.Module):
    def __init__(self, dim=768):
        super(MixDWConv, self).__init__()
        self.dwconv = Conv(dim, dim, k=3, s=1, g=math.gcd(dim, dim), act=True)

    def forward(self, x):
        H = x[1]
        W = x[2]
        x = x[0]
        N, B, C = x.shape

        x = x.permute(1, 2, 0).view(B, C, H, W)
        x = self.dwconv(x)
        x = x.flatten(2).permute(2, 0, 1)
        return [x, H, W]

class MIXMlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.dwconv = MixDWConv(hidden_features)
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        H = x[1]
        W = x[2]
        x = x[0]
        x = self.fc1(x)
        x = self.dwconv([x, H, W])
        x = self.drop(x[0])
        x = self.fc2(x)
        x = self.drop(x)
        return [x, H, W]


class MIXAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., sr_ratio=1):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = nn.Conv2d(
                dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        H = x[1]
        W = x[2]
        x = x[0]
        N, B, C = x.shape
        q = self.q(x).reshape(B, N, self.num_heads, C //
                              self.num_heads).permute(0, 2, 1, 3)

        if self.sr_ratio > 1:
            x_ = x.permute(0, 2, 1).reshape(B, C, H, W)
            x_ = self.sr(x_).reshape(B, C, -1).permute(0, 2, 1)
            x_ = self.norm(x_)
            kv = self.kv(x_).reshape(B, -1, 2, self.num_heads,
                                     C // self.num_heads).permute(2, 0, 3, 1, 4)
        else:
            kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C //
                                    self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(N, B, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return [x, H, W]


class MIXBlock(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.SiLU, norm_layer=nn.LayerNorm, sr_ratio=1):
        super().__init__()
        # self.norm1 = norm_layer(dim)
        self.attn = MIXAttention(
            dim,
            num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop, sr_ratio=sr_ratio)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(
            drop_path) if drop_path > 0. else nn.Identity()
        # self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio / 2)
        # mlp_hidden_dim = dim
        self.mlp = MIXMlp(in_features=dim, hidden_features=mlp_hidden_dim,
                       drop=drop)

    def forward(self, x):
        H = x[1]
        W = x[2]
        x = x[0]
        x = x + self.drop_path(self.attn([x, H, W])[0])
        x = x + self.drop_path(self.mlp([x, H, W])[0])

        return [x, H, W]

class MixTRB(nn.Module):
    # Vision Transformer https://arxiv.org/abs/2010.11929
    def __init__(self, c1, c2, num_heads, num_layers):
        super().__init__()
        self.conv = None
        if c1 != c2:
            self.conv = Conv(c1, c2)
        self.tr = nn.Sequential(*[MIXBlock(c2, num_heads, qkv_bias=True) for _ in range(num_layers)])
        self.c2 = c2

    def forward(self, x):
        if self.conv is not None:
            x = self.conv(x)
            x = self.conv(x)
        b, _, h, w = x.shape
        p = x.flatten(2)
        p = p.unsqueeze(0)
        p = p.transpose(0, 3)
        p = p.squeeze(3)
        # e = self.linear(p)
        # x = p + e
        x = self.tr([p, h, w])[0]
        x = x.unsqueeze(3)
        x = x.transpose(0, 3)
        x = x.reshape(b, self.c2, h, w)
        return x


class C3MIXTR(C3):
    # C3 module with TransformerBlock()
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)
        self.m = MixTRB(c_, c_, 8, n)


#-----mix_transformer+vit transformer
class MixTransformerLayer(nn.Module):
    # Transformer layer https://arxiv.org/abs/2010.11929 (LayerNorm layers removed for better performance)
    def __init__(self, c, num_heads):
        super().__init__()
        self.q = nn.Linear(c, c, bias=False)
        self.k = nn.Linear(c, c, bias=False)
        self.v = nn.Linear(c, c, bias=False)
        self.ma = nn.MultiheadAttention(embed_dim=c, num_heads=num_heads)
        self.fc1 = nn.Linear(c, c, bias=False)
        self.dwconv = MixDWConv(c)
        self.fc2 = nn.Linear(c, c, bias=False)

    def forward(self, x):
        h = x[1]
        w = x[2]
        x = x[0]
        x = self.ma(self.q(x), self.k(x), self.v(x))[0] + x
        x = self.fc2(self.dwconv([self.fc1(x), h, w])[0]) + x
        x = [x, h, w]
        return x


class MixTransformerBlock(nn.Module):
    # Vision Transformer https://arxiv.org/abs/2010.11929
    def __init__(self, c1, c2, num_heads, num_layers):
        super().__init__()
        self.conv = None
        if c1 != c2:
            self.conv = Conv(c1, c2)
        self.tr = nn.Sequential(*[MixTransformerLayer(c2, num_heads) for _ in range(num_layers)])
        self.c2 = c2

    def forward(self, x):
        if self.conv is not None:
            x = self.conv(x)
            x = self.conv(x)
        b, _, h, w = x.shape
        p = x.flatten(2)
        p = p.unsqueeze(0)
        p = p.transpose(0, 3)
        p = p.squeeze(3)
        # e = self.linear(p)
        # x = p + e
        x = self.tr([p, h, w])[0]
        x = x.unsqueeze(3)
        x = x.transpose(0, 3)
        x = x.reshape(b, self.c2, h, w)
        return x

class C3MIX(C3):
    # C3 module with TransformerBlock()
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)
        self.m = MixTransformerBlock(c_, c_, 4, n)
#-----------


#---------------------vit transformer+detr postion_embedding

#bottlenck+MHSA(including positionembedding)
class PositionEmbeddingLearned(nn.Module):
    """
    Absolute pos embedding, learned.
    """

    def __init__(self, num_embeddings, num_pos_feats):
        super().__init__()
        self.row_embed = nn.Embedding(num_embeddings=num_embeddings, embedding_dim=num_pos_feats)
        self.col_embed = nn.Embedding(num_embeddings=num_embeddings, embedding_dim=num_pos_feats)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.uniform_(self.row_embed.weight)
        nn.init.uniform_(self.col_embed.weight)

    def forward(self, x):
        h, w = x.shape[-2:]
        i = torch.arange(w, device=x.device)
        j = torch.arange(h, device=x.device)

        x_emb = self.col_embed(i)
        y_emb = self.row_embed(j)

        pos = torch.cat([
            x_emb.unsqueeze(0).repeat(h, 1, 1),
            y_emb.unsqueeze(1).repeat(1, w, 1),
        ], dim=-1).permute(2, 0, 1).unsqueeze(0).repeat(x.shape[0], 1, 1, 1)
        return pos

class C3DETR(C3):
    # C3 module with TransformerBlock()
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)
        self.m = TransformerBlockDETR(c_, c_, 4, n)

class TransformerBlockDETR(nn.Module):
    # Vision Transformer https://arxiv.org/abs/2010.11929
    def __init__(self, c1, c2, num_heads, num_layers):
        super().__init__()
        self.conv = None
        if c1 != c2:
            self.conv = Conv(c1, c2)
        # self.linear = nn.Linear(c2, c2)  # learnable position embedding
        self.linear = PositionEmbeddingLearned(c2, c2 // 2)
        self.tr = nn.Sequential(*[TransformerLayer(c2, num_heads) for _ in range(num_layers)])
        self.c2 = c2

    def forward(self, x):
        if self.conv is not None:
            x = self.conv(x)
            x = self.conv(x)
        b, _, w, h = x.shape
        e = self.linear(x)
        x = x + e
        p = x.flatten(2)
        p = p.unsqueeze(0)
        p = p.transpose(0, 3)
        p = p.squeeze(3)

        x = self.tr(p)
        x = x.unsqueeze(3)
        x = x.transpose(0, 3)
        x = x.reshape(b, self.c2, w, h)
        return x

#-----------------------------------------------------------

#--------------------add model receptive field
class SPP(nn.Module):
    # Spatial pyramid pooling layer used in YOLOv3-SPP
    def __init__(self, c1, c2, k=(5, 9, 13)):
        super(SPP, self).__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * (len(k) + 1), c2, 1, 1)
        self.m = nn.ModuleList([nn.MaxPool2d(kernel_size=x, stride=1, padding=x // 2) for x in k])

    def forward(self, x):
        x = self.cv1(x)
        return self.cv2(torch.cat([x] + [m(x) for m in self.m], 1))

class SPPF(nn.Module):
    # Spatial Pyramid Pooling - Fast (SPPF) layer for YOLOv5 by Glenn Jocher
    def __init__(self, c1, c2, k=5):  # equivalent to SPP(k=(5, 9, 13))
        super().__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * 4, c2, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x):
        x = self.cv1(x)

        y1 = self.m(x)
        y2 = self.m(y1)
        return self.cv2(torch.cat([x, y1, y2, self.m(y2)], 1))


#dialtion conv instead max pool in SPP
class ASPP(nn.Module):
    def __init__(self, c1, c2, k=(2, 4, 6)):
        super(ASPP, self).__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * (len(k) + 1), c2, 1, 1)
        self.m = nn.ModuleList([nn.Conv2d(c_, c_, kernel_size=3, stride=1, dilation=x, padding=x) for x in k])

    def forward(self, x):
        x = self.cv1(x)
        return self.cv2(torch.cat([x] + [m(x) for m in self.m], 1))

#PPM
class PyramidPooling(nn.Module):
    """Pyramid pooling module"""

    def __init__(self, in_channels, out_channels, **kwargs):
        super(PyramidPooling, self).__init__()
        inter_channels = int(in_channels / 4)
        self.conv1 = Conv(in_channels, inter_channels, 1, **kwargs)
        self.conv2 = Conv(in_channels, inter_channels, 1, **kwargs)
        self.conv3 = Conv(in_channels, inter_channels, 1, **kwargs)
        self.conv4 = Conv(in_channels, inter_channels, 1, **kwargs)
        self.out = Conv(in_channels * 2, out_channels, 1)

    def pool(self, x, size):
        avgpool = nn.AdaptiveAvgPool2d(size)
        return avgpool(x)

    def upsample(self, x, size):
        return F.interpolate(x, size, mode='bilinear', align_corners=True)

    def forward(self, x):
        size = x.size()[2:]
        feat1 = self.upsample(self.conv1(self.pool(x, 1)), size)
        feat2 = self.upsample(self.conv2(self.pool(x, 2)), size)
        feat3 = self.upsample(self.conv3(self.pool(x, 4)), size)
        feat4 = self.upsample(self.conv4(self.pool(x, 6)), size)
        x = torch.cat([x, feat1, feat2, feat3, feat4], dim=1)  # concat 四个池化的结果
        x = self.out(x)
        return x
#-------------------------------------
#-------------feature fusion:ASFF
class ASFF(nn.Module):
    def __init__(self, level, ch=[1024, 512, 256, 128], rfb=False, vis=False):
        super(ASFF, self).__init__()
        self.level = level

        # self.dim = [512, 256, 256]
        self.dim = ch
        self.inter_dim = self.dim[self.level]

        if level == 0:
            self.stride_level_1 = Conv(self.dim[1], self.inter_dim, 3, 2)
            self.stride_level_2 = Conv(self.dim[2], self.inter_dim, 3, 2)
            self.stride_level_3 = Conv(self.dim[3], self.inter_dim, 3, 2)
            self.expand = Conv(self.inter_dim, 1024, 3, 1)

        elif level == 1:
            self.compress_level_0 = Conv(self.dim[0], self.inter_dim, 1, 1)
            self.stride_level_2 = Conv(self.dim[2], self.inter_dim, 3, 2)
            self.stride_level_3 = Conv(self.dim[3], self.inter_dim, 3, 2)
            self.expand = Conv(self.inter_dim, 512, 3, 1)

        elif level == 2:
            self.compress_level_0 = Conv(self.dim[0], self.inter_dim, 1, 1)
            if self.dim[1] != self.dim[2]:
                self.compress_level_1 = Conv(self.dim[1], self.inter_dim, 1, 1)
            self.stride_level_3 = Conv(self.dim[3], self.inter_dim, 3, 2)
            self.expand = Conv(self.inter_dim, 256, 3, 1)

        elif level == 3:
            self.compress_level_0 = Conv(self.dim[0], self.inter_dim, 1, 1)
            self.compress_level_1 = Conv(self.dim[1], self.inter_dim, 1, 1)
            self.compress_level_2 = Conv(self.dim[2], self.inter_dim, 1, 1)
            self.expand = Conv(self.inter_dim, 128, 3, 1)

        compress_c = 8 if rfb else 16  #when adding rfb, we use half number of channels to save memory

        self.weight_level_0 = Conv(self.inter_dim, compress_c, 1, 1)
        self.weight_level_1 = Conv(self.inter_dim, compress_c, 1, 1)
        self.weight_level_2 = Conv(self.inter_dim, compress_c, 1, 1)
        self.weight_level_3 = Conv(self.inter_dim, compress_c, 1, 1)

        self.weight_levels = nn.Conv2d(compress_c*4, 4, kernel_size=1, stride=1, padding=0)
        self.vis= vis

	# level_0 < level_1 < level_2
    def forward(self, x):
        x_level_0, x_level_1, x_level_2, x_level_3 = x
        if self.level == 0:
            level_0_resized = x_level_0
            level_1_resized = self.stride_level_1(x_level_1)

            level_2_downsampled_inter =F.max_pool2d(x_level_2, 3, stride=2, padding=1)
            level_2_resized = self.stride_level_2(level_2_downsampled_inter)

            level_3_downsampled_inter1 = F.max_pool2d(x_level_3, 3, stride=2, padding=1)
            level_3_downsampled_inter = F.max_pool2d(level_3_downsampled_inter1, 3, stride=2, padding=1)
            level_3_resized = self.stride_level_3(level_3_downsampled_inter)

        elif self.level == 1:
            level_0_compressed = self.compress_level_0(x_level_0)
            level_0_resized =F.interpolate(level_0_compressed, scale_factor=2, mode='nearest')

            level_1_resized =x_level_1

            level_2_resized =self.stride_level_2(x_level_2)

            level_3_downsampled_inter = F.max_pool2d(x_level_3, 3, stride=2, padding=1)
            level_3_resized = self.stride_level_3(level_3_downsampled_inter)

        elif self.level == 2:
            level_0_compressed = self.compress_level_0(x_level_0)
            level_0_resized =F.interpolate(level_0_compressed, scale_factor=4, mode='nearest')

            if self.dim[1] != self.dim[2]:
                level_1_compressed = self.compress_level_1(x_level_1)
                level_1_resized = F.interpolate(level_1_compressed, scale_factor=2, mode='nearest')
            else:
                level_1_resized =F.interpolate(x_level_1, scale_factor=2, mode='nearest')

            level_2_resized =x_level_2

            level_3_resized = self.stride_level_3(x_level_3)

        elif self.level == 3:
            level_0_compressed = self.compress_level_0(x_level_0)
            level_0_resized = F.interpolate(level_0_compressed, scale_factor=8, mode='nearest')

            level_1_compressed = self.compress_level_1(x_level_1)
            level_1_resized = F.interpolate(level_1_compressed, scale_factor=4, mode='nearest')

            level_2_compressed = self.compress_level_2(x_level_2)
            level_2_resized = F.interpolate(level_2_compressed, scale_factor=2, mode='nearest')

            level_3_resized = x_level_3



        level_0_weight_v = self.weight_level_0(level_0_resized)
        level_1_weight_v = self.weight_level_1(level_1_resized)
        level_2_weight_v = self.weight_level_2(level_2_resized)
        level_3_weight_v = self.weight_level_3(level_3_resized)

        levels_weight_v = torch.cat((level_0_weight_v, level_1_weight_v, level_2_weight_v, level_3_weight_v),1)
        levels_weight = self.weight_levels(levels_weight_v)
        levels_weight = F.softmax(levels_weight, dim=1)

        fused_out_reduced = level_0_resized * levels_weight[:,0:1,:,:]+\
                            level_1_resized * levels_weight[:,1:2,:,:]+\
                            level_2_resized * levels_weight[:,2:3,:,:]+\
                            level_3_resized * levels_weight[:, 3:, :, :]

        out = self.expand(fused_out_reduced)

        if self.vis:
            return out, levels_weight, fused_out_reduced.sum(dim=1)
        else:
            return out
#-------------

class Focus(nn.Module):
    # Focus wh information into c-space
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):  # ch_in, ch_out, kernel, stride, padding, groups
        super(Focus, self).__init__()
        self.conv = Conv(c1 * 4, c2, k, s, p, g, act)
        # self.contract = Contract(gain=2)

    def forward(self, x):  # x(b,c,w,h) -> y(b,4c,w/2,h/2)
        return self.conv(torch.cat([x[..., ::2, ::2], x[..., 1::2, ::2], x[..., ::2, 1::2], x[..., 1::2, 1::2]], 1))
        # return self.conv(self.contract(x))


class Contract(nn.Module):
    # Contract width-height into channels, i.e. x(1,64,80,80) to x(1,256,40,40)
    def __init__(self, gain=2):
        super().__init__()
        self.gain = gain

    def forward(self, x):
        N, C, H, W = x.size()  # assert (H / s == 0) and (W / s == 0), 'Indivisible gain'
        s = self.gain
        x = x.view(N, C, H // s, s, W // s, s)  # x(1,64,40,2,40,2)
        x = x.permute(0, 3, 5, 1, 2, 4).contiguous()  # x(1,2,2,64,40,40)
        return x.view(N, C * s * s, H // s, W // s)  # x(1,256,40,40)


class Expand(nn.Module):
    # Expand channels into width-height, i.e. x(1,64,80,80) to x(1,16,160,160)
    def __init__(self, gain=2):
        super().__init__()
        self.gain = gain

    def forward(self, x):
        N, C, H, W = x.size()  # assert C / s ** 2 == 0, 'Indivisible gain'
        s = self.gain
        x = x.view(N, s, s, C // s ** 2, H, W)  # x(1,2,2,16,80,80)
        x = x.permute(0, 3, 4, 1, 5, 2).contiguous()  # x(1,16,80,2,80,2)
        return x.view(N, C // s ** 2, H * s, W * s)  # x(1,16,160,160)


class Concat(nn.Module):
    # Concatenate a list of tensors along dimension
    def __init__(self, dimension=1):
        super(Concat, self).__init__()
        self.d = dimension

    def forward(self, x):
        return torch.cat(x, self.d)


class NMS(nn.Module):
    # Non-Maximum Suppression (NMS) module
    conf = 0.25  # confidence threshold
    iou = 0.45  # IoU threshold
    classes = None  # (optional list) filter by class
    max_det = 1000  # maximum number of detections per image

    def __init__(self):
        super(NMS, self).__init__()

    def forward(self, x):
        return non_max_suppression(x[0], self.conf, iou_thres=self.iou, classes=self.classes, max_det=self.max_det)


#-------------
#SElayer:
#-------------
class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)
#-----------
#MSAM:
#-----------
class MSAM(nn.Module):
    def __init__(self, low_channel, high_channel):
        super(MSAM, self).__init__()
        self.low_channel = low_channel
        self.high_channel = high_channel
        self.attention_weight = nn.Sequential(
            Conv(self.low_channel, self.high_channel, k=3, p=1, s=2),
            nn.Sigmoid(),
        )
        # self.downsample = nn.Sequential(
        #     Conv(self.low_channel, self.high_channel, k=1, p=0, s=1),
        #     nn.Sigmoid(),
        # )

        # self.conv_downsample = Conv(self.low_channel, self.high_channel, k=3, p=1, s=2)

    def forward(self, x):
        low_x = x[0]
        high_x = x[1]
        attention_weight = self.attention_weight(low_x)#(b, c, h, w)->(b, c, h, w)
        # down_attention_weight = self.downsample(F.interpolate(attention_weight, size=None, scale_factor=1/2, mode='nearest'))
        # down_attention_weight = self.conv_downsample(attention_weight)
        out = high_x + attention_weight * high_x
        return out


#-------ARF:add receptive field
class ARF(nn.Module):
    def __init__(self, in_dim, reduction = 1/4):
        super(ARF, self).__init__()
        #4个串联的3*3conv提高感受野
        self.in_dim = in_dim
        self.re_dim = int(self.in_dim * reduction)
        self.arf_de = Conv(self.in_dim, self.re_dim, k=1, s=1, p=0)
        self.arf_one = Conv(self.re_dim, self.re_dim, k=3, s=1, p=1)
        self.arf_two = Conv(self.re_dim, self.re_dim, k=3, s=1, p=1)
        self.arf_three = Conv(self.re_dim, self.re_dim, k=3, s=1, p=1)
        self.arf_four = Conv(self.re_dim, self.re_dim, k=3, s=1, p=1)


    def forward(self, x):
        input = x if len(x[0].shape) == 3 else x[0]
        re_x = self.arf_de(input)
        out1 = self.arf_one(re_x)#out1
        out2 = self.arf_two(out1)#out2
        out3 = self.arf_three(out2)#out3
        out4 = self.arf_four(out3)#out4
        out = torch.cat([out1, out2, out3, out4], dim=1)
        return out

class DARF(nn.Module):
    def __init__(self, in_dim, reduction = 1/4):
        super(DARF, self).__init__()
        self.in_dim = in_dim
        self.re_dim = int(self.in_dim * reduction)
        self.arf_de = Conv(self.in_dim, self.re_dim, k=1, s=1, p=0)
        self.arf_one = Conv(self.re_dim, self.re_dim, k=3, s=1, p=2, r=2)
        self.arf_two = Conv(self.re_dim, self.re_dim, k=3, s=1, p=2, r=2)
        self.arf_three = Conv(self.re_dim, self.re_dim, k=3, s=1, p=2, r=2)
        self.arf_four = Conv(self.re_dim, self.re_dim, k=3, s=1, p=2, r=2)


    def forward(self, x):
        input = x if len(x[0].shape) == 3 else x[0]
        re_x = self.arf_de(input)
        out1 = self.arf_one(re_x)#out1
        out2 = self.arf_two(out1)#out2
        out3 = self.arf_three(out2)#out3
        out4 = self.arf_four(out3)#out4
        out = torch.cat([out1, out2, out3, out4], dim=1)
        return out


#------
class Add(nn.Module):
    def __init__(self, ignore):
        super(Add, self).__init__()
        self.ignore = ignore

    def forward(self, x):
        step_2_x = x[1]
        step_2_f = x[0]
        out = step_2_x + step_2_f
        return out

class ShortCut(nn.Module):
    def __init__(self, channel):
        super(ShortCut, self).__init__()
        self.channel = channel

    def forward(self, x):
        fpn_1 = x[0]
        fpn_2 = x[1]
        out = fpn_1 + fpn_2

        return out
#---
class CBAM(nn.Module):
    def __init__(self):
        super(CBAM, self).__init__()


    def forward(self):
        return 0

#------------grainess_feature:Graininess-Aware_Deep_Feature_Learning_for_Pedestrian_Detection
#up:nn.Upsample()
#down:nn.functional.interpolate()
import torch.nn.functional as F
class PAM(nn.Module):
    def __init__(self, c1, c2, Sc=4, r=2):
        super(PAM, self).__init__()
        self.P3_conv = nn.Sequential(
            Conv(c1, Sc, k=1, s=1, p=0),
            nn.Upsample(scale_factor=8, mode='nearest'),
        )
        self.P4_conv = nn.Sequential(
            Conv(c2, Sc, k=1, s=1, p=0),
            nn.Upsample(scale_factor=16, mode='nearest'),
        )
        self.P4_Conv_mask = nn.Sequential(
            nn.MaxPool2d(kernel_size=2),
            Conv(c2, c2, k=3, s=1, r=2, p=2),
            Conv(c2, c2, k=3, s=1, r=2, p=2),
            Conv(c2, c2, k=3, s=1, r=2, p=2),
            Conv(c2, Sc, k=1, s=1),
            nn.Upsample(scale_factor=32, mode='nearest'),
        )

        self.Image_mask = nn.Sequential(
            Concat(),
            Conv(Sc * 3, Sc, k=1, s=1),
            nn.Softmax(dim=1)#channel dimention to every pixel to background,small,medium
        )

    def forward(self, x):

        P3_conv = self.P3_conv(x[0])#(2, 4, 640, 640)
        P4_conv = self.P4_conv(x[1])#(2, 4, 640, 640)
        P4_conv_mask = self.P4_Conv_mask(x[1])#(2, 4, 640, 640)
        Image_mask = self.Image_mask([P3_conv, P4_conv, P4_conv_mask])#(2,4,640,640)

        return Image_mask


class ZIZOM(nn.Module):#aggrated small pedestrian feature
    def __init__(self, c1, c2, c3):
        super(ZIZOM, self).__init__()

        self.up_sample2 = nn.Upsample(scale_factor=2, mode='nearest')
        self.up_sample4 = nn.Upsample(scale_factor=4, mode='nearest')

        self.convP4 = Conv(c2, c1, k=1, s=1)
        self.convP5 = Conv(c3, c1, k=1, s=1)

        self.down_mask = nn.Sequential(
            nn.MaxPool2d(kernel_size=2),
            nn.MaxPool2d(kernel_size=2),
            nn.MaxPool2d(kernel_size=2),
            # nn.MaxPool2d(kernel_size=8),
        )

        #L2 norm
        self.F3_l2 = L2Norm(c1, 10.)
        self.F4_l2 = L2Norm(c1, 10.)
        self.F5_l2 = L2Norm(c1, 10.)

        #concat+conv
        self.concat_conv = nn.Sequential(
            Concat(),
            Conv(c1*3, c1, k=1, s=1),
        )

        #P4,P5
        self.down_P4_mask = nn.Sequential(
            nn.MaxPool2d(kernel_size=2),
        )
        self.down_P5_mask = nn.Sequential(
            nn.MaxPool2d(kernel_size=2),
        )


        #P3,P4,P5 down sample to Image mask
        # self.down_P3_mask = nn.Sequential(
        #     nn.MaxPool2d(kernel_size=8),
        # )
        # self.down_P4_mask = nn.Sequential(
        #     nn.MaxPool2d(kernel_size=2),
        # )
        # self.down_P5_mask = nn.Sequential(
        #     nn.MaxPool2d(kernel_size=2),
        # )



    def forward(self, x):
        #x[0](2,192,80,80),x[1](2,384,40,40),x[2](2,768,20,20) to P3,P4,P5 out,x[3](2,4,640,640) to image_mask
        conv_F4 = self.convP4(x[1])#(2,256,40,40)
        conv_F5 = self.convP5(x[2])#(2,256,20,20)

        down_mask_result = self.down_mask(x[3])#(2,4,80,80)
        small_pes_mask = down_mask_result[:, 1:2, :, :]#(2,1,80,80)
        medium_pes_mask = down_mask_result[:, 2:3, :, :]#(2,1,80,80)
        large_pes_mask = down_mask_result[:, 3:4, :, :]#(2,1,80,80)

        F3_m_mask = small_pes_mask * x[0] #(2,256,80,80)
        F4_m_mask = medium_pes_mask * self.up_sample2(conv_F4)#P4*rescale(mask)#(2,256,80,80)
        F5_m_mask = large_pes_mask * self.up_sample4(conv_F5)#P5*rescale(mask)#(2,256,80,80)

        #L2-norm,concat
        F3_l2 = self.F3_l2(F3_m_mask)
        F4_l2 = self.F4_l2(F4_m_mask)
        F5_l2 = self.F5_l2(F5_m_mask)

        outP3 = self.concat_conv([F3_l2,F4_l2,F5_l2])#(2,256,80,80)
        outP4 = self.down_P4_mask(medium_pes_mask) * x[1]#(2,1,40,40),(2,512,40,40)
        outP5 = self.down_P5_mask(self.down_P4_mask(large_pes_mask)) * x[2]#(2,1,20,20),(2,1024,20,20)

        #out P3,P4,P5 mask and image mask
        out = []
        out.append(outP3)#out_P3'
        out.append(outP4)#out_P4'
        out.append(outP5)#out_P5'
        out.append(x[3])#image mask

        #P3,P4,P5 * Image mask
        # out = []
        # image_P3_mask = self.down_P3_mask(x[3])
        # small_pes_P3_mask = image_P3_mask[:, 1:2, :, :]  # (2,1,80,80)
        # medium_pes_P3_mask = image_P3_mask[:, 2:3, :, :]  # (2,1,80,80)
        # large_pes_P3_mask = image_P3_mask[:, 3:4, :, :]  # (2,1,80,80)
        #
        # medium_pes_P4_mask = self.down_P4_mask(medium_pes_P3_mask)
        # large_pes_P4_mask = self.down_P4_mask(large_pes_P3_mask)
        # large_pes_P5_mask = self.down_P5_mask(large_pes_P4_mask)
        # P3_mask = x[0] * small_pes_P3_mask
        # P4_mask = x[1] * medium_pes_P4_mask
        # P5_mask = x[2] * large_pes_P5_mask
        # out.append(P3_mask)
        # out.append(P4_mask)
        # out.append(P5_mask)
        # out.append(x[3])

        return out

class Get_P_mask(nn.Module):
    def __init__(self, channel):
        super(Get_P_mask, self).__init__()
        self.channel = channel

    def forward(self, x):
        if self.channel == 256:#P3
            out = x[0]
        elif self.channel == 512:#P4
            out = x[1]
        elif self.channel == 1024:#P5
            out = x[2]
        return out

class Mul_Mask(nn.Module):#medium large * mask
    def __init__(self, downsample_ratio, mask_index):#down_sample,mask_index
        super(Mul_Mask, self).__init__()
        self.mask_index = mask_index
        self.downsample_ratio = downsample_ratio
        self.downsample = nn.MaxPool2d(kernel_size=downsample_ratio)

    def forward(self, x):
        x_m_mask = x[0] * self.downsample(x[1][:, self.mask_index:self.mask_index+1, :, :]).expand_as(x[0])
        # x_m_mask = x[0] * F.interpolate(x[1][:, self.mask_index:self.mask_index + 1, :, :], scale_factor=1/self.downsample_ratio).expand_as(x[0])#(2,512,40,40),(2,1024,20,20)
        return x_m_mask

#--------------CA-GDFL
class Common_AS(nn.Module):
    def __init__(self, c1, c2):
        # c1 to now detect channel,c2=256
        super(Common_AS,  self).__init__()
        # attention and segmentation common part
        self.common_conv = nn.Sequential(
            Conv(c1, c2, k=3, s=1, p=1),
            Conv(c2, c2, k=3, s=1, p=1),
            Conv(c2, c2, k=3, s=1, p=1),
        )
    def forward(self, x):
        out = self.common_conv(x)
        return out

class Attention_AS(nn.Module):
    def __init__(self, c1, c2):
        super(Attention_AS, self).__init__()
        #attention branch
        self.attention_branch = nn.Sequential(
            Conv(c1, c2, k=1, s=1, p=0),
        )#attention map and feature map to same shape
    def forward(self, x):
        p_in = x[0]  # detect layer original out
        common_in = x[1]  # attention branch and segmentation branch common part
        attention_weight = self.attention_branch(common_in)
        out = attention_weight * p_in
        return out

class Segmentation_AS(nn.Module):
    def __init__(self, c1, c2, scale_factor):
        super(Segmentation_AS, self).__init__()
        #segmentation branch
        self.segmentation_branch = nn.Sequential(
            Conv(c2, 2, k=1, s=1, p=0),
            nn.Upsample(scale_factor=scale_factor, mode='nearest'),
            nn.Softmax(dim=1),
            # nn.Sigmoid(),

        )

    def forward(self, x):
        out = self.segmentation_branch(x)
        return out


class Collect_segmentation(nn.Module):
    def __init__(self, c1, c2, c3):
        super(Collect_segmentation, self).__init__()

    def forward(self, x):
        #channel=256,512,1024 detect layer out
        seg_256 = x[0][None]
        seg_512 = x[1][None]
        seg_1024 = x[2][None]
        out = torch.cat([seg_256, seg_512, seg_1024], dim=0)#(3, batch_size, 2, h_img, w_img) or (3, batch_size, 1, h_img, w_img)

        return out
#--------------------
#--------------------Improving Occlusion and Hard Negative Handling for Single-Stage Pedestrian Detectors
class Get_grid_confidence_map(nn.Module):
    def __init__(self):
        super(Get_grid_confidence_map, self).__init__()


    def forward(self, x):
        #x[0]:feature map,x[1]:targets,x[2]:stride
        features = x[0]
        targets = x[1]
        stride = x[2]
        b, c, h, w = features.shape
        yv, xv = torch.meshgrid([torch.arange(h), torch.arange(w)])
        stride_h, stride_w = stride, stride

        #get detection layers to gt grid map
        xyxy = []
        for j in range(h - 1):
            x1_values = xv[0][:-1][None]  # [0....80]
            y1_values = yv[j][:-1][None]  # [j....j]
            x2_values = xv[0][1:][None]  # [1....81]
            y2_values = yv[j + 1][1:][None]  # [j+1....j+1]
            xyxy_j = torch.cat([x1_values, y1_values, x2_values, y2_values], dim=0).T
            xyxy.append(xyxy_j)
        xyxy_tensor = torch.cat(xyxy, dim=0)

        for index in range(len(features)):
            target = targets[targets[:, 0] == index][:, 2:6]
            if len(target):
                target_abs = target.clone()
                w_abs, h_abs = target[:, 2:3] * stride_w, target[:, 3:4] * stride_h
                target_abs[:, 0:1] = target[:, 0:1] * stride_w - w_abs / 2  # x1
                target_abs[:, 1:2] = target[:, 1:2] * stride_h - h_abs / 2  # y1
                target_abs[:, 2:3] = target[:, 0:1] * stride_w + w_abs / 2  # x2
                target_abs[:, 3:4] = target[:, 1:2] * stride_h + h_abs / 2  # y2


                tar_grid_confidence_map = []
                for tar_index in range(len(target_abs)):
                    grid_confidence_tar_index = bbox_iou(target_abs[tar_index], xyxy_tensor, grid_cell_flag=True)
                    grid_confidence_tensor = grid_confidence_tar_index.view(h - 1, w - 1)[None]
                    tar_grid_confidence_map.append(grid_confidence_tensor)
                tar_grid_confidence_map_tensor = torch.cat(tar_grid_confidence_map, dim=0).max(0)[0]
        return 0
#-------------------

#-----------------------Adaptive NMS
class ANMS(nn.Module):
    def __init__(self):
        super(ANMS, self).__init__()


    def forward(self, x):
        #(b, 256, h, w) -> (b, 18)
        return 0

#-----------------------
#-----------------------
#CBAM
def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes,
                     out_planes,
                     kernel_size=3,
                     stride=stride,
                     padding=1,
                     bias=False)


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=4):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.sharedMLP = nn.Sequential(
            nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False),
            nn.SiLU(),
            nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avgout = self.sharedMLP(self.avg_pool(x))
        maxout = self.sharedMLP(self.max_pool(x))
        return self.sigmoid(avgout + maxout)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), "kernel size must be 3 or 7"
        padding = 3 if kernel_size == 7 else 1

        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avgout = torch.mean(x, dim=1, keepdim=True)
        maxout, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avgout, maxout], dim=1)
        x = self.conv(x)
        return self.sigmoid(x)


class CBAM(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(CBAM, self).__init__()
        self.conv = conv3x3(inplanes, planes, stride)
        self.bn = nn.BatchNorm2d(planes)
        self.silu = nn.SiLU(inplace=True)

        self.ca = ChannelAttention(planes)
        self.sa = SpatialAttention()

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv(x)
        out = self.bn(out)
        out = self.silu(out)


        out = self.ca(out) * out
        out = self.sa(out) * out

        # if self.downsample is not None:
        #     print("downsampling")
        #     residual = self.downsample(x)
        #
        # print(out.shape, residual.shape)

        out += residual
        out = self.silu(out)

        return out
#----------------------------------------------
#--------------CCNet
import torch.nn as nn
from torch.nn import Softmax
class CrissCrossAttention(nn.Module):
    """ Criss-Cross Attention Module"""
    def __init__(self, in_dim):
        super(CrissCrossAttention, self).__init__()
        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.softmax = Softmax(dim=3)
        self.gamma = nn.Parameter(torch.zeros(1))

    def INF(self, B, H, W, device):
        return -torch.diag(torch.tensor(float("inf")).repeat(H), 0).unsqueeze(0).repeat(B * W, 1, 1).to(device)  # (b*w, h, h)

    def forward(self, x1, x2, subtract_flag=False):

        m_batchsize, _, height, width = x2.size()  # (2,192,80,80)
        #Q
        proj_query = self.query_conv(x2)  # (2, 24, 80, 80)
        proj_query_H = proj_query.permute(0, 3, 1, 2).contiguous().view(m_batchsize * width, -1, height).permute(0, 2, 1)  # (2,24,80,80)->(2*80,80,24)
        proj_query_W = proj_query.permute(0, 2, 1, 3).contiguous().view(m_batchsize * height, -1, width).permute(0, 2, 1)  # (2,24,80,80)-(2*80,80,24)
        #K
        proj_key = self.key_conv(x2)  # (2, 24, 80, 80)
        proj_key_H = proj_key.permute(0, 3, 1, 2).contiguous().view(m_batchsize * width, -1, height)  # (2,24,80,80)->(2*80,24,80)
        proj_key_W = proj_key.permute(0, 2, 1, 3).contiguous().view(m_batchsize * height, -1, width)  # (2,24,80,80)->(2*80,24,80)

        #V
        proj_value = self.value_conv(x1)  # (2, 192, 80, 80)
        proj_value_H = proj_value.permute(0, 3, 1, 2).contiguous().view(m_batchsize * width, -1, height)  # (2,192,80,80)->(2*80,192,80)
        proj_value_W = proj_value.permute(0, 2, 1, 3).contiguous().view(m_batchsize * height, -1, width)  # (2,192,80,80)->(2*80,192,80)

        #Aggregation
        energy_H = (torch.bmm(proj_query_H, proj_key_H)).view(m_batchsize, width, height, height).permute(0, 2, 1, 3)  #()
        energy_W = torch.bmm(proj_query_W, proj_key_W).view(m_batchsize, height, width, width)  # (b, h, w, w)-(2, 4, 6, 6)
        concate = self.softmax(torch.cat([energy_H, energy_W], 3))

        att_H = concate[:, :, :, 0:height].permute(0, 2, 1, 3).contiguous().view(m_batchsize * width, height, height)
        att_W = concate[:, :, :, height:height + width].contiguous().view(m_batchsize * height, width, width)

        if subtract_flag:
            att_H = 1 - att_H
            att_W = 1 - att_W

        out_H = torch.bmm(proj_value_H, att_H.permute(0, 2, 1)).view(m_batchsize, width, -1, height).permute(0, 2, 3, 1)
        out_W = torch.bmm(proj_value_W, att_W.permute(0, 2, 1)).view(m_batchsize, height, -1, width).permute(0, 2, 1, 3)

        return out_H + out_W

class RCCA(nn.Module):
    def __init__(self, in_dim):
        super(RCCA, self).__init__()
        self.ccnet = CrissCrossAttention(in_dim)

    def forward(self, x):
        x_low_level = x[1]
        x_high_level = x[0]
        feature_fusion = x_high_level + x_low_level
        ccnet_feature_fusion = self.ccnet(feature_fusion, feature_fusion)

        x_low_level_ccnet = self.ccnet(x_low_level, ccnet_feature_fusion)
        x_high_level_ccnet = self.ccnet(x_high_level, ccnet_feature_fusion, subtract_flag=True)
        out = x_high_level_ccnet + x_low_level_ccnet
        return out
#-----RFPN



class Shared_fi(nn.Module):
    def __init__(self):
        super(Shared_fi, self).__init__()


    def forward(self, x):
        return 0

#-----

#--------------------
#------------AFF_NL
# class Simplified_NL(nn.Module):
#     def __init__(self, in_dim):
#         super(Simplified_NL, self).__init__()
#         self.conv1x1 = nn.Conv2d(in_dim, 1, kernel_size=1, stride=1, padding=0)
#         self.in_dim = in_dim
#         self.softmax = nn.Softmax(dim = 1)
#     def forward(self, x):
#         wk = self.conv1x1(x)#(b, 1, h, w)->(b, h, w)
#         b = wk.shape[0]
#         wk = wk.view(b, 1, -1).permute(0, 2, 1).contiguous()#(b, 1, h*w)->(b, h*w, 1)
#         wk_softmax = self.softmax(wk)
#         x = x.view(b, self.in_dim, -1)#(b, c, h, w)->(b, c, h*w)
#         x_wk = torch.bmm(x, wk_softmax).unsqueeze(-1)#(b, c, 1)->(b, c, 1, 1)
#
#         return x_wk

class Simplified_NL(nn.Module):
    def __init__(self, inplanes, pooling_type='att',
                 fusion_types=('channel_add',)):
        super(Simplified_NL, self).__init__()

        assert pooling_type in ['avg', 'att']
        assert isinstance(fusion_types, (list, tuple))
        assert len(fusion_types) > 0, 'at least one fusion should be used'

        self.inplanes = inplanes
        self.pooling_type = pooling_type
        self.fusion_types = fusion_types

        if pooling_type == 'att':
            self.conv_mask = nn.Conv2d(inplanes, 1, kernel_size=1)
            self.softmax = nn.Softmax(dim=2)
        else:
            self.avg_pool = nn.AdaptiveAvgPool2d(1)


    def spatial_pool(self, x):
        batch, channel, height, width = x.size()
        if self.pooling_type == 'att':
            input_x = x
            # [N, C, H * W]
            input_x = input_x.view(batch, channel, height * width)
            # [N, 1, C, H * W]
            input_x = input_x.unsqueeze(1)
            # [N, 1, H, W]
            context_mask = self.conv_mask(x)
            # [N, 1, H * W]
            context_mask = context_mask.view(batch, 1, height * width)
            # [N, 1, H * W]
            context_mask = self.softmax(context_mask)
            # [N, 1, H * W, 1]
            context_mask = context_mask.unsqueeze(-1)
            # [N, 1, C, 1]
            context = torch.matmul(input_x, context_mask)
            # [N, C, 1, 1]
            context = context.view(batch, channel, 1, 1)
        else:
            # [N, C, 1, 1]
            context = self.avg_pool(x)

        return context

    def forward(self, x):
        # [N, C, 1, 1]
        context = self.spatial_pool(x)
        out = context
        return out


class MS_CAM(nn.Module):
    def __init__(self, in_dim, reduction):
        super(MS_CAM, self).__init__()
        self.in_dim = in_dim
        self.local_range = nn.Sequential(
            Conv(in_dim, in_dim // reduction, k=1, s=1, p=0),
            # Conv(in_dim // reduction, in_dim // reduction, k=3, s=1, r=4, p=4),
            Conv(in_dim // reduction, in_dim, k=1, s=1, p=0, act=False),
        )

        #local
        # self.local_conv1 = Conv(in_dim, in_dim // reduction, k=1, s=1, p=0)
        # self.local_conv2 = Conv(in_dim // reduction, in_dim // reduction, k=3, s=1, p=1, act=False)
        # self.local_conv3 = Conv(in_dim // reduction, in_dim // reduction, k=3, s=1, p=1, act=False)
        # self.local_conv4 = Conv(in_dim // reduction, in_dim // reduction, k=3, s=1, p=1, act=False)
        # self.local_conv5 = Conv(in_dim // reduction, in_dim // reduction, k=3, s=1, p=1, act=False)#不需要激活函数,即恒等输出

        self.global_range = nn.Sequential(
            Simplified_NL(in_dim),
            # Conv(in_dim, in_dim // reduction, k=1, s=1, p=0),
            # Conv(in_dim // reduction, in_dim, k=1, s=1, p=0, act=False),
            nn.Conv2d(in_dim, in_dim // reduction, kernel_size=1, stride=1, padding=0),
            nn.GroupNorm(num_groups=4, num_channels=in_dim // reduction, eps=0, affine=False),#because every channel is 1*1，can't use BN
            nn.SiLU(),
            nn.Conv2d(in_dim // reduction, in_dim, kernel_size=1, stride=1, padding=0),
            nn.GroupNorm(num_groups=4, num_channels=in_dim, eps=0, affine=False),
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # local_out1 = self.local_conv1(x)
        # local_out21 = self.local_conv2(local_out1)
        # local_out22 = self.local_conv3(local_out21)
        # local_out23 = self.local_conv4(local_out22)
        # local_out24 = self.local_conv5(local_out23)
        # local_out = torch.cat([local_out21, local_out22, local_out23, local_out24], dim=1)

        local_out = self.local_range(x)
        global_out = self.global_range(x)


        local_global = self.sigmoid(local_out + global_out)
        x = x * local_global

        return x



class AFF_NL(nn.Module):
    def __init__(self, in_dim, reduction):
        super(AFF_NL, self).__init__()
        self.attention_fusion = MS_CAM(in_dim, reduction)

    def forward(self, x):

        low_level_feature = x[1]
        high_level_feature = x[0]
        attention_out = self.attention_fusion(low_level_feature + high_level_feature)
        low_attention = low_level_feature * attention_out
        high_attention = high_level_feature * (1 - attention_out)
        out = low_attention + high_attention
        # out = torch.cat((high_attention, low_attention), dim = 1)
        return out

class MS_CAM_backbone(nn.Module):
    def __init__(self, low_channel, high_channel, reduction):
        super(MS_CAM_backbone, self).__init__()
        self.low_channel = low_channel
        self.high_channel = high_channel

        self.local_range = nn.Sequential(
            Conv(self.low_channel, self.low_channel // reduction, k=1, s=1, p=0),
            # Conv(in_dim // reduction, in_dim // reduction, k=3, s=1, r=4, p=4),
            Conv(self.low_channel // reduction, self.low_channel, k=1, s=1, p=0, act=False),
        )

        self.global_range = nn.Sequential(
            Simplified_NL(self.low_channel),
            Conv(self.low_channel, self.low_channel // reduction, k=1, s=1, p=0),
            Conv(self.low_channel // reduction, self.low_channel, k=1, s=1, p=0, act=False),
        )
        self.downsample = nn.Sequential(
            Conv(self.low_channel, self.high_channel, k=1, p=0, s=1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        low_x = x[0]
        high_x = x[1]

        local_out = self.local_range(low_x)

        global_out = self.global_range(low_x)

        attention_weight = self.downsample(F.interpolate(local_out + global_out, size=None, scale_factor=1/2, mode='nearest'))
        out = high_x  * attention_weight

        return out
#Fused结构:backbone部分最后一层使用所有层的结果相加
class Fused(nn.Module):
    def __init__(self, c1, c2, c3, c4):
        super(Fused,self).__init__()
        self.conv2_5 = Conv(c1, c4, k=1, s=1, p=0)
        self.conv3_5 = Conv(c2, c4, k=1, s=1, p=0)
        self.conv4_5 = Conv(c3, c4, k=1, s=1, p=0)

    def forward(self, x):
        #x[0]-P2, x[1]-P3, x[2]-P4, x[3]-P5
        down_conv2_5 = self.conv2_5(F.interpolate(x[0], size=None, scale_factor=1/8, mode='nearest'))
        down_conv3_5 = self.conv3_5(F.interpolate(x[1], size=None, scale_factor=1/4, mode='nearest'))
        down_conv4_5 = self.conv4_5(F.interpolate(x[2], size=None, scale_factor=1/2, mode='nearest'))
        out = down_conv2_5 + down_conv3_5 + down_conv4_5 + x[3]
        return out
#------------
#---------------SML
import  torch.nn.init as init
class L2Norm(nn.Module):
    def __init__(self,n_channels, scale):
        super(L2Norm,self).__init__()
        self.n_channels = n_channels
        self.gamma = scale or None
        self.eps = 1e-10
        self.weight = nn.Parameter(torch.Tensor(self.n_channels))
        self.reset_parameters()

    def reset_parameters(self):
        init.constant_(self.weight, self.gamma)

    def forward(self, x):
        norm = x.pow(2).sum(dim=1, keepdim=True).sqrt() + self.eps
        x = torch.div(x,norm)
        out = self.weight.unsqueeze(0).unsqueeze(2).unsqueeze(3) * x
        return out

class SML(nn.Module):
    def __init__(self, c2, c3, c4, c5):
        super(SML, self).__init__()
        P2_c, P3_c, P4_c, P5_c = c2, c3, c4, c5
        #deconv for in and out:N_out = (N_in - 1) * s + k - 2*p
        #deconv instaed of upsample and conv
        self.deconv_P5 = nn.Sequential(
            Conv(P5_c, P2_c, k=1, s=1, p=0),
            nn.Upsample(size = None, scale_factor = 8, mode='nearest'),
        )#P5

        self.deconv_P4 = nn.Sequential(
            Conv(P4_c, P2_c, k=1, s=1, p=0),
            nn.Upsample(size=None, scale_factor=4, mode='nearest'),
        )#P4

        self.deconv_P3 = nn.Sequential(
            Conv(P3_c, P2_c, k=1, s=1, p=0),
            nn.Upsample(size=None, scale_factor=2, mode='nearest'),
        ) # P3

        self.conv3x3 = Conv(P2_c*4, 128, k=3, s=1, p=1)

        self.l2_P2 = L2Norm(P2_c, 20.)
        self.l2_P3 = L2Norm(P3_c, 20.)
        self.l2_P4 = L2Norm(P4_c, 20.)
        self.l2_P5 = L2Norm(P5_c, 20.)

    def forward(self, x):
        #L2 normalization
        x_P2 = self.l2_P2(x[0])
        x_P3 = self.l2_P3(x[1])
        x_P4 = self.l2_P4(x[2])
        x_P5 = self.l2_P5(x[3])
        #P3, P4 and P5 --> P2
        P3_deconv = self.deconv_P3(x_P3)
        P4_deconv = self.deconv_P4(x_P4)
        P5_deconv = self.deconv_P5(x_P5)

        P2_3_4_5 = torch.cat([x_P2, P3_deconv, P4_deconv, P5_deconv], dim=1)

        output = self.conv3x3(P2_3_4_5)
        return output

#---------------
#------RFB:
class BasicConv(nn.Module):

    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True, bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes,eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU(inplace=True) if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class RFB(nn.Module):
    def __init__(self, in_planes, out_planes, stride=1, scale = 0.1, visual = 1):
        super(RFB, self).__init__()
        self.scale = scale
        self.out_channels = out_planes
        inter_planes = in_planes // 8
        self.branch0 = nn.Sequential(
                BasicConv(in_planes, 2*inter_planes, kernel_size=1, stride=stride),
                BasicConv(2*inter_planes, 2*inter_planes, kernel_size=3, stride=1, padding=visual, dilation=visual, relu=False)
                )
        self.branch1 = nn.Sequential(
                BasicConv(in_planes, inter_planes, kernel_size=1, stride=1),
                BasicConv(inter_planes, 2*inter_planes, kernel_size=(3,3), stride=stride, padding=(1,1)),
                BasicConv(2*inter_planes, 2*inter_planes, kernel_size=3, stride=1, padding=visual+1, dilation=visual+1, relu=False)
                )
        self.branch2 = nn.Sequential(
                BasicConv(in_planes, inter_planes, kernel_size=1, stride=1),
                BasicConv(inter_planes, (inter_planes//2)*3, kernel_size=3, stride=1, padding=1),
                BasicConv((inter_planes//2)*3, 2*inter_planes, kernel_size=3, stride=stride, padding=1),
                BasicConv(2*inter_planes, 2*inter_planes, kernel_size=3, stride=1, padding=2*visual+1, dilation=2*visual+1, relu=False)
                )

        self.ConvLinear = BasicConv(6*inter_planes, out_planes, kernel_size=1, stride=1, relu=False)
        self.shortcut = BasicConv(in_planes, out_planes, kernel_size=1, stride=stride, relu=False)
        self.relu = nn.ReLU(inplace=False)

    def forward(self,x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)

        out = torch.cat((x0,x1,x2),1)
        out = self.ConvLinear(out)
        short = self.shortcut(x)
        out = out*self.scale + short
        out = self.relu(out)

        return out

#------
#----------ATT-part
class Identity_input(nn.Module):
    def __init__(self, par):
        super(Identity_input, self).__init__()

    def forward(self, x):
        return x

class ATT_part_att(nn.Module):
    def __init__(self, in_dim):
        super(ATT_part_att, self).__init__()
        from torchvision.ops import MultiScaleRoIAlign

        self.box_roi_pool = MultiScaleRoIAlign(
            featmap_names=['0'],
            output_size=[8, 8],
            sampling_ratio=2)
        self.in_dim = in_dim
        self.roi_conv = Conv(in_dim, 128, k=1, s=1, p=0)
        self.roi_att_fc1 = nn.Linear(128 * 8 * 8, 512)

    def forward(self, x):

        input = x[0]
        b, c, h_img, w_img = input.shape
        features = x[1]
        targets = x[2]

        if self.training and len(targets):
            roi_att_fc = []
            for index in range(len(features)):
                feature = features[index:index+1]
                target = targets[targets[:, 0] == index][:, 2:6]
                if len(target):
                    target_abs = target.clone()
                    # gt bbox --> (x1, y1, x2, y2)
                    w_abs, h_abs = target[:, 2:3] * w_img, target[:, 3:4] * h_img
                    target_abs[:, 0:1] = target[:, 0:1] * w_img - w_abs / 2  # x1
                    target_abs[:, 1:2] = target[:, 1:2] * h_img - h_abs / 2  # y1
                    target_abs[:, 2:3] = target[:, 0:1] * w_img + w_abs / 2  # x2
                    target_abs[:, 3:4] = target[:, 1:2] * h_img + h_abs / 2  # y2

                    feature_dict = {'0':feature}
                    box_feature = self.box_roi_pool(feature_dict, [target], [(h_img, w_img)])
                    roi_conv_out = self.roi_conv(box_feature)#(len_target, 128, 8, 8)
                    roi_ravel = self.roi_att_fc1(roi_conv_out.flatten(1))#(len_target, 512)
                    index_column = torch.ones((len(roi_ravel), 1), device= target.device) * index
                    roi_ravel = torch.cat((index_column, roi_ravel), dim=1)
                    roi_att_fc.append(roi_ravel)

            roi_att_fc = torch.cat(roi_att_fc, dim = 0)
        else:
            roi_att_fc = targets.clone()

        return roi_att_fc

class ATT_part_feature(nn.Module):
    def __init__(self, in_dim):
        super(ATT_part_feature, self).__init__()
        self.in_dim = in_dim

    def forward(self, x):
        #x --> feature and roi_att_fc
        roi_att_fc = x[0]
        features = x[1]
        features_out = features.clone()

        if self.training and len(roi_att_fc):
            for index in range(len(features)):
                feature = features[index:index+1]
                roi_att_fc_feature = roi_att_fc[roi_att_fc[:, 0] == index]
                if len(roi_att_fc_feature):
                    roi_att = roi_att_fc_feature[:, 1:]
                    # att_weights_mean = roi_att.mean(0).view(1, self.in_dim, 1, 1)
                    att_weights_max = roi_att.max(0)[0].view(1, self.in_dim, 1, 1)
                    # att_weights = att_weights_mean + att_weights_max

                    features_out[index:index+1] = feature * att_weights_max.expand_as(feature)


        return features_out

class ATT_part_loss(nn.Module):#occlusion-mode
    def __init__(self, in_dim):
        super(ATT_part_loss, self).__init__()
        self.in_dim = in_dim
        self.roi_fc = nn.Linear(in_dim, 4)

    def forward(self, x):
        #x and roi_att_fc(include index), features

        if self.training and len(x):
            index_column = x[:, 0:1]
            roi_att_fc = x[:, 1:]#occlusion mode class vector

            roi_mode = torch.cat((index_column, self.roi_fc(roi_att_fc)), dim=1)

        else:
            roi_mode = x.clone()
        return roi_mode

# class ATT_part(nn.Module):
#     def __init__(self, in_dim):
#         super(ATT_part, self).__init__()
#         from torchvision.ops import MultiScaleRoIAlign
#         # 构建roi align层和对应的conv和fc层
#         self.box_roi_pool = MultiScaleRoIAlign(
#             featmap_names=['0'],  # 在哪些特征层进行roi pooling
#             output_size=[8, 8],
#             sampling_ratio=2)
#         self.in_dim = in_dim
#         self.roi_conv = Conv(in_dim, 128, k=1, s=1, p=0)
#         self.roi_att_fc1 = nn.Linear(128*8*8, 512)
#         self.roi_fc2 = nn.Linear(512, 4)
#         #构建注意力部分的conv和fc层
#         # self.att_conv = Conv(in_dim, 128, k=1, s=1, p=0)
#         # self.att_global_pool = nn.AdaptiveMaxPool2d(1)
#         # self.att_fc2 = nn.Linear(128, in_dim)
#
#
#     def forward(self, x):
#         #roi align层输出
#         input = x[0]
#         b, c, h_img, w_img = input.shape
#         features = x[1]
#         targets = x[2]
#         features_out = features.clone()
#
#         if self.training:#训练时进行weights和roi features相关求解
#             if len(targets):#如果不含有gt bbox,则不需要进行注意力权重求解
#                 roi_fc_dict = {}  # 存储roi输出的字典
#                 for index in range(len(features)):#单张图片进行操作
#                     roi_fc_dict.setdefault(index, [])#设置默认初始化键对应值为空列表
#                     feature = features[index:index+1]
#                     target = targets[targets[:, 0] == index][:, 2:6]
#                     if len(target):
#                         target_abs = target.clone()
#                         # 将gt bbox转换为绝对的(x1, y1, x2, y2)
#                         w_abs, h_abs = target[:, 2:3] * w_img, target[:, 3:4] * h_img
#                         target_abs[:, 0:1] = target[:, 0:1] * w_img - w_abs / 2  # x1
#                         target_abs[:, 1:2] = target[:, 1:2] * h_img - h_abs / 2  # y1
#                         target_abs[:, 2:3] = target[:, 0:1] * w_img + w_abs / 2  # x2
#                         target_abs[:, 3:4] = target[:, 1:2] * h_img + h_abs / 2  # y2
#
#                         feature_dict = {'0':feature}
#                         box_feature = self.box_roi_pool(feature_dict, [target], [(h_img, w_img)])
#                         roi_conv_out = self.roi_conv(box_feature)#(len_target, 128, 8, 8)
#                         roi_ravel = self.roi_att_fc1(roi_conv_out.flatten(1))#(len_target, 512)
#
#                         att_weights_mean = roi_ravel.mean(0).view(1, self.in_dim, 1, 1)#对应index的图片的注意力权重
#                         att_weights_max = roi_ravel.max(0)[0].view(1, self.in_dim, 1, 1)
#                         att_weights = att_weights_mean + att_weights_max
#
#                         features_out[index:index+1] = feature_dict['0'] * att_weights.expand_as(feature_dict['0'])
#                         roi_out = self.roi_fc2(roi_ravel)
#                         roi_fc_dict[index] = roi_out
#
#             else:
#                 roi_fc_dict = torch.zeros((1, 1), device=input.device)
#         else:#测试时直接前向传播
#             features_out = features
#             roi_fc_dict = torch.zeros((1, 1), device=input.device)
#
#         out = [roi_fc_dict, features_out]
#
#         return out
#
# class ATT_part_out(nn.Module):
#     def __init__(self, out_flag, channel):
#         super(ATT_part_out, self).__init__()
#         self.channel = channel
#         self.out_flag = out_flag
#
#     def forward(self, x):
#         if self.out_flag == 1:#对应loss计算损失的输出
#             out = x[0]
#         elif self.out_flag == 2:#对应前向传播部分
#             out = x[1]
#         return out
#-----------------------
#---------------------add seg feature(backbone)
class ConcatSegConv(nn.Module):
    def __init__(self, par):
        super(ConcatSegConv, self).__init__()
        #downsample feature map
        self.max_pool1 = nn.MaxPool2d(2)
        self.max_pool2 = nn.MaxPool2d(2)
        self.max_pool3 = nn.MaxPool2d(2)

        # upsample feature map
        self.upsample2 = nn.Upsample(scale_factor=2, mode='nearest')
        self.upsample4 = nn.Upsample(scale_factor=4, mode='nearest')
        self.upsample8 = nn.Upsample(scale_factor=8, mode='nearest')


    def forward(self, x):
        #x:seg_input, seg_P1, seg_P2, seg_P3, conv_P3
        #get seg up
        conv_out = x[4]
        seg_P3_up = self.upsample8(x[3])#seg_P3*8
        seg_P2_up = self.upsample4(x[2])#seg_P2*4
        seg_P1_up = self.upsample2(x[1])#seg_P1*2
        seg_input = x[0]
        concat_seg = torch.cat([seg_input, seg_P1_up, seg_P2_up, seg_P3_up], dim=1)

        #get seg conv
        concat_seg_down = self.max_pool3(self.max_pool2(self.max_pool1(concat_seg)))
        concat_seg_conv = torch.cat([conv_out, concat_seg_down], dim=1)

        return concat_seg, concat_seg_conv


class Get_ConcatSeg(nn.Module):
    def __init__(self, input_channel):
        super(Get_ConcatSeg, self).__init__()
        self.c = input_channel
        # self.conv1x1 = Conv(self.c, 21, 1, 1)#get seg_result for 21 class
        # self.conv1x1 = Conv(self.c, 2, 1, 1)#get seg_result for 2 class
        self.conv1x1 = Conv(self.c, 1, 1, 1)#get edge_result
    def forward(self, x):
        # x:concat_seg, concat_seg_conv
        concat_seg_out = self.conv1x1(x[0])

        return concat_seg_out

class Get_ConcatConv(nn.Module):
    def __init__(self, par):
        super(Get_ConcatConv, self).__init__()


    def forward(self, x):
        # x:concat_seg, concat_seg_conv
        concat_seg_conv = x[1]

        return concat_seg_conv
#--------------------------------------add seg feature(head)
class ConcatSegP(nn.Module):
    def __init__(self, c1):
        super(ConcatSegP, self).__init__()
        self.conv1 = Conv(c1, 256, 3, 1, 1)
        self.conv2 = Conv(256, 128, 3, 1, 1)

    def forward(self, x):
        # x:P3 or P4 or P5(N,c1,h,w)
        conv1_out = self.conv1(x)#(N,256,h,w)
        conv2_out = self.conv2(conv1_out)#(N,128,h,w)
        concat_result = torch.cat([x, conv2_out], dim=1)

        return concat_result


class ConcatSegUp(nn.Module):
    def __init__(self, par):
        super(ConcatSegUp, self).__init__()
        # upsample feature map
        self.upsample8 = nn.Upsample(scale_factor=8, mode='nearest')
        self.upsample16 = nn.Upsample(scale_factor=16, mode='nearest')
        self.upsample32 = nn.Upsample(scale_factor=32, mode='nearest')
        self.conv1x1 = Conv(128 * 3, 21, 1, 1, 0)

    def forward(self, x):
        # x:seg_P3 and seg_P4 and seg_P5(N,128,h,w)
        seg_P3 = x[0][:, 256-128:256, :, :]#80*80->256
        seg_P4 = x[1][:, 512-128:512, :, :]#40*40->512
        seg_P5 = x[2][:, 1024-128:1024, :, :]#20*20->1024


        seg_P3_up = self.upsample8(seg_P3)#640*640
        seg_P4_up = self.upsample16(seg_P4)#640*640
        seg_P5_up = self.upsample32(seg_P5)#640*640

        seg_result = self.conv1x1(torch.cat([seg_P3_up, seg_P4_up, seg_P5_up], dim=1))#(N, 21, 640, 640)

        return seg_result
#--------------------------------------

#--------------------------------------
class AutoShape(nn.Module):
    # input-robust model wrapper for passing cv2/np/PIL/torch inputs. Includes preprocessing, inference and NMS
    conf = 0.25  # NMS confidence threshold
    iou = 0.45  # NMS IoU threshold
    classes = None  # (optional list) filter by class
    max_det = 1000  # maximum number of detections per image

    def __init__(self, model):
        super(AutoShape, self).__init__()
        self.model = model.eval()

    def autoshape(self):
        print('AutoShape already enabled, skipping... ')  # model already converted to model.autoshape()
        return self

    @torch.no_grad()
    def forward(self, imgs, size=640, augment=False, profile=False):
        # Inference from various sources. For height=640, width=1280, RGB images example inputs are:
        #   filename:   imgs = 'data/images/zidane.jpg'
        #   URI:             = 'https://github.com/ultralytics/yolov5/releases/download/v1.0/zidane.jpg'
        #   OpenCV:          = cv2.imread('image.jpg')[:,:,::-1]  # HWC BGR to RGB x(640,1280,3)
        #   PIL:             = Image.open('image.jpg')  # HWC x(640,1280,3)
        #   numpy:           = np.zeros((640,1280,3))  # HWC
        #   torch:           = torch.zeros(16,3,320,640)  # BCHW (scaled to size=640, 0-1 values)
        #   multiple:        = [Image.open('image1.jpg'), Image.open('image2.jpg'), ...]  # list of images

        t = [time_synchronized()]
        p = next(self.model.parameters())  # for device and type
        if isinstance(imgs, torch.Tensor):  # torch
            with amp.autocast(enabled=p.device.type != 'cpu'):
                return self.model(imgs.to(p.device).type_as(p), augment, profile)  # inference

        # Pre-process
        n, imgs = (len(imgs), imgs) if isinstance(imgs, list) else (1, [imgs])  # number of images, list of images
        shape0, shape1, files = [], [], []  # image and inference shapes, filenames
        for i, im in enumerate(imgs):
            f = f'image{i}'  # filename
            if isinstance(im, str):  # filename or uri
                im, f = np.asarray(Image.open(requests.get(im, stream=True).raw if im.startswith('http') else im)), im
            elif isinstance(im, Image.Image):  # PIL Image
                im, f = np.asarray(im), getattr(im, 'filename', f) or f
            files.append(Path(f).with_suffix('.jpg').name)
            if im.shape[0] < 5:  # image in CHW
                im = im.transpose((1, 2, 0))  # reverse dataloader .transpose(2, 0, 1)
            im = im[:, :, :3] if im.ndim == 3 else np.tile(im[:, :, None], 3)  # enforce 3ch input
            s = im.shape[:2]  # HWC
            shape0.append(s)  # image shape
            g = (size / max(s))  # gain
            shape1.append([y * g for y in s])
            imgs[i] = im if im.data.contiguous else np.ascontiguousarray(im)  # update
        shape1 = [make_divisible(x, int(self.stride.max())) for x in np.stack(shape1, 0).max(0)]  # inference shape
        x = [letterbox(im, new_shape=shape1, auto=False)[0] for im in imgs]  # pad
        x = np.stack(x, 0) if n > 1 else x[0][None]  # stack
        x = np.ascontiguousarray(x.transpose((0, 3, 1, 2)))  # BHWC to BCHW
        x = torch.from_numpy(x).to(p.device).type_as(p) / 255.  # uint8 to fp16/32
        t.append(time_synchronized())

        with amp.autocast(enabled=p.device.type != 'cpu'):
            # Inference
            y = self.model(x, augment, profile)[0]  # forward
            t.append(time_synchronized())

            # Post-process
            y = non_max_suppression(y, self.conf, iou_thres=self.iou, classes=self.classes, max_det=self.max_det)  # NMS
            for i in range(n):
                scale_coords(shape1, y[i][:, :4], shape0[i])

            t.append(time_synchronized())
            return Detections(imgs, y, files, t, self.names, x.shape)


class Detections:
    # detections class for YOLOv5 inference results
    def __init__(self, imgs, pred, files, times=None, names=None, shape=None):
        super(Detections, self).__init__()
        d = pred[0].device  # device
        gn = [torch.tensor([*[im.shape[i] for i in [1, 0, 1, 0]], 1., 1.], device=d) for im in imgs]  # normalizations
        self.imgs = imgs  # list of images as numpy arrays
        self.pred = pred  # list of tensors pred[0] = (xyxy, conf, cls)
        self.names = names  # class names
        self.files = files  # image filenames
        self.xyxy = pred  # xyxy pixels
        self.xywh = [xyxy2xywh(x) for x in pred]  # xywh pixels
        self.xyxyn = [x / g for x, g in zip(self.xyxy, gn)]  # xyxy normalized
        self.xywhn = [x / g for x, g in zip(self.xywh, gn)]  # xywh normalized
        self.n = len(self.pred)  # number of images (batch size)
        self.t = tuple((times[i + 1] - times[i]) * 1000 / self.n for i in range(3))  # timestamps (ms)
        self.s = shape  # inference BCHW shape

    def display(self, pprint=False, show=False, save=False, crop=False, render=False, save_dir=Path('')):
        for i, (im, pred) in enumerate(zip(self.imgs, self.pred)):
            str = f'image {i + 1}/{len(self.pred)}: {im.shape[0]}x{im.shape[1]} '
            if pred is not None:
                for c in pred[:, -1].unique():
                    n = (pred[:, -1] == c).sum()  # detections per class
                    str += f"{n} {self.names[int(c)]}{'s' * (n > 1)}, "  # add to string
                if show or save or render or crop:
                    for *box, conf, cls in pred:  # xyxy, confidence, class
                        label = f'{self.names[int(cls)]} {conf:.2f}'
                        if crop:
                            save_one_box(box, im, file=save_dir / 'crops' / self.names[int(cls)] / self.files[i])
                        else:  # all others
                            plot_one_box(box, im, label=label, color=colors(cls))

            im = Image.fromarray(im.astype(np.uint8)) if isinstance(im, np.ndarray) else im  # from np
            if pprint:
                print(str.rstrip(', '))
            if show:
                im.show(self.files[i])  # show
            if save:
                f = self.files[i]
                im.save(save_dir / f)  # save
                print(f"{'Saved' * (i == 0)} {f}", end=',' if i < self.n - 1 else f' to {save_dir}\n')
            if render:
                self.imgs[i] = np.asarray(im)

    def print(self):
        self.display(pprint=True)  # print results
        print(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {tuple(self.s)}' % self.t)

    def show(self):
        self.display(show=True)  # show results

    def save(self, save_dir='runs/hub/exp'):
        save_dir = increment_path(save_dir, exist_ok=save_dir != 'runs/hub/exp', mkdir=True)  # increment save_dir
        self.display(save=True, save_dir=save_dir)  # save results

    def crop(self, save_dir='runs/hub/exp'):
        save_dir = increment_path(save_dir, exist_ok=save_dir != 'runs/hub/exp', mkdir=True)  # increment save_dir
        self.display(crop=True, save_dir=save_dir)  # crop results
        print(f'Saved results to {save_dir}\n')

    def render(self):
        self.display(render=True)  # render results
        return self.imgs

    def pandas(self):
        # return detections as pandas DataFrames, i.e. print(results.pandas().xyxy[0])
        new = copy(self)  # return copy
        ca = 'xmin', 'ymin', 'xmax', 'ymax', 'confidence', 'class', 'name'  # xyxy columns
        cb = 'xcenter', 'ycenter', 'width', 'height', 'confidence', 'class', 'name'  # xywh columns
        for k, c in zip(['xyxy', 'xyxyn', 'xywh', 'xywhn'], [ca, ca, cb, cb]):
            a = [[x[:5] + [int(x[5]), self.names[int(x[5])]] for x in x.tolist()] for x in getattr(self, k)]  # update
            setattr(new, k, [pd.DataFrame(x, columns=c) for x in a])
        return new

    def tolist(self):
        # return a list of Detections objects, i.e. 'for result in results.tolist():'
        x = [Detections([self.imgs[i]], [self.pred[i]], self.names, self.s) for i in range(self.n)]
        for d in x:
            for k in ['imgs', 'pred', 'xyxy', 'xyxyn', 'xywh', 'xywhn']:
                setattr(d, k, getattr(d, k)[0])  # pop out of list
        return x

    def __len__(self):
        return self.n


class Classify(nn.Module):
    # Classification head, i.e. x(b,c1,20,20) to x(b,c2)
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1):  # ch_in, ch_out, kernel, stride, padding, groups
        super(Classify, self).__init__()
        self.aap = nn.AdaptiveAvgPool2d(1)  # to x(b,c1,1,1)
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g)  # to x(b,c2,1,1)
        self.flat = nn.Flatten()

    def forward(self, x):
        z = torch.cat([self.aap(y) for y in (x if isinstance(x, list) else [x])], 1)  # cat if list
        return self.flat(self.conv(z))  # flatten to x(b,c2)
