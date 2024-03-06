#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 12 17:28:19 2023

@author: mehdi
"""




#====== in this model, we uesed complex encoder and passed feature from encoder to decoder, compare ro model 16
import torch.nn as nn
import torch
from basicmodel import Decoder_module
from config import device
from basicmodel import Encoder_layer

filter_size  = [64,64,128,256,512]


class Conv_GRU(nn.Module):
    def __init__(self,n_ch_img, n_ch_eve, filter_size, img_size):
        super(Conv_GRU, self).__init__()
        
        self.n_ch_eve = n_ch_eve
        self.n_ch_img = n_ch_img
        
        self.encoder_img = Encoder_layer(n_ch_img, filter_size)
        self.encoder_eve = Encoder_layer(n_ch_eve, filter_size)
        self.encoder_lab = Encoder_layer(1,        filter_size)

        self.decoder_module = Decoder_module(filter_size)


    def forward(self,img,event,label,pre,pre_feature):
        
        ####### encoder part ########
        batch_size, timesteps, C, H, W = img.size()
        if pre == None:
            pass
        else:
            pre = pre.reshape(1,1,H,W)

        i1, i2, i3 = self.encoder_img(img)
        e1, e2, e3 = self.encoder_eve(event)
        l1, l2, l3 = self.encoder_lab(label)
        
        out , pre_feature = self.decoder_module(i1,i2,i3,e1,e2,e3,l1,l2,l3,label,pre,pre_feature)
        
        return out, pre_feature
        
              

'''
start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)
pre_feature = [None,None,None]
pre = None
model = Conv_GRU(1,5, filter_size, (32,32)).to(device)
e = torch.rand((4,10,5,32,32) , dtype=torch.float32).to(device)
i = torch.rand((4,10,1,32,32) , dtype=torch.float32).to(device)
l = torch.rand((4,10,1,32,32) , dtype=torch.float32).to(device)

start.record()
y = model(i,e,l,pre,pre_feature)
end.record()
torch.cuda.synchronize()
print(start.elapsed_time(end))


pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(pytorch_total_params)
'''






