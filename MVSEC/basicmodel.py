#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 13 13:20:02 2023

@author: mehdi
"""


import torch.nn as nn
import torch
from ConvGRU import ConvGRU
from config import device



class conv_layer(nn.Module):
    def __init__(self,n_in_feature,n_filter,f_size=5,strides=1,padding=2,norm = None, activation= "relu"):
        super(conv_layer, self).__init__()
        
        self.norm_set = norm
        self.activation = activation
        
        
        self.conv0  = nn.Conv2d(n_in_feature, n_filter, ( f_size,f_size), strides, padding)
        
        if norm == "IN":
            self.norm  = nn.GroupNorm(4, n_filter)
        if norm == "BN":
            self.norm  = nn.BatchNorm2d(n_filter)


        if activation == "relu":
            self.activation_f  = nn.ReLU()
        if activation == "sigmoid":
            self.activation_f  = nn.Sigmoid()
    
    def forward(self,x):
        x = self.conv0(x)
        
        if self.norm_set in ["IN","BN"]:
            x = self.norm(x)
        
        if self.activation in ["relu","sigmoid"]:
            x = self.activation_f(x)

        return x
    
    


class res_block(nn.Module):
    def __init__(self, n_ch_input, n_ch_output):
        super(res_block, self).__init__()
        
        self.conv1 = conv_layer(n_ch_input,  n_ch_output,f_size=3,strides=1,padding=1)
        self.conv2 = conv_layer(n_ch_output, n_ch_output,f_size=3,strides=1,padding=1,activation=None)

    def forward(self, x):
        i = x
        
        x = self.conv1(x)
        x = self.conv2(x)
        x = x + i
        x = torch.relu(x)
        
        return  x
    




class Conv_decoder(nn.Module):
    def __init__(self,n_in_feature, n_out_feature):
        super(Conv_decoder, self).__init__()
        
        self.conv1  = conv_layer(n_in_feature, n_out_feature,f_size=5,strides=1,padding=2)

    def forward(self,x):
        
        x = self.conv1(x)

        return x


class final_layer(nn.Module):
    def __init__(self,n_in_feature, n_out_feature):
        super(final_layer, self).__init__()

        self.conv1  = conv_layer(n_in_feature, 1, f_size=1,strides=1,padding=0, activation="sigmoid")

    def forward(self,x):
        
        x = self.conv1(x)        

        return x




class Encoder_layer(nn.Module):
    def __init__(self,n_ch, filter_size):
        super(Encoder_layer, self).__init__()
        
        self.conv0     = conv_layer(n_ch, filter_size[0],f_size=5,strides=2,padding=2)
        
        self.conv1_1   = conv_layer(filter_size[0], filter_size[1],f_size=5,strides=1,padding=2)
        self.conv1_2   = conv_layer(filter_size[1], filter_size[1],f_size=3,strides=1,padding=1)
    
        self.conv2_1   = conv_layer(filter_size[1], filter_size[2],f_size=5,strides=2,padding=2)
        self.conv2_2   = conv_layer(filter_size[2], filter_size[2],f_size=3,strides=1,padding=1)
    
        self.conv3_1   = conv_layer(filter_size[2], filter_size[3],f_size=5,strides=2,padding=2)
        self.conv3_2   = conv_layer(filter_size[3], filter_size[3],f_size=3,strides=1,padding=1)
    
    
    def forward(self,x):
        
        shape = x.size()
        L = 1
        if len(shape) == 5:
            L = shape[1]
        
        out1 = list()
        out2 = list()
        out3 = list()
        
        for i in range(0,L):
            x0 = self.conv0(x[:,i,:,:,:])
            
            x1 = self.conv1_1(x0)
            x1 = self.conv1_2(x1)
            
            x2 = self.conv2_1(x1)
            x2 = self.conv2_2(x2)
            
            x3 = self.conv3_1(x2)
            x3 = self.conv3_2(x3)
            
            out1.append(x1)
            out2.append(x2)
            out3.append(x3)
        
        
        out1 = torch.stack(out1)
        out1 = out1.permute(1, 0, 2, 3, 4)
        
        out2 = torch.stack(out2)
        out2 = out2.permute(1, 0, 2, 3, 4)
        
        out3 = torch.stack(out3)
        out3 = out3.permute(1, 0, 2, 3, 4)
        
        return out1, out2, out3





class GRU_module(nn.Module):
    def __init__(self,filter_size):
        super(GRU_module, self).__init__()
                
        self.GRU1 = ConvGRU(filter_size[1], filter_size[1], 3)
        self.GRU2 = ConvGRU(filter_size[2], filter_size[2], 3)
        self.GRU3 = ConvGRU(filter_size[3], filter_size[3], 3)
        

    def forward(self,img1,img2,img3,event1,event2,event3,label1,label2,label3,pre,pre_features):
                
        out1 = torch.relu(self.GRU1(img1,event1,label1,pre_features[0]))
        out2 = torch.relu(self.GRU2(img2,event2,label2,pre_features[1]))
        out3 = torch.relu(self.GRU3(img3,event3,label3,pre_features[2]))
        

        return out1, out2, out3, [out1,out2,out3]



class Decoder_module(nn.Module):
    def __init__(self,filter_size):
        super(Decoder_module, self).__init__()
        
        self.GRU = GRU_module(filter_size)
        
        self.res_block3_1  = res_block(filter_size[3], filter_size[3])
        self.res_block3_2  = res_block(filter_size[3], filter_size[3])
        
        self.res_block2_1  = res_block(filter_size[2], filter_size[2])
        self.res_block2_2  = res_block(filter_size[2], filter_size[2])
        
        self.res_block1_1  = res_block(filter_size[1], filter_size[1])
        self.res_block1_2  = res_block(filter_size[1], filter_size[1])
        
        #======= up sample part ======================================
        self.upsample1  = nn.Upsample(scale_factor=2, mode='bilinear')
        self.upsample2  = nn.Upsample(scale_factor=2, mode='bilinear')
        self.upsample3  = nn.Upsample(scale_factor=2, mode='bilinear')

        #==== conv in decoder =======================
        self.convd3 = Conv_decoder(filter_size[3],  filter_size[2])
        self.convd2 = Conv_decoder(filter_size[2] + filter_size[2], filter_size[1])
        self.convd1 = Conv_decoder(filter_size[1] + filter_size[1], filter_size[0])

        self.final_layer = final_layer(filter_size[0], 1)


    def forward(self,img1,img2,img3,event1,event2,event3,label1,label2,label3,label,pre,pre_features):
        
        out = list()
        B,L,C,H,W = img1.size()
        for i in range(0,L):
            
            x1, x2, x3, pre_features = self.GRU(img1[:,i,:,:,:],img2[:,i,:,:,:],img3[:,i,:,:,:],event1[:,i,:,:,:],event2[:,i,:,:,:],event3[:,i,:,:,:],label1[:,i,:,:,:],label2[:,i,:,:,:],label3[:,i,:,:,:],pre,pre_features)
            
            #===== Res blocks ==========
            x3  = self.res_block3_1(x3)
            x3  = self.res_block3_2(x3)
            
            x2  = self.res_block2_1(x2)
            x2  = self.res_block2_2(x2)
            
            x1  = self.res_block1_1(x1)
            x1  = self.res_block1_2(x1)
            
            #===== first upsample block ==========
            x3  = self.convd3(x3)
            x3  = self.upsample3(x3)
            
            #===== second upsample block====
            x2  = torch.cat((x2,x3),1)
            x2  = self.convd2(x2)
            x2  = self.upsample2(x2)
            
            #=== third upsample block ===========
            x1  = torch.cat((x1,x2),1)
            x1  = self.convd1(x1)
            x1  = self.upsample1(x1)
            
            #====== finall block =================
            pre = self.final_layer(x1)
            out.append(pre)

        out = torch.stack(out)
        out = out.permute(1, 0, 2, 3, 4)
        
        return out, pre_features





