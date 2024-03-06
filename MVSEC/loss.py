#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Sun Apr 23 12:32:56 2023

@author: mehdi
"""


import torch 
from kornia.filters.sobel import spatial_gradient, sobel



def scale_invariant_loss(y_input, y_targets, weight = 1.0, n_lambda = 1.0):
    
    log_diff = y_input - y_targets
    is_nan = torch.isnan(log_diff)    
    
    return weight * ((log_diff[~is_nan]**2).mean()-(n_lambda*(log_diff[~is_nan].mean())**2))


def scale_gra_loss(inputs, target, target_h, scale_loss=4):
    
    B, L, C, H, W = inputs.size()
    
    targets = target.clone()

    is_nan = torch.isnan(targets)
    targets[is_nan] = target_h[is_nan]

    x1 = targets[:,:,:,0:int(3*H/4),:]
    x2 = targets[:,:,:,int(3*H/4): ,:]
    
    x1 = torch.nan_to_num(x1, nan=1.0)
    x2 = torch.nan_to_num(x2, nan=0.0)

    targets[:,:,:,0:int(3*H/4),:] = x1 
    targets[:,:,:,int(3*H/4): ,:] = x2
    
    dff = inputs - targets
    error = 0
    
    for b in range(0,B):
        for s in range(0,scale_loss):
            m = torch.nn.AvgPool2d(2**s,2**s)
            diff = m(dff[b])
            edg = spatial_gradient(diff)
            is_not_nan = ~torch.isnan(edg)
            edg  = torch.abs(edg)
            temp = edg[is_not_nan].mean()
            if torch.isnan(temp):
              pass
            else:
              error += 2*temp
            
    error = error / (B*scale_loss)
    return error



def scale_gra_loss_nan(inputs, targets,scale_loss=4):
    
    B, L, C, H, W = inputs.size()    
    dff = inputs - targets
    error = 0
    for b in range(0,B):
        for s in range(0,scale_loss):
            m = torch.nn.AvgPool2d(2**s,2**s)
            diff = m(dff[b])
            edg = spatial_gradient(diff)
            is_not_nan = ~torch.isnan(edg)
            edg  = torch.abs(edg)
            temp = edg[is_not_nan].mean()
            if torch.isnan(temp):
              pass
            else:
              error += 2*temp
            
    error = error / (B*scale_loss)
    return error




            
def edge_loss_time(inputs, targets, scale_loss=4):
    
    B, L, C, H, W = inputs.size()
    error = 0
    for b in range(0,B):
        for s in range(0,scale_loss):
            m = torch.nn.AvgPool2d(2**s,2**s)
            
            inputs_s  = m(inputs[b])
            targets_s = m(targets[b])

            edg_in  = spatial_gradient(inputs_s)
            edg_tar = spatial_gradient(targets_s)

            dff = edg_in - edg_tar
            dff = torch.abs(dff)
            e = dff.mean()
            if torch.isnan(e):
              pass
            else:
              error += 2*e            
            
    error = error / (B*scale_loss)
    
    return error
         

def loss_depth(inputs, targets, targets_h, lambda_scale = 0.05, lambda_scale_nan = 0.0, movement_scale = 0.0, scale_loss=4):
    
    B,L,C,H,W = inputs.size()
    error = 0
    for i in range(0,B):
        for j in range(0,L):
            error += scale_invariant_loss(inputs[i:i+1,j:j+1,:,:,:], targets[i:i+1,j:j+1,:,:,:])
            error += lambda_scale_nan * scale_gra_loss_nan(inputs[i:i+1,j:j+1,:,:,:], targets[i:i+1,j:j+1,:,:,:])   
            error += lambda_scale * scale_gra_loss(inputs[i:i+1,j:j+1,:,:,:], targets[i:i+1,j:j+1,:,:,:], targets_h[i:i+1,j:j+1,:,:,:])
            
            
            #==== movement error ==========
            if j>1:
              inputs_edge  = torch.nan_to_num(inputs[i:i+1,j+1:j+2,:,:,:])  - torch.nan_to_num(inputs[i:i+1,j:j+1,:,:,:])
              targets_edge = torch.nan_to_num(targets[i:i+1,j+1:j+2,:,:,:]) - torch.nan_to_num(targets[i:i+1,j:j+1,:,:,:])
              movement_error = scale_gra_loss_nan(inputs_edge, targets_edge)
              error += movement_scale * movement_error
            
    
    
    
    return error / (B*L)



'''
x = torch.rand((2,8,1,32,32), dtype=torch.float32)
y = torch.rand((2,8,1,32,32), dtype=torch.float32)
       
l = loss_depth(x,y)  
'''

