#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 27 11:56:47 2023

@author: mehdi
"""

import torch
# Device configuration
from config import device

cuttoff = [10,20,30,80]

def mse_metric(input_log, target_log, min_s, max_s, a=3.7):
    
    min_s = torch.tensor([min_s,]).to(device)
    max_s = torch.tensor([max_s,]).to(device)
    
    input_log  = max_s * torch.exp(a*(input_log-1))
    target_log = max_s * torch.exp(a*(target_log-1))

    diff = target_log - input_log
    diff = torch.nan_to_num(diff)

    #=== calculate mse ===============
    input_log = input_log.reshape(-1)
    target_log = target_log.reshape(-1)
    diff = diff.reshape(-1)

    shape = input_log.size()
    
    error = list()
    for i in range(0,len(cuttoff)):
        k1 = torch.zeros(shape, dtype=torch.float32)
        k1_0  = torch.where(torch.isnan(target_log) , 0.0 , 1.0 )
        k1_1  = torch.where(target_log < cuttoff[i] , 1.0 , 0.0 )
        k1_2  = torch.where(target_log > min_s      , 1.0 , 0.0 )
        k1 = k1_0 * k1_1 * k1_2
        
        if k1.sum() > 0:
            k_e =  ((torch.abs(diff) * k1 ).sum())  /  k1.sum()
        else: 
            k_e =0
            
        error.append(k_e)
            
    
    return error[0] , error[1] , error[2] , error[3]


'''
import numpy as np
path = "/home/mehdi/Downloads/"
name = "depth_0000000000.npy"

t = np.load(path + name) 

t = torch.from_numpy(t).to(device)
t1 = torch.zeros((260,346), dtype=torch.float32).to(device)
test = mse_metric(t1, t, 2, 80)
print(test)
'''

