#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 29 20:36:39 2023

@author: mehdi
"""

import torch
LINE_CLEAR = '\x1b[2K'

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

batch_size_train = 4
batch_size_test  = 1


images_size = (256,192)
test_size   = (344,200)
n_seq = 10
n_seq_test = 10
n_sub_seq = 10
n_iter = int(n_seq/n_sub_seq)

#-- max depth of the train dataset: 152.43
#-- max depth of day1: 132.06

min_d = 2.0
alpha = 3.7
max_d = 160.0

n_img_ch = 1
n_eve_ch = 5
scale_loss = 4
filter_size  = [32,64,128,256,512]

model_save_path = "CNN_GRU.pt"
load_carla_weight = "/content/gdrive/Shareddrives/shared_drive2/Combine_Lidar_camera/Model1/CNN_GRU.pt"
best_model_path = "bestmodel6.pt"




