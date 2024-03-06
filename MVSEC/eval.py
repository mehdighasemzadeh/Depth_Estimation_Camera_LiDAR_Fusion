#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 17 12:00:09 2023

@author: mehdi
"""


import torch
#from torch._C import S
from torch.optim import lr_scheduler
#====== import data loader ======
from datagen import train_loader , test_loader , images_size , n_seq , batch_size_train, batch_size_test
#======= import model ===========
from model import Conv_GRU
from torch import nn
from loss import loss_depth
from metric import mse_metric
import cv2
import numpy as np


from config import LINE_CLEAR, device , n_img_ch, n_eve_ch , scale_loss , filter_size , best_model_path , model_save_path,load_carla_weight
from config import batch_size_test, batch_size_train , images_size , n_seq , min_d , max_d, n_seq_test, alpha



#====== load best model =========
model = Conv_GRU(n_img_ch, n_eve_ch, filter_size, images_size).to(device)

learning_rate = 0.0005
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
lambda1 = lambda epoch: 0.95 ** epoch
scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda1)


checkpoint = torch.load(best_model_path)
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
epoch_s = checkpoint['epoch']
print(epoch_s)
loss = checkpoint['loss']
model.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=optimizer.param_groups[0]['lr'])
lambda1 = lambda epoch: 0.95 ** epoch
scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda1)

len_test = len(test_loader)

'''
with torch.no_grad():
    total_loss_view = 0 
    rmse1 = 0
    rmse2 = 0
    rmse3 = 0
    rmsea = 0
    for i, (images, events, labels) in enumerate(test_loader):
        h1 = None
        h2 = None
        h3 = None
        h1e = None
        h2e = None
        h3e = None


        for s in range(0,n_seq_test):
          images_s = images[:,s:s+1,:,:,:].to(device)
          events_s = events[:,s:s+1,:,:,:].to(device)
          labels_s = labels[:,s:s+1,:,:,:].to(device)

          #B,L,C,H,W = images_s.size()
          #images_s = torch.zeros((B,L,C,H,W),dtype=torch.float32).to(device)

          outputs,h1,h2,h3,h1e,h2e,h3e = model(images_s,events_s,h1,h2,h3,h1e,h2e,h3e)
              
          loss = loss_depth(outputs, labels_s)
          rmse1_t , rmse2_t , rmse3_t , rmsea_t = mse_metric(outputs, labels_s, min_d, max_d)
          
          rmse1 += rmse1_t
          rmse2 += rmse2_t
          rmse3 += rmse3_t
          rmsea += rmsea_t

          total_loss_view += loss.item()
              
          print(end=LINE_CLEAR)
          print (f'\r Results on Test part, Step [{i*n_seq_test+s+1}/{len_test*n_seq_test}], Loss: {total_loss_view/(i*n_seq_test+s+1):.4f} , Error1: {rmse1/(i*n_seq_test+s+1):.4f} , Error2: {rmse2/(i*n_seq_test+s+1):.4f} , Error3: {rmse3/(i*n_seq_test+s+1):.4f} , ErrorAll: {rmsea/(i*n_seq_test+s+1):.4f} ', end='')
    
    print()

'''


'''
with torch.no_grad():
    total_loss_view = 0 
    rmse1 = 0
    rmse2 = 0
    rmse3 = 0
    rmsea = 0
    pre = None
    pre_feature = [None,None,None,None,None,None]
    for i, (images, events, labels, label_in, labels_h) in enumerate(test_loader):
        
        for s in range(0,n_seq_test):
          images_s = images[:,s:s+1,:,:,:].to(device)
          events_s = events[:,s:s+1,:,:,:].to(device)
          labels_s = labels[:,s:s+1,:,:,:].to(device)
          label_in_s = label_in[:,s:s+1,:,:,:].to(device)
          #label_in_s = torch.nan_to_num(labels_s)
          
          #B,L,C,H,W = images_s.size()
          #images_s = torch.zeros((B,L,C,H,W),dtype=torch.float32).to(device)
          
          outputs, pre_feature = model(images_s,events_s,label_in_s,pre,pre_feature)
          pre = outputs  
          loss = loss_depth(outputs, labels_s, labels_s)
          rmse1_t , rmse2_t , rmse3_t , rmsea_t = mse_metric(outputs, labels_s, min_d, max_d)
          
          rmse1 += rmse1_t
          rmse2 += rmse2_t
          rmse3 += rmse3_t
          rmsea += rmsea_t

          total_loss_view += loss.item()
              
          print(end=LINE_CLEAR)
          print (f'\r Results on Test part, Step [{i*n_seq_test+s+1}/{len_test*n_seq_test}], Loss: {total_loss_view/(i*n_seq_test+s+1):.4f} , Error1: {rmse1/(i*n_seq_test+s+1):.4f} , Error2: {rmse2/(i*n_seq_test+s+1):.4f} , Error3: {rmse3/(i*n_seq_test+s+1):.4f} , ErrorAll: {rmsea/(i*n_seq_test+s+1):.4f} ', end='')

          
          im = (images_s[0][0][0].to('cpu').numpy())
          np.save('/content/sampletoshow/image.npy', im)
          
          im = (events_s[0][0][4].to('cpu').numpy())
          np.save('/content/sampletoshow/events.npy', im)
          
          im = (label_in_s[0][0][0].to('cpu').numpy())
          np.save('/content/sampletoshow/input_labels.npy', im)

          im = (labels_s[0][0][0].to('cpu').numpy())
          np.save('/content/sampletoshow/label.npy', im)

          im = (outputs[0][0][0].to('cpu').numpy())
          np.save('/content/sampletoshow/output.npy', im)
        #if i > 1:
          #break

'''




#========= save for the video ==========
with torch.no_grad():
    total_loss_view = 0 
    rmse1 = 0
    rmse2 = 0
    rmse3 = 0
    rmsea = 0
    pre = None
    pre_feature = [None,None,None,None,None,None]
    for i, (images, events, labels, label_in, labels_h) in enumerate(test_loader):
        
        for s in range(0,n_seq_test):
          images_s = images[:,s:s+1,:,:,:].to(device)
          events_s = events[:,s:s+1,:,:,:].to(device)
          labels_s = labels[:,s:s+1,:,:,:].to(device)
          label_in_s = label_in[:,s:s+1,:,:,:].to(device)
          #label_in_s = torch.nan_to_num(labels_s)


          
          #B,L,C,H,W = events_s.size()
          #events_s = torch.zeros((B,L,C,H,W),dtype=torch.float32).to(device)
          
          
          outputs, pre_feature = model(images_s,events_s,label_in_s,pre,pre_feature)
          pre = outputs  
          loss = loss_depth(outputs, labels_s, labels_s)
          rmse1_t , rmse2_t , rmse3_t , rmsea_t = mse_metric(outputs, labels_s, min_d, max_d)
          
          rmse1 += rmse1_t
          rmse2 += rmse2_t
          rmse3 += rmse3_t
          rmsea += rmsea_t

          total_loss_view += loss.item()
              
          print(end=LINE_CLEAR)
          print (f'\r Results on Test part, Step [{i*n_seq_test+s+1}/{len_test*n_seq_test}], Loss: {total_loss_view/(i*n_seq_test+s+1):.4f} , Error1: {rmse1/(i*n_seq_test+s+1):.4f} , Error2: {rmse2/(i*n_seq_test+s+1):.4f} , Error3: {rmse3/(i*n_seq_test+s+1):.4f} , ErrorAll: {rmsea/(i*n_seq_test+s+1):.4f} ', end='')

          im = (images_s[0][0][0].to('cpu').numpy())
          np.save("/content/night03/img/" + str( n_seq_test*i +s).zfill(5), im)
          
          im = (events_s[0][0][0].to('cpu').numpy())
          np.save("/content/night03/event/" + str( n_seq_test*i +s).zfill(5), im)

          im = (labels_s[0][0][0].to('cpu').numpy())
          np.save("/content/night03/label/" + str( n_seq_test*i +s).zfill(5), im)

          im = (label_in_s[0][0][0].to('cpu').numpy())
          np.save("/content/night03/label_in/" + str( n_seq_test*i +s).zfill(5), im)

          im = (outputs[0][0][0].to('cpu').numpy())
          np.save("/content/night03/output/" + str( n_seq_test*i +s).zfill(5), im)











    