#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 17 01:34:50 2023

@author: mehdi
"""


import torch
from torch.optim import lr_scheduler
#====== import data loader ======
from datagen import train_loader , test_loader , images_size , n_seq , batch_size_train, batch_size_test
#======= import model ===========
from model import Conv_GRU
from torch import nn
from loss import loss_depth
from metric import mse_metric
from config import LINE_CLEAR, device , n_img_ch , n_eve_ch ,  scale_loss , filter_size , best_model_path , model_save_path, load_carla_weight
from config import batch_size_test, batch_size_train , images_size , n_seq , min_d , max_d, n_iter, n_sub_seq, n_seq_test, alpha
import math

#====== import model =========
model = Conv_GRU(n_img_ch, n_eve_ch, filter_size, images_size).to(device)

#======== set hyperparameter ===============
num_epochs = 200
learning_rate = 0.0003
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
lambda1 = lambda epoch: 0.98 ** epoch
scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda1)
 

total_loss_view = 0
# Train the model
n_total_steps = len(train_loader)
len_test = len(test_loader)
epoch_s = 0
min_error = 1000


#load carla model
checkpoint = torch.load(load_carla_weight)
model.load_state_dict(checkpoint['model_state_dict'])
model.to(device)



#load main model
checkpoint = torch.load(model_save_path)
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
epoch_s = checkpoint['epoch']
loss = checkpoint['loss']
model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=optimizer.param_groups[0]['lr'])
lambda1 = lambda epoch: 0.98 ** epoch
scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda1)


'''
learning_rate = 0.0005
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
lambda1 = lambda epoch: 0.98 ** epoch
scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda1)
'''

print(optimizer.param_groups[0]['lr'])


start_epoch = epoch_s
for epoch in range(start_epoch , num_epochs):

    total_loss_view = 0
    rmse1 = 0
    rmse2 = 0
    rmse3 = 0
    rmsea = 0

    for i, (images, events, labels, label_in, labels_h) in enumerate(train_loader):

        pre = None
        pre_feature = [None,None,None,None,None,None]

        optimizer.zero_grad()

        # Forward pass
        images = images.to(device)
        events = events.to(device)
        labels = labels.to(device)
        label_in = label_in.to(device)
        labels_h = labels_h.to(device)


        outputs, pre_feature = model(images,events,label_in,pre,pre_feature)
        loss = loss_depth(outputs, labels, labels_h)
        if math.isnan(loss):
          pass
        else:
          # Backward and optimize
          loss.backward()
          optimizer.step()
          
          with torch.no_grad():
            rmse1_t , rmse2_t , rmse3_t , rmsea_t= mse_metric(outputs, labels, min_d, max_d)
          
            rmse1 += rmse1_t
            rmse2 += rmse2_t
            rmse3 += rmse3_t
            rmsea += rmsea_t
            total_loss_view += loss.item()
            print(end=LINE_CLEAR)
            print (f'\r Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{n_total_steps}], Loss: {total_loss_view/(i+1):.4f} , Error1: {rmse1/(i+1):.4f} , Error2: {rmse2/(i+1):.4f} , Error3 {rmse3/(i+1):.4f} , ErrorAll: {rmsea/(i+1):.4f} ', end='')
          
        
        #======= save model after 200 iterations ====
        if i%200==0:    
          torch.save({
                  'epoch': epoch,
                  'model_state_dict': model.state_dict(),
                  'optimizer_state_dict': optimizer.state_dict(),
                  'loss': loss
                  }, model_save_path)
    
    
    torch.save({
                  'epoch': epoch,
                  'model_state_dict': model.state_dict(),
                  'optimizer_state_dict': optimizer.state_dict(),
                  'loss': loss
                  }, model_save_path)
    
    print()    
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

            
                outputs, pre_feature = model(images_s,events_s,label_in_s,pre,pre_feature)
                pre = outputs
            
                loss = loss_depth(outputs, labels_s, labels_s)
                if math.isnan(loss):
                  pass
                else:
                  rmse1_t , rmse2_t , rmse3_t , rmsea_t = mse_metric(outputs, labels_s, min_d, max_d)
        
                  rmse1 += rmse1_t
                  rmse2 += rmse2_t
                  rmse3 += rmse3_t
                  rmsea += rmsea_t

                  total_loss_view += loss.item()
            
                  print(end=LINE_CLEAR)
                  print (f'\r Results on Test part, Step [{i*n_seq_test+s+1}/{len_test*n_seq_test}], Loss: {total_loss_view/(i*n_seq_test+s+1):.4f} , Error1: {rmse1/(i*n_seq_test+s+1):.4f} , Error2: {rmse2/(i*n_seq_test+s+1):.4f} , Error3: {rmse3/(i*n_seq_test+s+1):.4f} , ErrorAll: {rmsea/(i*n_seq_test+s+1):.4f} ', end='')
    print()
    rmse = ((rmse1 + rmse2 + rmse3 ) / (i*n_seq_test+s+1)).to('cpu').item()
    if rmse < min_error:
        print("save best model")
        min_error = rmse
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': rmse
            }, best_model_path)
    
    scheduler.step()
      