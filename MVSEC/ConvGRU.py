#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 23 00:23:06 2023

@author: mehdi
"""


import torch
import torch.nn as nn
from torch.nn import init
from config import device




class ConvGRU(nn.Module):
    """
    Generate a convolutional GRU cell
    Adapted from: https://github.com/jacobkimmel/pytorch_convgru/blob/master/convgru.py
    """

    def __init__(self, input_size, hidden_size, kernel_size):
        super().__init__()
        padding = kernel_size // 2
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        self.reset_gate       = nn.Conv2d(input_size + input_size +  input_size + hidden_size, hidden_size, kernel_size, padding=padding)
        self.update_gate      = nn.Conv2d(input_size + input_size +  input_size + hidden_size, hidden_size, kernel_size, padding=padding)
        self.out_gate_img     = nn.Conv2d(input_size + hidden_size,  hidden_size, kernel_size, padding=padding)
        self.out_gate_event   = nn.Conv2d(input_size + hidden_size,  hidden_size, kernel_size, padding=padding)
        self.out_gate_label   = nn.Conv2d(input_size + hidden_size,  hidden_size, kernel_size, padding=padding)


        init.orthogonal_(self.reset_gate.weight)
        init.orthogonal_(self.update_gate.weight)
        init.orthogonal_(self.out_gate_img.weight)
        init.orthogonal_(self.out_gate_event.weight)
        init.orthogonal_(self.out_gate_label.weight)

        
        init.constant_(self.reset_gate.bias, 0.)
        init.constant_(self.update_gate.bias, 0.)
        init.constant_(self.out_gate_img.bias, 0.)
        init.constant_(self.out_gate_event.bias, 0.)
        init.constant_(self.out_gate_label.bias, 0.)



    def forward(self, img, event, label, prev_state):
                
        # get batch and spatial sizes
        batch_size = img.data.size()[0]
        spatial_size = img.data.size()[2:]

        # generate empty prev_state_local and prev_state_local, if None is provided
        if prev_state is None:
            state_size = [batch_size, self.hidden_size] + list(spatial_size)
            prev_state = torch.zeros(state_size, dtype=img.dtype).to(img.device)

        
        # data size is [batch, channel, height, width]        
        stacked_inputs = torch.cat([img, event, label, prev_state], dim=1)
        
        update = self.update_gate(stacked_inputs)
        update = torch.sigmoid(update)
        
        reset_gate = self.reset_gate(stacked_inputs)
        reset_gate = torch.sigmoid(reset_gate)

        out_inputs_img = self.out_gate_img(torch.cat([img,     prev_state * reset_gate], dim=1))        
        out_gate_event = self.out_gate_event(torch.cat([event, prev_state * reset_gate], dim=1))
        out_gate_label = self.out_gate_label(torch.cat([label, prev_state * reset_gate], dim=1))

        out_gate = torch.tanh(out_inputs_img + out_gate_event + out_gate_label)
        

        new_state = ((1 - update) * prev_state) + (out_gate * update)

        return torch.relu(new_state)
    
    
    
    
    





