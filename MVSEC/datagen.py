#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 16 21:45:25 2023

@author: mehdi
"""


import torch
import torchvision.transforms as transforms
import numpy as np
import os
import cv2
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import torchvision
import torchvision.transforms.functional as fn
from sklearn.model_selection import train_test_split
import random
from config import batch_size_test, batch_size_train , images_size , n_seq, n_seq_test , min_d , max_d , test_size, alpha


def depth_creator(labels,min_d,max_d, a=3.7):
    
    max_depth = torch.tensor([max_d,])
    labels = (1/a) * torch.log(labels/max_depth) + 1.0
    labels = torch.where(labels < 0.0, 0.0, labels)
    labels = torch.where(labels > 1.0, 1.0, labels)
    return labels



def random_crop(image,x,y,img_size,interpolation):
    temp_img = image[x:x+img_size[1], y:y+img_size[0],:]  
    return temp_img 

def random_crop_resize(image,x,y,img_size,interpolation):
    temp_img = image[x:,y:,:]
    re_img = cv2.resize(temp_img, img_size, interpolation = interpolation)
    return re_img 

def vertical_shift(x,h):
    H,W,C = x.shape
    out = np.zeros((H,W,C), np.float32)
    temp1 = x[0:h,:,:]
    temp2 = x[h:,:,:]
    out[:H-h,:,:] = temp2
    out[H-h:,:,:] = temp1
    return out



def add_noise(label):
    label = torch.nan_to_num(label)
    
    L, C, H, W = label.size()
    
    '''
    #==== add gussian noise ===========
    b = torch.rand(label.size()) ** 3
    bernoulli  = torch.bernoulli(b)
    
    random_num = torch.rand(label.size()) ** 4
    sign = torch.rand(label.size())
    sign = torch.where(sign > 0.5, 1.0, -1.0)
    noise = bernoulli * random_num * sign
    label = label + noise
    '''
    
    #==== add noise ====
    step_x = int(H/4) 
    step_y = int(W/16) 
    for i in range(0,step_x):
        for j in range(0,step_y):
            for l in range(0,L):
                random_state = torch.rand(1).item()
                if random_state > 0.5:         
                    label[l:l+1,:,i*4:i*4+2,j*16:j*16+16] = 0
                else:
                    label[l:l+1,:,i*4+2:i*4+4,j*16:j*16+16] = 0
    '''
    #==== convert to zeros top and down of Lidar ====
    label[:,:,0:int(H/4),:]  = 0
    label[:,:,int(3*H/4):,:] = 0
    '''
    '''
    #=== convert to zeros ==============
    b = torch.rand(label.size()) ** 2
    bernoulli  = torch.bernoulli(b)
    label = label * (1-bernoulli)
    '''
    
    label = torch.where(label > 1.0, 1.0, label)
    label = torch.where(label < 0.0, 0.0, label)
    

    return label







class Dataloader_seq(Dataset):
    def __init__(self, images_dir, events_dir, labels_dir, enhanced_depth_dir, transform=None):
        
        self.images_dir = images_dir
        self.events_dir = events_dir
        self.labels_dir = labels_dir
        self.enhanced_depth_dir = enhanced_depth_dir
        self.transform  = transform

    def __len__(self):
        return len(self.images_dir)

    def __getitem__(self, idx):
        
        #====== load images ==============
        images = list()
        dir_images = self.images_dir[idx]
        for i in range(0,len(dir_images)):
            image = cv2.imread(dir_images[i] , 0)
            rgb_img = np.zeros((200,346,1),np.uint8)
            rgb_img[:,:,0] = image[:200,:]
            images.append(rgb_img)

        #====== load events ==============
        events = list()
        dir_events = self.events_dir[idx]
        for i in range(0,len(dir_events)):
            event = np.load(dir_events[i])
            event_t = event[:,:200,:]
            temp  = np.moveaxis(event_t, 0, 2)
            events.append(temp)

            
        #====== load labels ==============
        dir_labels = self.labels_dir[idx]
        labels = list()
        for i in range(0,len(dir_labels)):
            label = np.load(dir_labels[i])
            label1 = label[:200,:]
            label1 = np.expand_dims(label1, axis=-1)
            labels.append(label1)
        
        #====== load enhanced labels ==============
        labels_h = list()
        dir_labels_h = self.enhanced_depth_dir[idx]
        labels_h = list()
        for i in range(0,len(dir_labels_h)):
            label  = np.load(dir_labels_h[i])
            label1 = label[:200,:]
            label1 = np.expand_dims(label1, axis=-1)
            labels_h.append(label1)
        
        #====== transformer ==========
        if self.transform:            
            sample = (images , events, labels, labels_h)
            images, events, labels, input_label, labels_h = self.transform(sample)
      
        return images, events, labels, input_label, labels_h


class ToTensor:
    # Convert ndarrays to Tensors
    def __call__(self, sample):
        images , events, labels, labels_h  = sample
        img_t = list()
        eve_t = list()
        lab_t = list()
        lab_h = list()

        for i in range(0,len(images)):
            image = np.moveaxis(images[i], -1, 0)
            img_t.append(image)
            
            event = np.moveaxis(events[i], -1, 0)
            eve_t.append(event)
            
            label = np.moveaxis(labels[i], -1, 0)
            lab_t.append(label)

            label_h = np.moveaxis(labels_h[i], -1, 0)
            lab_h.append(label_h)
            
        
        img_t = np.array(img_t, np.float32)
        eve_t = np.array(eve_t, np.float32)
        lab_t = np.array(lab_t, np.float32)
        lab_h = np.array(lab_h, np.float32)
        
        image , event, label, label_h = torch.from_numpy(img_t) , torch.from_numpy(eve_t), torch.from_numpy(lab_t), torch.from_numpy(lab_h)
        label   = depth_creator(label,  min_d,max_d)
        label_h = depth_creator(label_h,min_d,max_d)

        input_label = add_noise(label)

        #======= images normalization ========
        for i in range(0,len(images)):
            if torch.max(image[i]) > torch.min(image[i]):
              image[i] = (image[i]) / 255.0    #(torch.max(image[i]))        
        for i in range(0,len(images)):
            for j in range(0,1):
                norm = transforms.Normalize((image[i].mean()), (image[i].std()))
                image[i] = norm(image[i])
                             
        '''
        #======= event scale to (-1,1) =======
        for i in range(0,len(images)):
            for j in range(0,5):
              if torch.max(event[i][j]) > torch.abs(torch.min(event[i][j])):
                event[i][j] = (event[i][j]) / (torch.max(event[i][j]))
              elif torch.max(event[i][j]) < torch.abs(torch.min(event[i][j])):
                event[i][j] = (event[i][j]) / torch.abs(torch.min(event[i][j]))
        '''

        #======= event normalize  =======
        for i in range(0,len(images)):
            for j in range(0,5):
              if torch.max(event[i][j]) > torch.abs(torch.min(event[i][j])):
                norm = transforms.Normalize((event[i:i+1,j:j+1,:,:].mean()), (event[i:i+1,j:j+1,:,:].std()))
                event[i:i+1,j:j+1,:,:] = norm(event[i:i+1,j:j+1,:,:])
                
        return image , event, label, input_label, label_h



class LRflip_rotate:
    # multiply inputs with a given factor
    def __init__(self,images_size):
        
        self.images_size = images_size
        

    def __call__(self, sample):
        
        #=== unpack data ==============
        images , events, labels, labels_h  = sample
        
        #====== fliping ===============
        # Random horizontal flipping
        random_state_flip = torch.rand(1)
        if random_state_flip.item() > 0.5:
            for i in range(0,len(images)):
                images[i]   = np.fliplr(images[i])
                events[i]   = np.fliplr(events[i])
                labels[i]   = np.fliplr(labels[i])
                labels_h[i] = np.fliplr(labels_h[i])

        

        #====== random crop in selected size ============
        p = torch.rand(1).item()
        if p >= 0.0:
            x = torch.randint(0,200 - self.images_size[1] ,(1,)).item()
            y = torch.randint(0,346 - self.images_size[0] ,(1,)).item()
            for i in range(0,len(images)):
                images[i]   = random_crop(images[i], x , y , self.images_size , cv2.INTER_LINEAR )
                events[i]   = random_crop(events[i], x , y , self.images_size , cv2.INTER_LINEAR )
                labels[i]   = random_crop(labels[i], x , y , self.images_size , cv2.INTER_NEAREST )    
                labels_h[i] = random_crop(labels_h[i], x , y , self.images_size , cv2.INTER_NEAREST )    


        #====== blur images ============
        for i in range(0,len(images)):
            p = torch.randint(5,11,(1,)).item()
            number_of_events = np.where(events[i] !=0)
            if p % 2 == 1 and len(number_of_events[0]) > 1000:
                    images[i] = cv2.GaussianBlur(images[i], (p,p),0)
                    images[i] = np.expand_dims(images[i], axis=-1)
            #else:
              #events[i] = np.zeros((self.images_size[1],self.images_size[0],5),dtype = np.float32)
        
        
        images   = np.array(images, np.float32)
        events   = np.array(events, np.float32)
        labels   = np.array(labels, np.float32)
        labels_h = np.array(labels_h, np.float32)
        
        return images , events ,labels, labels_h


    

class resize_test:
    # multiply inputs with a given factor
    def __init__(self, test_size):
        self.images_size = test_size


    def __call__(self, sample):
        
        #=== unpack data ==============
        images, events, labels, _  = sample
        #====== resize ===============
        re_images = list()
        re_events = list()
        re_labels = list()
        for i in range(0,len(images)):
            re_images.append(cv2.resize(images[i], self.images_size, interpolation = cv2.INTER_LINEAR))
            re_images[i] = np.expand_dims(re_images[i], axis=-1)

            
            re_events.append(cv2.resize(events[i], self.images_size, interpolation = cv2.INTER_LINEAR))
            
            re_labels.append(cv2.resize(labels[i], self.images_size, interpolation = cv2.INTER_NEAREST))
            re_labels[i] = np.expand_dims(re_labels[i], axis=-1)

        
        re_images = np.array(re_images, np.float32)
        re_events = np.array(re_events, np.float32)
        re_labels = np.array(re_labels, np.float32)
        
        return re_images, re_events, re_labels, re_labels


composed_train = torchvision.transforms.Compose([LRflip_rotate(images_size) , ToTensor()])
composed_test  = torchvision.transforms.Compose([resize_test(test_size) , ToTensor()])

# create dataset and set directories and pathes=========
#======== create directories ===========================
def get_dir(path):
    out = list()
    arr  = sorted(os.listdir(path))
    for i in arr :
        filename, file_extension = os.path.splitext(i)
        if file_extension == ".png" or file_extension== ".npy":
            temp = path + i 
            out.append(temp)
    return out



#========= colab path ==========
images_path_train  =  [
                    "/content/mvsec_dataset_day2/train/rgb/davis/", 
                    ]

events_path_train  = ["/content/mvsec_dataset_day2/train/events/voxels/",
                    ]

labels_path_train  = ["/content/mvsec_dataset_day2/train/depth/data/",
                    ]

labels_h_path_train = ["/content/content/enhanced_depth/",
                    ]


#======== test path day1 ============
images_path_test =  ["/content/mvsec_outdoor_day1/rgb/davis_left_sync/",]
events_path_test =  ["/content/mvsec_outdoor_day1/events/voxels/",]
labels_path_test =  ["/content/mvsec_outdoor_day1/depth/data/",]


'''
#======== test path night1 ============
images_path_test  =  ["/content/mvsec_outdoor_night1/rgb/davis_left_sync/",]
events_path_test  =  ["/content/mvsec_outdoor_night1/events/voxels/",]
labels_path_test  =  ["/content/mvsec_outdoor_night1/depth/data/",]
'''

'''
#======== test path night2 ============
images_path_test = ["/content/mvsec_outdoor_night2/rgb/davis_left_sync/",]
events_path_test = ["/content/mvsec_outdoor_night2/events/voxels/",]
labels_path_test = ["/content/mvsec_outdoor_night2/depth/data/",]
'''


#======== test path night3 ============
images_path_test = images_path_train =  ["/content/mvsec_outdoor_night3/rgb/davis_left_sync/",]
events_path_test = events_path_train =  ["/content/mvsec_outdoor_night3/events/voxels/",]
labels_path_test = labels_path_train =  ["/content/mvsec_outdoor_night3/depth/data/",]


#====== create train directories ===================
train_images_dir   = list()
train_events_dir   = list()
train_labels_dir   = list()
train_labels_h_dir = list()


for i in images_path_train:
    train_images_dir_s = get_dir(i)
    train_images_dir.append(train_images_dir_s)

for i in events_path_train:
    train_events_dir_s = get_dir(i)
    train_events_dir.append(train_events_dir_s)
    
for i in labels_path_train:
    train_labels_dir_s = get_dir(i)
    train_labels_dir.append(train_labels_dir_s)

for i in labels_h_path_train:
    train_labels_h_dir_s = get_dir(i)
    train_labels_h_dir.append(train_labels_h_dir_s)


#====== create test directories ===================
test_images_dir = list()
test_events_dir = list()
test_labels_dir = list()

for j in images_path_test:
    test_images_dir_s = get_dir(j)
    test_images_dir.append(test_images_dir_s)

for j in events_path_test:
    test_events_dir_s = get_dir(j)
    test_events_dir.append(test_events_dir_s)
    
for j in labels_path_test:
    test_labels_dir_s = get_dir(j)
    test_labels_dir.append(test_labels_dir_s)


def create_seq(images,events,labels,n_seq):
    
    output_img = list()
    output_eve = list()
    output_lab = list()
    for s in range(0,len(images)):
        step = len(images[s]) // n_seq
        step = int(step-1)
        r = 0
        for i in range(0,step):
            temp = list()
            temp_eve = list()
            temp_lab = list()
            for j in range(r , n_seq + r):
                temp.append(images[s][i*n_seq + j])
                temp_eve.append(events[s][i*n_seq + j])
                temp_lab.append(labels[s][i*n_seq + j])
            output_img.append(temp)
            output_eve.append(temp_eve)
            output_lab.append(temp_lab)
    return output_img , output_eve ,output_lab




train_img_seq , train_eve_seq, train_lab_seq_h = create_seq(train_images_dir, train_events_dir, train_labels_h_dir, n_seq)
train_img_seq , train_eve_seq, train_lab_seq   = create_seq(train_images_dir, train_events_dir, train_labels_dir,   n_seq)
test_img_seq  , test_eve_seq,  test_lab_seq    = create_seq(test_images_dir , test_events_dir , test_labels_dir,    n_seq_test )

#========= set dataset ==============
train_dataset = Dataloader_seq(train_img_seq,
                               train_eve_seq,
                               train_lab_seq,
                               train_lab_seq_h,
                               transform=composed_train)

test_dataset  = Dataloader_seq(test_img_seq,
                               test_eve_seq,
                               test_lab_seq,
                               test_lab_seq,
                               transform=composed_test)


#======= set loader ===========================
train_loader = DataLoader(dataset=train_dataset,
                          batch_size=batch_size_train,
                          shuffle=True,
                          num_workers=2)

#======= set loader ===========================
test_loader  = DataLoader(dataset=test_dataset,
                          batch_size=batch_size_test,
                          #shuffle= True,
                          num_workers=2)


'''
#======== test dataset =============
# get first sample and unpack
first_data = train_dataset[0]
images_t, labels_t = first_data

#======== plot image ======
plt.imshow(images_t[9][0].numpy())
plt.show()
#======== plot labels ======
plt.imshow(labels_t[0].numpy())
plt.show()
'''



'''
#======== test dadaloder ===============
sample = next(iter(train_loader))
images_t , events_t, labels_t = sample
for i in range(0,batch_size_train):
    for j in range(0,3):
        plt.imshow(images_t[i][0][j].numpy())
        plt.show()
        print(torch.max(images_t))
        print(torch.min(images_t))
    
    for j in range(0,5):
        plt.imshow(events_t[i][0][j].numpy())
        plt.show()
        print(torch.max(events_t))
        print(torch.min(events_t))
        
    plt.imshow(labels_t[i][0][0].numpy())
    plt.show()
    print(torch.max(labels_t))
    print(torch.min(labels_t))

'''
'''
#======== test dadaloder ===============
sample = next(iter(test_loader))
images_t , events_t, labels_t = sample
for i in range(0,batch_size_train):
    for j in range(0,3):
        plt.imshow(images_t[i][0][j].numpy())
        plt.show()
        print(torch.max(images_t))
        print(torch.min(images_t))
    
    for j in range(0,5):
        plt.imshow(events_t[i][0][j].numpy())
        plt.show()
        print(torch.max(events_t))
        print(torch.min(events_t))
        
    plt.imshow(labels_t[i][0][0].numpy())
    plt.show()
    print(torch.max(labels_t))
    print(torch.min(labels_t))
'''
'''
import numpy as np
path = "/home/mehdi/Downloads/"
name = "depth_0000000000.npy"

t = np.load(path + name) 

t1 = np.nan_to_num(t)
plt.imshow(t1)
plt.show()
'''
