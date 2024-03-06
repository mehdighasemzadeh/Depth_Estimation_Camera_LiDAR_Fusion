#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  8 22:32:34 2023

@author: mehdi
"""
import numpy as np
import skimage.exposure
import cv2
import torch
from config import device
import matplotlib.pyplot as plt
from config import images_size


def LogToDepth(input_log ,min_s,max_s):
    input_log = input_log.to(device)
    min_s = torch.tensor([min_s,]).to(device)
    max_s = torch.tensor([max_s,]).to(device)
    alpha = -1 * torch.log( max_s / min_s )
    input_log  = max_s * torch.exp( alpha * (1 - input_log) )
    
    return input_log




#===== define color map =======
# define colors
color1 = (255, 0, 0) # red
color2 = (200, 0, 50) # red
color3 = (150, 0, 100) # red 
color4 = (100, 0, 150) #  
color5 = (50, 0, 200) # blue
color6 = (0, 0, 250) # blue
color7 = (0, 0, 0)   # black

colorArr = np.array([[color1, color2, color3,color4, color5, color6,color7]], dtype=np.uint8)
# resize lut to 256 (or more) values
lut = cv2.resize(colorArr, (256,1), interpolation = cv2.INTER_LINEAR)
for i in range(0,256):
    lut[0][i][0] = 255 - i
    lut[0][i][1] = 255 - i
    lut[0][i][2] = 127 - i/2
    
    



def depth_to_color(img_path):
    # load image as grayscale
    img = np.load(img_path)
    img = torch.from_numpy(img)
    img = LogToDepth(img,2,80)
    img = img.to('cpu').numpy()
    img = img.astype(np.uint8)
    # stretch to full dynamic range
    stretch = skimage.exposure.rescale_intensity(img, in_range='image', out_range=(0,255)).astype(np.uint8)
    # convert to 3 channels
    stretch = cv2.merge([stretch,stretch,stretch])
    # apply lut
    result = cv2.LUT(stretch, lut)
    return result

test_path = "/home/mehdi/Downloads/label.npy"
img_depth = depth_to_color(test_path)
plt.imshow(img_depth)
plt.show()

test_path = "/home/mehdi/Downloads/output.npy"
img_depth = depth_to_color(test_path)
plt.imshow(img_depth)
plt.show()

label_path  = "/home/mehdi/Downloads/labels/content/labels/"
output_path = "/home/mehdi/Downloads/outputs/content/outputs/"
event_path = "/home/mehdi/Downloads/test_sequence_00_town10/events/frames_white/"
img_path   = "/home/mehdi/Downloads/test_sequence_00_town10/rgb/frames/"

# create dataset and set directories and pathes=========
#======== create directories ===========================
import os
def get_dir(path):
    out = list()
    arr  = sorted(os.listdir(path))
    for i in arr :
        filename, file_extension = os.path.splitext(i)
        if file_extension == ".png" or file_extension== ".npy":
            temp = path + i 
            out.append(temp)
    return out

label_dir  = get_dir(label_path)
output_dir = get_dir(output_path)
event_dir  = get_dir(event_path)
img_dir    = get_dir(img_path)

font = cv2.FONT_HERSHEY_SIMPLEX
color_map = (255,32,100) 
for i in range(0,len(label_dir)):
    label  = depth_to_color(label_dir[i])
    output = depth_to_color(output_dir[i])
    
    image = cv2.imread(img_dir[i])
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, images_size, interpolation = cv2.INTER_LINEAR)
    
    event = cv2.imread(event_dir[i])
    event = cv2.cvtColor(event, cv2.COLOR_BGR2RGB)
    event = cv2.resize(event, images_size, interpolation = cv2.INTER_LINEAR)
    
    img = np.zeros((192,1024,3) , np.uint8)
    img[:,0:256,:]   = image
    img[:,256:512,:] = event
    img[:,512:768,:] = output
    img[:,768:,:]    = label
    save_name = str(i).zfill(5)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    
    img = cv2.line(img, (254,0), (254,192), color_map, 3)
    img = cv2.line(img, (511,0), (511,192), color_map, 3)
    img = cv2.line(img, (767,0), (767,192), color_map, 3)
   
    
    img = cv2.putText(img, 'Event', (10,15), font, 
                   0.5, color_map, 1, cv2.LINE_AA)
    
    img = cv2.putText(img, 'Image', (270,15), font, 
                   0.5, color_map, 1, cv2.LINE_AA)
    
    img = cv2.putText(img, 'P', (535,15), font, 
                   0.5, color_map, 1, cv2.LINE_AA)
    
    img = cv2.putText(img, 'GT', (780,15), font, 
                   0.5, color_map, 1, cv2.LINE_AA)
    cv2.imwrite("/home/mehdi/Desktop/depth_output/img_" + save_name + ".png", img)



#============= create videos ================================================
width  = 1024
height = 192
#this fourcc best compatible for avi
fourcc = cv2.VideoWriter_fourcc('M','J','P','G')
video=cv2.VideoWriter('/home/mehdi/Desktop/output.avi',fourcc, 15.0, (width,height))

Image_path = "/home/mehdi/Desktop/depth_output/"
output_Image_dir = get_dir(Image_path)


for i in range(0,len(output_Image_dir)):
     x=cv2.imread(output_Image_dir[i])
     video.write(x)

cv2.destroyAllWindows()
video.release()

