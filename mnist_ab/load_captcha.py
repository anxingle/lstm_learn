#!/usr/bin/env python
# coding=utf-8
import numpy as np
import struct
import os
import cv2
import random

ROW   = 28
COL   = 28
depth = 1


def dense_to_one_hot(labels_dense, num_classes):
  """Convert class labels from scalars to one-hot vectors."""
  num_labels = labels_dense.shape[0]
  index_offset = np.arange(num_labels) * num_classes
  labels_one_hot = np.zeros((num_labels, num_classes))
  labels_one_hot.flat[[index_offset + labels_dense.ravel()]] = 1
  return labels_one_hot
def read_images(train_dir,length_label):
    images = os.listdir(train_dir)
    # volume of data_set (train_dir)
    data_volume = len(images)
    # get a random list intend to shuffle the data_set 
    images_list = range(data_volume)
    random.shuffle(images_list)
    random.shuffle(images_list)
    # return to the main Call function
    # data.shape  = (volume,row,col,depth)
    data  = np.zeros((data_volume,ROW,COL,depth))
    # label.shape = (volume,row,col,depth)
    #label = np.zeros((data_volume,1))
    label = np.zeros((data_volume,2))
    for loop in xrange(data_volume):
        image_name  =  images[images_list[loop]]
        image       =  os.path.join(train_dir,image_name)
        img         =  cv2.imread(image,cv2.IMREAD_GRAYSCALE)
        img         =  img.reshape((ROW,COL,depth))
        data[loop]  =  img

        image_name  =  image_name.strip()
        image_name  =  image_name.split('.')[0]
        image_name  =  image_name.split('_')[1]
        #image_label =  image_name[:]
        image_label = [0,0]
        if image_name[0] == 'A':
            image_label[0] = 10
        elif image_name[0] == 'B':
            image_label[0] = 11
        else:
            image_label[0] = image_name[0] 
        if image_name[1] == 'A':
            image_label[1] = 10
        elif image_name[1] == 'B':
            image_label[1] = 11
        else:
            image_label[1] = image_name[1]
        #label.append(image_label)
        label[loop] = image_label 
    return data,label


def read_data(train_dir,one_hot=True,length_label=1,num_class=10,reshape=True):
#def read_data(train_dir,one_hot=True,dtype=dtypes.float32,reshape=True):
    train_images,train_labels = read_images(train_dir,length_label)
    if one_hot:
        train_labels = dense_to_one_hot(train_labels,num_class)
    return train_images,train_labels
