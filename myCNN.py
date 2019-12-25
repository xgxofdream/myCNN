# -*- coding: utf-8 -*-
"""
Created on Thu Jul 12 16:34:16 2018

@author: yy

Modified on Wed Dec 24 10:24:47 2019

@author: liujie

"""

import sys
import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models
from keras.utils import np_utils

IMAGE_SIZE = 32
train_dir = "./data/jay"
test_dir = "./data/jane"


#从指定路径读取训练数据
def load_dataset(data_path):
    images = []
    labels = []
    for dir_item in os.listdir(data_path):
        #从初始路径开始叠加，合并成可识别的操作路径
        full_path = os.path.abspath(os.path.join(data_path, dir_item))
        
        if os.path.isdir(full_path):    
            #如果是文件夹，继续递归调用
            read_path(full_path)
        else:   #文件 cv2.imread 能自动识别格式
            if dir_item.endswith('.jpg') or dir_item.endswith('.bmp'):
                #cv2.imread 能自动识别格式
                image = cv2.imread(full_path)
                #cv2.resize 整理成4维数据                
                image = cv2.resize(image, (IMAGE_SIZE, IMAGE_SIZE), 3)
                
                #放开这个代码，可以看到resize_image()函数的实际调用效果
                cv2.imwrite('1.jpg', image)
                
                images.append(image)                
                labels.append(data_path)

    #将输入的所有图片转成四维数组，尺寸为(图片数量*IMAGE_SIZE*IMAGE_SIZE*3)
    #训练集图片总数 num，IMAGE_SIZE为64，故尺寸为 num * 64 * 64 * 3
    #图片为64 * 64像素,一个像素3个颜色值(BGR)
    images = np.array(images)
  
  
    #标注数据
    newlabels = []
    for label in labels:
        newlabels.append(1)
    labels = np.array(newlabels)
    
    return images, labels                                
                    

x_train, y_train = load_dataset(train_dir)
#y_train = np_utils.to_categorical(y_train, 1)

x_test, y_test = load_dataset(test_dir)
#y_test = np_utils.to_categorical(y_test, 1)

''''''
print("Train data: ", x_train.shape)
print("Train labels: ", y_train.shape) 
print("Test data: ", x_test.shape)
print("Test labels: ", y_test.shape)


plt.figure(figsize=(10,10))
for i in range(25):
        plt.subplot(5,5,i+1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(x_train[i], cmap=plt.cm.binary)
        # The CIFAR labels happen to be arrays, 
        # which is why you need the extra index

plt.show()


''''''
# the following code is from 
# https://colab.research.google.com/github/tensorflow/docs/blob/master/site/en/tutorials/images/cnn.ipynb#scrollTo=0LvwaKhtUdOo
# it is based on Keras directly, not using tensorflow as backend    
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
model.add(layers.MaxPooling2D((2, 2)))

model.summary()
    
model.add(layers.Flatten())
model.add(layers.Dense(2, activation='relu'))
model.add(layers.Dense(2, activation='softmax'))
model.summary()
    
model.compile(optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy'])
# 迭代2次 epochs=2
history = model.fit(x_train, y_train, epochs=10, 
                    validation_data=(x_test, y_test))


plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0.5, 1.2])
plt.legend(loc='lower right')
    
test_loss, test_acc = model.evaluate(x_test,  y_test, verbose=10)
print(test_acc)

