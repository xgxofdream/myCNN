# -*- coding: utf-8 -*-
# 第2种偶联数据集合和算法模型的方法
"""
Created @author: Cloudox

Modified on Wed Dec 24 10:24:47 2019

@author: liujie

"""

import glob
import os
import numpy as np
from skimage import io,transform
import matplotlib.pyplot as plt
from tensorflow.keras import layers, models

#--------------------------------------基本参数配置-----------------------------
IMAGE_SIZE = 32
Channel = 3
train_dir = "./data/train/"
test_dir = "./data/test/"
#图片类别为10
num_classes = 10

#模型保存地址
save_dir = os.path.join(os.getcwd(), 'saved_models')
model_name = 'keras_cifar10_trained_model.h5'

#迭代次数
epoch_num = 1

#初始化
images = []
labels = []
#--------------------------------------函数定义-----------------------------
#读取图片,仅支持2级文件夹
def load_dataset(path):
    cate=[path+x for x in os.listdir(path) if os.path.isdir(path+x)]
    imgs=[]
    labels=[]
    print("reading:", cate)
    #读取子文件里的图片文件
    #！将循环次数idx赋值给图片的标记label
    for idx,folder in enumerate(cate):
        for im in glob.glob(folder+'/*.jpg'):
            #print('reading the images:%s'%(im))
            #print('reading the image label:%s'%(idx))
            img=io.imread(im)
            #将所有的图片resize
            #img=transform.resize(img,(w,h))
            imgs.append(img)
            labels.append(idx)
    return np.asarray(imgs,np.float32),np.asarray(labels,np.int32)

#--------------------------------------读入数据集-----------------------------
#导入数据到训练变量x_test, y_test，x_train, y_train
x_test, y_test = load_dataset(test_dir)
x_train, y_train = load_dataset(train_dir)

#归一化数据（可以不要）
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

'''
# 将label转化成2进制值（可以不要）
# Convert class vectors to binary class matrices.
# 但是进入训练进程后，会报错：
# Can not squeeze dim[1], expected a dimension of 1, got 10 for 'metrics/accuracy/Squeeze' (op: 'Squeeze') with input shapes: [?,10].
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)
'''


#验证数据集信息
print("Train data: ", x_train.shape)
print("Train labels: ", y_train.shape) 
print("Test data: ", x_test.shape)
print("Test labels: ", y_test.shape)

plt.figure(figsize=(10,10))
for i in range(9):
        plt.subplot(3,3,i+1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(x_train[i], cmap=plt.cm.binary)
plt.show()


#--------------------------------------Keras网络配置-----------------------------
# the following code is from 
# https://colab.research.google.com/github/tensorflow/docs/blob/master/site/en/tutorials/images/cnn.ipynb#scrollTo=0LvwaKhtUdOo
# it is based on Keras directly, not using tensorflow as backend    
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.summary()
    
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))
model.summary()
    
model.compile(optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy'])

#--------------------------------------训练模型-----------------------------
# 迭代
history = model.fit(x_train, y_train, epochs=epoch_num, 
                    validation_data=(x_test, y_test))

#--------------------------------------保存模型-----------------------------
# Save model and weights
if not os.path.isdir(save_dir):
    os.makedirs(save_dir)
model_path = os.path.join(save_dir, model_name)
model.save(model_path)
print('Saved trained model at %s ' % model_path)

#--------------------------------------模型性能display-----------------------------

plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0.2, 1.0])
plt.legend(loc='lower right')

    
test_loss, test_acc = model.evaluate(x_test,  y_test, verbose=2)
print(test_acc)

