# -*- coding: utf-8 -*-
"""
Modified on Wed Dec 24 10:24:47 2019

@author: liujie

"""
import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import layers, models

IMAGE_SIZE = 32
train_dir = "./data/train"
test_dir = "./data/test"

#读取训练数据
images = []
labels = []

def read_path(data_path):    
    for dir_item in os.listdir(data_path):
        #从初始路径开始叠加，合并成可识别的操作路径
        full_path = os.path.abspath(os.path.join(data_path, dir_item))
        
        if data_path.endswith('airplane'):labels.append(0)
        if data_path.endswith('automobile'):labels.append(1)
        if data_path.endswith('bird'):labels.append(2)
        if data_path.endswith('cat'):labels.append(3)
        if data_path.endswith('deer'):labels.append(4)
        if data_path.endswith('dog'):labels.append(5)
        if data_path.endswith('frog'):labels.append(6)
        if data_path.endswith('horse'):labels.append(7)
        if data_path.endswith('ship'):labels.append(8)
        if data_path.endswith('truck'):labels.append(9)
        
        if os.path.isdir(full_path):    
            #如果是文件夹，继续递归调用           
            read_path(full_path)
            
        else:   #文件 cv2.imread 能自动识别格式
            if dir_item.endswith('.jpg') or dir_item.endswith('.bmp'):
                #cv2.imread 能自动识别格式
                image = cv2.imread(full_path)                
                images.append(image)                                              
                    
    return images,labels
    

#从指定路径读取训练数据
def load_dataset(data_path):
    images,labels = read_path(data_path)    

    images = np.array(images)
    labels = np.array(labels) 
   
    return images, labels                                
                    


x_test, y_test = load_dataset(test_dir)
x_train, y_train = load_dataset(train_dir)


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
        # The CIFAR labels happen to be arrays, 
        # which is why you need the extra index

plt.show()



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
# 迭代10次 epochs=10
history = model.fit(x_train, y_train, epochs=10, 
                    validation_data=(x_test, y_test))


plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0.2, 1.0])
plt.legend(loc='lower right')



    
test_loss, test_acc = model.evaluate(x_test,  y_test, verbose=2)
print(test_acc)

''''''
