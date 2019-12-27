import glob
import os
import numpy as np
import tensorflow as tf
from skimage import io,transform
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score


#--------------------------------------基本参数配置-----------------------------
test_dir = "./data/test/"

class_names = cifar10_classes = ["airplane", "automobile", "bird", "cat", "deer", 
                   "dog", "frog", "horse", "ship", "truck"]



#--------------------------------------自定义函数-----------------------------
def plot_images(images, labels_true, class_names, labels_pred=None):

    assert len(images) == len(labels_true)

    # Create a figure with sub-plots
    fig, axes = plt.subplots(12, 27, figsize = (40,40))

    # Adjust the vertical spacing
    if labels_pred is None:
        hspace = 0.2
    else:
        hspace = 0.5
    fig.subplots_adjust(hspace=hspace, wspace=0.3)

    for i, ax in enumerate(axes.flat):
        # Fix crash when less than 9 images
        if i < len(images):
            # Plot the image
            ax.imshow(images[i], interpolation='spline16')
            
            # Name of the true class
            labels_true_name = class_names[labels_true[i]]

            # Show true and predicted classes
            if labels_pred is None:
                xlabel = "True: "+labels_true_name
            else:
                # Name of the predicted class
                labels_pred_name = class_names[labels_pred[i]]

                xlabel = "True: "+labels_true_name+"\nPredicted: "+ labels_pred_name

            # Show the class on the x-axis
            ax.set_xlabel(xlabel)
        
        # Remove ticks from the plot
        ax.set_xticks([])
        ax.set_yticks([])

    
    # Show the plot
    plt.show()
    

#------------------------------调入训练好的模型--------------------------------------
# 不要用Keras的load功能：from keras.models import load_model
# 因为我model是它创建的：from tensorflow.keras import layers, models
model = tf.keras.models.load_model('./saved_models/keras_cifar10_trained_model.h5')
model.summary()


#------------------------------调入拟预测的图片--------------------------------------
def load_dataset(path):
    cate=[path+x for x in os.listdir(path) if os.path.isdir(path+x)]
    imgs=[]
    labels=[]
    #print("reading:", cate)
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

#归一化数据（可以不要）
x_test = x_test.astype('float32')
x_test /= 255

#------------------------------执行预测--------------------------------------
# get predictions on the test set
prediction = model.predict_proba(x_test)
print(prediction)
print(prediction[0])

# np.argmax返回沿轴axis最大值的索引
labels_pred = np.argmax(prediction,axis=1)
correct = (labels_pred == y_test)
num_images = len(correct)

print(labels_pred)
print(correct)
print("Number of correct predictions: %d" % sum(correct))
print("Accuracy: %.2f%%" % ((sum(correct)*100)/num_images))

#------------------------------可视化--------------------------------------
#-----------------confusion matrix and accuracy
plt.figure(figsize=(7, 6))
plt.title('Confusion matrix', fontsize=16)
plt.imshow(confusion_matrix(y_test, labels_pred))
plt.xticks(np.arange(10), cifar10_classes, rotation=45, fontsize=12)
plt.yticks(np.arange(10), cifar10_classes, fontsize=12)
plt.colorbar()
plt.show()

#----------------------------------list wrong predicted images
# Get the incorrectly classified images
incorrect = (correct == False)
images_error = x_test[incorrect]
labels_error = labels_pred[incorrect]
labels_true = y_test[incorrect]
# Plot the first 9 mis-classified images
plot_images(images=images_error[0:324],labels_true=labels_true[0:324],class_names=class_names,labels_pred=labels_error[0:324])

''''''
