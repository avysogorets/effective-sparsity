import tensorflow as tf
import numpy as np
from tensorflow.keras import datasets
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tinyimagenet import *

def get_data(dataset,path_to_data=None):
  aug_config={}
  if dataset.lower()=='mnist':
    (train_X,train_y),(test_X,test_y)=datasets.mnist.load_data()
    aug_config['rotation_range']=4
    mean,std=0.1307,0.3081
    num_classes=10
  elif dataset.lower()=='cifar10':
    (train_X,train_y),(test_X,test_y)=datasets.cifar10.load_data()
    aug_config['horizontal_flip']=True
    aug_config['width_shift_range']=aug_config['height_shift_range']=4
    mean,std=[0.4914,0.4822,0.4465],[0.2470,0.2435,0.2616]
    num_classes=10
  elif dataset.lower()=='cifar100':
    aug_config['horizontal_flip']=True
    aug_config['width_shift_range']=aug_config['height_shift_range']=4
    (train_X,train_y),(test_X,test_y)=datasets.cifar100.load_data()
    mean,std=[0.5071,0.4865,0.4409],[0.2673,0.2564,0.2762]
    num_classes=100
  elif dataset.lower()=='tinyimagenet':
    aug_config['horizontal_flip']=True
    aug_config['width_shift_range']=aug_config['height_shift_range']=4
    train_X,train_y,test_X,test_y=load_images(path_to_data+'/tiny-imagenet-200',200)
    train_X,test_X=np.float32(np.transpose(train_X,axes=(0,2,3,1))),np.float32(np.transpose(test_X,axes=(0,2,3,1))) 
    mean,std=[0.4802,0.4481,0.3975],[0.2770,0.2691,0.2821]
    num_classes=200
  train_y=tf.keras.utils.to_categorical(train_y,num_classes)
  test_y=tf.keras.utils.to_categorical(test_y,num_classes)
  train_X,test_X=np.divide(train_X,255),np.divide(test_X,255)
  train_X,test_X=(train_X-mean)/std,(test_X-mean)/std
  train_X=np.expand_dims(train_X,axis=3) if len(train_X.shape)<4 else train_X # adding channel dimensions when it's not there
  test_X=np.expand_dims(test_X,axis=3) if len(test_X.shape)<4 else test_X
  datagen=ImageDataGenerator(**aug_config)
  datagen.fit(train_X)
  return datagen,train_X,train_y,test_X,test_y