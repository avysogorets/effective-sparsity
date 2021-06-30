import tensorflow as tf
import numpy as np
import logging
from utils import *

class SubnetworkCallback(tf.keras.callbacks.Callback):
  def __init__(self,model,tensors,masks):
    super().__init__()
    self.model=model
    self.tensors=tensors
    self.masks=masks
  def on_train_batch_begin(self,batch,logs={}):
    curr_weights=[self.model.layers[layer].get_weights()[0] for layer in self.tensors]
    set_weights_model(self.model,self.tensors,curr_weights,masks=self.masks)

class LogCallback(tf.keras.callbacks.Callback):
  def __init__(self,model,tensors,masks,log_list,test):
    super().__init__()
    self.model=model
    self.masks=masks
    self.final_weights=None
    self.test_X,self.test_y=test
    self.tensors=tensors
    self.log_list=log_list
    self.iteration=0
    self.epoch=1
    self.losses=[]
    self.accuracies=[]
  def on_train_batch_begin(self,batch,logs={}):
    if self.iteration in self.log_list:
      test_metrics=self.model.evaluate(x=self.test_X,y=self.test_y,batch_size=128,verbose=False)
      self.losses.append(test_metrics[0])
      self.accuracies.append(test_metrics[1])
      if len(self.accuracies)>5 and sum(self.accuracies[-3:])<=3*(1./len(self.test_y[0])):
        self.model.stop_training # stop training if accuracy doesn't improve
        self.on_train_end(self.iteration)
      logging.info(f"<callbacks> [iteration/epoch: {self.iteration}/{self.epoch}][lr: {self.model.optimizer.lr(self.iteration):.4f}][val loss: {test_metrics[0]:.4f}][val acc: {test_metrics[1]:.4f}]")
    self.iteration+=1
  def on_epoch_end(self,batch,logs={}):
    self.epoch+=1
  def on_train_end(self,batch,logs={}):
    self.final_weights=[self.model.layers[layer].get_weights()[0] for layer in self.tensors]
