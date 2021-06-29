import tensorflow as tf
import numpy as np
import callbacks
import pruning
import data
import os
import models
import argparse
from effective_masks import *
from utils import *
import logging

os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
try:
  tf.keras.backend.set_floatx('float64')
except:
  tf.keras.backend.set_floatx('float32')
parser=argparse.ArgumentParser()
parser.add_argument('--data',type=str,default='mnist',help='dataset (mnist, cifar10, cifar100, or tinyimagenet)')
parser.add_argument('--sample',type=str,default='0',help='seed code')
parser.add_argument('--path_to_data',type=str,help='path to tinyimagenet folder')
parser.add_argument('--save',type=int,default=1,help='save results (1) or not (0)')
parser.add_argument('--architecture',type=str,default='lenet300100',help='network (lenet300100, lenet5, vgg16, vgg19, or resnet18)')
parser.add_argument('--pruner',type=str,default='snip_global',help='pruner (snip, synflow, etc.)')
parser.add_argument('--com_exp',type=float,default=None,help='target compression = 10 ** (com_exp)')
parser.add_argument('--target_sparsity',type=float,default=0.9,help='target sparsity (overwritten by --com_exp if given)')
parser.add_argument('--pruning_type',type=str,default='direct',help='direct or effective pruning')
parser.add_argument('--train',type=int,default=1,help='train (1) or prune only (0)')
parser.add_argument('--out_path',type=str,default='EffectiveSparsity',help='path to directory for outputs')
args=parser.parse_args()
args.target_sparsity=0 if args.pruner=='dense' else args.target_sparsity
args.pruning_type='' if args.pruner=='dense' else args.pruning_type

lenet300100_config={'lr':0.1,'batch_size_train':100,'iterations':96000,'weight_decay':0.0005,'batchnorm':True,'momentum':0.9,'lr_decay':[25000,50000,75000,100000],'batch_size_snip':100} # source: Lee et al., 2018
lenet5_config={'lr':0.1,'batch_size_train':128,'iterations':120000,'weight_decay':0.0005,'batchnorm':True,'momentum':0.9,'lr_decay':[30000,60000,90000,120000],'batch_size_snip':128} # source: Lee et al., 2018
vgg16_config={'lr':0.1,'batch_size_train':128,'iterations':62500,'weight_decay':0.0001,'batchnorm':True,'momentum':0.9,'lr_decay':[31250,46875],'batch_size_snip':128} # source: Frankle et al., 2020
vgg19_config={'lr':0.1,'batch_size_train':128,'iterations':62500,'weight_decay':0.0001,'batchnorm':True,'momentum':0.9,'lr_decay':[31250,46875],'batch_size_snip':1280} # source: Wang et al., 2020
resnet18_config={'lr':0.2,'batch_size_train':256,'iterations':78200,'weight_decay':0.0001,'batchnorm':True,'momentum':0.9,'lr_decay':[39100,58650],'batch_size_snip':2560} # source: Frankle et al., 2020

if args.architecture=='lenet300100':
  config=lenet300100_config
if args.architecture=='lenet5':
  config=lenet5_config
if args.architecture=='vgg16':
  config=vgg16_config
if args.architecture=='vgg19':
  config=vgg19_config
if args.architecture=='resnet18':
  config=resnet18_config

def main(args):
  target_compression=10**args.com_exp if args.com_exp is not None else 1./(1-args.target_sparsity)
  extension=f'{args.sample}_{int(target_compression)}_'
  args.out_path=os.path.join(args.out_path,args.architecture,args.pruner,args.pruning_type)
  path_to_dense=os.path.join(args.out_path,args.architecture,'dense')
  if not os.path.exists(args.out_path):
    os.makedirs(args.out_path)
  logging.basicConfig(filename=os.path.join(args.out_path,extension+'info.log'),level=logging.INFO,filemode='w')
  datagen,train_X,train_y,test_X,test_y=data.get_data(args.data,path_to_data=args.path_to_data)
  epochs=int(config['batch_size_train']*config['iterations']/len(train_X))
  model,tensors=models.get_model(shape=train_X[0].shape,architecture=args.architecture,batchnorm=config['batchnorm'],decay=config['weight_decay'],output_classes=len(train_y[0]))
  values=[config['lr']*(0.1**i) for i in range(len(config['lr_decay'])+1)]
  learningrate=tf.keras.optimizers.schedules.PiecewiseConstantDecay(config['lr_decay'],values)
  model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=learningrate,momentum=config['momentum']),loss='categorical_crossentropy',metrics=['accuracy'])
  log_list=np.arange(0,config['iterations'],1000)
  pruner=pruning.Pruner(args.pruner)
  masks=pruner.prune(model,tensors,1-1./target_compression,args.pruning_type,epochs=epochs,train_X=train_X,train_y=train_y,out_path=os.path.join(args.out_path,extension),config=config,sample=args.sample,path_to_dense=path_to_dense)
  inits=[model.layers[layer].get_weights()[0] for layer in tensors]
  log_cb=callbacks.LogCallback(model,tensors,masks,log_list,(test_X,test_y),(train_X,train_y),os.path.join(args.out_path,extension),args.save)
  fit_callbacks=[callbacks.SubnetworkCallback(model,tensors,masks),log_cb]
  eff_masks_custom=effective_masks_custom(model.name,masks)
  eff_masks_synflow=effective_masks_synflow(model,tensors,masks)
  logging.info(f'<main> [direct sparsity: {get_overall_direct_sparsity(masks):.6f}][effective sparsity: {get_overall_direct_sparsity(eff_masks_synflow):.6f}][epochs to train: {epochs}][iterations to train: {config["iterations"]}][pruner: {args.pruner}][sample: {args.sample}]')
  if args.train:
    if np.array(get_direct_sparsity(eff_masks_synflow)==1).all():
      if args.save:
        np.save(os.path.join(args.out_path,extension)+'accuracies.npy',np.array([1./len(train_y[0])]))
    else:
      model.fit(datagen.flow(train_X,train_y,batch_size=config['batch_size_train']),steps_per_epoch=len(train_X)//config['batch_size_train'],epochs=epochs,shuffle=True,verbose=False,validation_data=(test_X,test_y),callbacks=fit_callbacks)
  if args.save:
    np.save(os.path.join(args.out_path,extension)+'sparsities_effective_synflow.npy',get_direct_sparsity(eff_masks_synflow))
    np.save(os.path.join(args.out_path,extension)+'sparsities_effective_custom.npy',get_direct_sparsity(eff_masks_custom))
    np.save(os.path.join(args.out_path,extension)+'sparsities_direct.npy',get_direct_sparsity(masks))
    if args.train and args.pruner=='dense':
      np.save(os.path.join(args.out_path,extension)+'inits.npy',inits)
      np.save(os.path.join(args.out_path,extension)+'final_weights.npy',log_cb.final_weights)
      np.save(os.path.join(args.out_path,extension)+'counts.npy',[np.prod(model.layers[layer].get_weights()[0].shape) for layer in tensors])

if __name__=="__main__":
  main(args)