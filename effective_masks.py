import numpy as np
import tensorflow as tf
from utils import *
from models import get_model
import logging

def stabilizer_synflow_layer(stab_low,stab_high,inter_model,layer,low_lim,high_lim,attempt):
  if attempt>10:
    logging.warning(f'<effective_masks> unable to stabilize layer {layer}')
    return 0
  stabilizer=(stab_low+stab_high)/2
  weight=inter_model.layers[layer].get_weights()[0]
  set_weights_layer(inter_model,layer,weight*(10**stabilizer))
  output=abs(inter_model(np.ones([1]+inter_model.inputs[0].shape[1:])).numpy())
  if sum(output.reshape(-1))>10**high_lim:
    set_weights_layer(inter_model,layer,weight)
    return stabilizer_synflow_layer(stab_low,stabilizer,inter_model,layer,low_lim,high_lim,attempt+1)
  elif sum(output.reshape(-1))<10**low_lim:
    set_weights_layer(inter_model,layer,weight)
    return stabilizer_synflow_layer(stabilizer,stab_high,inter_model,layer,low_lim,high_lim,attempt+1)

def stabilizer_synflow(stab_low,stab_high,model,tensors,low_lim,high_lim):
  for layer in tensors:
    inter_model=models.Model(inputs=model.inputs,outputs=model.layers[layer].output)
    stabilizer_synflow_layer(stab_low,stab_high,inter_model,layer,low_lim,high_lim,0)
    weight=model.layers[layer].get_weights()[0]

def effective_masks_synflow(model,tensors,masks):
  in_shape=model.inputs[0].get_shape().as_list()[1:]
  num_classes=model.output[0].get_shape().as_list()[0]
  shapes=[model.layers[layer].get_weights()[0].shape for layer in tensors]
  counts=[np.prod(shape) for shape in shapes]
  counts_sum=[sum(counts[:layer]) for layer in range(len(shapes)+1)]
  linear_model,linear_tensors=get_model(shape=in_shape,architecture=model.name.replace('-','').lower(),batchnorm=False,activation=False,pool='average',output_classes=num_classes)
  if model.name=='ResNet-18':
    main_tensors=linear_tensors[:7]+linear_tensors[8:12]+linear_tensors[13:17]+linear_tensors[18:]
  abs_inits=[abs(linear_model.layers[layer].get_weights()[0]) for layer in linear_tensors]
  set_weights_model(linear_model,linear_tensors,abs_inits,masks=masks)
  #stabilizer_synflow(-7,7,linear_model,main_tensors if 'ResNet' in model.name else linear_tensors,-10,38)
  with tf.GradientTape(persistent=False) as tape:
    output=linear_model(np.ones([1]+linear_model.inputs[0].shape[1:]))
    saliency=tf.reduce_sum(output)
    weights=[linear_model.layers[layer].trainable_weights[0] for layer in linear_tensors]
  gradients=tape.gradient(saliency,weights)
  scores=np.concatenate([(gradient.numpy()*abs_init*mask).reshape(-1) for gradient,abs_init,mask in zip(gradients,abs_inits,masks)])
  true_masks=[scores[counts_sum[layer]:counts_sum[layer+1]].reshape(shapes[layer])!=0 for layer in range(len(tensors))]
  return true_masks

def effective_masks_custom(model_name,masks):
  if model_name=='LeNet-300-100':
    return effective_masks_denseonly(masks)
  elif 'ResNet' in model_name:
    return effective_masks_residual(model_name,masks)
  else:
    return effective_masks_convolutional(masks)

def effective_masks_denseonly(masks):
  units=[mask.shape[-2] for mask in masks]+[masks[-1].shape[-1]]
  next_layer=np.ones(units[-1])
  way_out=[next_layer]
  for mask in masks[::-1]:
    curr_mask=np.matmul(mask,next_layer.reshape((len(next_layer),1)))
    next_layer=np.sum(curr_mask,axis=1)>0
    way_out.append(next_layer)
  way_out=way_out[::-1]
  prev_layer=np.ones(units[0])
  way_in=[prev_layer]
  for mask in masks:
    curr_mask=np.matmul(prev_layer.reshape((1,len(prev_layer))),mask)
    prev_layer=np.sum(curr_mask,axis=0)>0
    way_in.append(prev_layer)
  activity=[w_in*w_out for w_in,w_out in zip(way_in,way_out)]
  effective_masks=[mask*np.matmul(activity[i].reshape((len(activity[i]),1)),activity[i+1].reshape((1,len(activity[i+1])))) for i,mask in enumerate(masks)]
  return effective_masks

def effective_masks_convolutional(masks):
  conv=[1]+[1 if len(mask.shape)>2 else 0 for mask in masks]+[0]
  next_layer=np.ones(masks[-1].shape[-1])
  way_out=[next_layer]
  for curr in range(len(masks)-1,-1,-1):
    if not conv[curr+1]:
      curr_mask=np.matmul(masks[curr],next_layer.reshape((len(next_layer),1)))
      next_layer=np.sum(curr_mask,axis=1)>0
      way_out.append(next_layer)
      if conv[curr]:
        channels=masks[curr-1].shape[3]
        area=len(next_layer)//channels
        next_layer=np.array([sum(next_layer[[channel+channels*i for i in range(area)]])>0 for channel in range(channels)])
        way_out.append(next_layer)
    else:
      mask=np.sum(masks[curr],axis=(0,1))>0
      curr_mask=np.matmul(mask,next_layer.reshape((len(next_layer),1)))
      next_layer=np.sum(curr_mask,axis=1)>0
      way_out.append(next_layer)
  way_out=way_out[::-1]
  prev_layer=np.ones(masks[0].shape[2])
  way_in=[prev_layer]
  for curr in range(1,len(masks)+1):
    if conv[curr]:
      mask=np.sum(masks[curr-1],axis=(0,1))>0
      curr_mask=np.matmul(prev_layer.reshape((1,len(prev_layer))),mask)
      prev_layer=np.sum(curr_mask,axis=0)>0
      way_in.append(prev_layer)
      if not conv[curr+1]:
        channels=masks[curr-1].shape[3]
        area=masks[curr].shape[0]//channels
        prev_layer=np.tile(prev_layer,area)
        way_in.append(prev_layer)
    else:
      curr_mask=np.matmul(prev_layer.reshape((1,len(prev_layer))),masks[curr-1])
      prev_layer=np.sum(curr_mask,axis=0)>0
      way_in.append(prev_layer)
  activity=[w_in*w_out for w_in,w_out in zip(way_in,way_out)]
  conv=np.array(conv)
  effective_masks_conv=[masks[i]*np.repeat(np.repeat(np.matmul(activity[i].reshape((len(activity[i]),1)),activity[i+1].reshape((1,len(activity[i+1]))))[np.newaxis,:,:],masks[i].shape[1],axis=0)[np.newaxis,:,:,:],masks[i].shape[0],axis=0) for i in np.where(conv==1)[0][:-1]]
  effective_masks_dense=[masks[i-1]*np.matmul(activity[i].reshape((len(activity[i]),1)),activity[i+1].reshape((1,len(activity[i+1])))) for i in np.where(conv==0)[0][:-1]]
  return effective_masks_conv+effective_masks_dense

def effective_masks_residual(model_name,masks):
  conv=[1]+[1 if len(mask.shape)>2 else 0 for mask in masks]+[0]
  if model_name=='ResNet-18':
    masks_shortcut=[np.ones(64),np.ones(64),masks[7],np.ones(128),masks[12],np.ones(256),masks[17],np.ones(512)] if model_name=='ResNet-18' else [np.ones(5),np.ones(5),masks[7],np.ones(6),masks[12],np.ones(7),masks[17],np.ones(8)]
    original_shortcut_masks=[masks[7],masks[12],masks[17]]
    masks=masks[:7]+masks[8:12]+masks[13:17]+masks[18:]
    conv=[1]+[1 if len(mask.shape)>2 else 0 for mask in masks]+[0]
    forward_support=[0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,0,0,0]
    backward_support=[0,0,0,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0]
    residual_skip,first_backward_supporter=2,1
    layer_shortcut_inds=[(5,8),(9,12),(13,16)]
    shortcut_masks_positions=[7,12,17]
  next_layer=np.ones(masks[-1].shape[-1])
  way_out=[next_layer]
  way_out_shortcuts=[]
  for curr in range(len(masks)-1,-1,-1):
    if not conv[curr+1]:
      curr_mask=np.matmul(masks[curr],next_layer.reshape((len(next_layer),1)))
      next_layer=np.sum(curr_mask,axis=1)>0
      way_out.append(next_layer)
      if conv[curr]:
        channels=masks[curr-1].shape[3]
        area=len(next_layer)//channels
        next_layer=np.array([sum(next_layer[[channel+channels*i for i in range(area)]])>0 for channel in range(channels)])
        way_out.append(next_layer)
    else:
      mask=np.sum(masks[curr],axis=(0,1))>0
      curr_mask=np.matmul(mask,next_layer.reshape((len(next_layer),1)))
      curr_mask=np.squeeze(curr_mask)
      supporter_mask=np.zeros(curr_mask.shape)
      if forward_support[curr]:
        supporter=way_out[-residual_skip]
        supporter_mask=masks_shortcut[sum(forward_support[:curr])]
        if len(supporter_mask.shape)>1:
          way_out_shortcuts.append(supporter)
          supporter_mask=np.sum(supporter_mask,axis=(0,1))
          supporter_mask=supporter_mask*np.tile(supporter,supporter_mask.shape[0]).reshape(supporter_mask.shape)
          supporter_mask=np.sum(supporter_mask,axis=1)
        else:
          supporter_mask=supporter_mask*supporter
      next_layer=(curr_mask+supporter_mask)>0
      way_out.append(next_layer)
  way_out=way_out[::-1]
  way_out_shortcuts=way_out_shortcuts[::-1]
  prev_layer=np.ones(masks[0].shape[2])
  way_in=[prev_layer]
  way_in_shortcuts=[]
  for curr in range(1,len(masks)+1):
    if len(way_in)==first_backward_supporter+1:
      supporter_mask_global=way_in[-1]
    if conv[curr]:
      mask=np.sum(masks[curr-1],axis=(0,1))>0
      if backward_support[curr]:
        supporter_mask=masks_shortcut[sum(backward_support[:curr])]
        if len(supporter_mask.shape)>1:
          supporter_mask=np.sum(supporter_mask,axis=(0,1))
          supporter_mask_global=np.matmul(supporter_mask_global.reshape(1,len(supporter_mask_global)),supporter_mask)
          supporter_mask_global=np.squeeze(supporter_mask_global)
        supporter_mask_global=(supporter_mask_global+prev_layer)>0
        prev_layer=np.matmul((supporter_mask_global).reshape(1,len(supporter_mask_global)),mask)
        if sum(backward_support[:curr])<len(masks_shortcut) and len(masks_shortcut[sum(backward_support[:curr])+1].shape)>1:
          way_in_shortcuts.append(supporter_mask_global)
      else:
        prev_layer=np.matmul((prev_layer).reshape(1,len(prev_layer)),mask)
      prev_layer=np.squeeze(prev_layer)>0
      way_in.append(prev_layer)
      if not conv[curr+1]:
        channels=masks[curr-1].shape[3]
        area=masks[curr].shape[0]//channels
        if backward_support[curr+1]:
          supporter_mask=masks_shortcut[sum(backward_support[:curr+1])]
          if len(supporter_mask.shape)>1:
            supporter_mask=np.sum(supporter_mask,axis=(0,1))
            supporter_mask_global=np.matmul(supporter_mask_global.reshape(1,len(supporter_mask_global)),supporter_mask)
            supporter_mask_global=np.squeeze(supporter_mask_global)
          prev_layer=supporter_mask_global=(supporter_mask_global+prev_layer)>0
        prev_layer=np.tile(prev_layer,area)
        way_in.append(prev_layer)
    else:
      curr_mask=np.matmul(prev_layer.reshape((1,len(prev_layer))),masks[curr-1])
      prev_layer=np.squeeze(curr_mask)>0
      way_in.append(prev_layer)
  activity=[w_in*w_out for w_in,w_out in zip(way_in,way_out)]
  shortcut_activity={first_backward_supporter:activity[first_backward_supporter]}
  global_supporter_activity=activity[first_backward_supporter]
  forward_support[max([i for i,val in enumerate(forward_support) if val==1])+residual_skip]=1
  for i,supporter in enumerate(forward_support[:-1]):
    if supporter and i>1 and conv[i]:
      supporter_mask=masks_shortcut[sum(forward_support[:i])-1]
      if len(supporter_mask.shape)>1:
        supporter_mask=np.sum(supporter_mask,axis=(0,1))
        global_supporter_activity=np.matmul(global_supporter_activity.reshape(1,len(global_supporter_activity)),supporter_mask)
        global_supporter_activity=np.squeeze(global_supporter_activity)
      global_supporter_activity=(global_supporter_activity+activity[i])>0
      shortcut_activity[i]=global_supporter_activity
      if not conv[i+1]:
        shortcut_activity[i+1]=np.tile(global_supporter_activity,area)
    else:
      shortcut_activity[i]=np.zeros(activity[i].shape)
  conv=np.array(conv)
  effective_masks_conv=[masks[i]*np.repeat(np.repeat(np.matmul((activity[i]+shortcut_activity[i]).reshape((len(activity[i]),1)),activity[i+1].reshape((1,len(activity[i+1]))))[np.newaxis,:,:],masks[i].shape[1],axis=0)[np.newaxis,:,:,:],masks[i].shape[0],axis=0) for i in np.where(conv==1)[0][:-1]]
  effective_masks_dense=[masks[i-1]*np.matmul((activity[i]+shortcut_activity[i]).reshape((len(activity[i]),1)),activity[i+1].reshape((1,len(activity[i+1])))) for i in np.where(conv==0)[0][:-1]]
  real_effective_masks=effective_masks_conv+effective_masks_dense
  shortcut_effective_masks=[original_shortcut_masks[i]*np.expand_dims(np.expand_dims(np.matmul(np.expand_dims(_in,1),np.expand_dims(_out,0)),0),0) for i,(_out,_in) in enumerate(zip(way_out_shortcuts,way_in_shortcuts))]
  for i,s_mask in zip(shortcut_masks_positions,shortcut_effective_masks):
    real_effective_masks.insert(i,s_mask)
  return real_effective_masks