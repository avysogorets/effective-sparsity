import numpy as np
from models import *
import callbacks
import tensorflow as tf
from tensorflow.keras import models
from effective_masks import *
from utils import *
import logging

def lamp(weights,target_sparsity):
  shapes=[weight.shape for weight in weights]
  counts=[np.prod(shape) for shape in shapes]
  counts_sum=[sum(counts[:layer]) for layer in range(len(shapes)+1)]
  wsorted_squared=[sorted((weight**2).reshape(-1)) for weight in weights]
  partial_sums=[]
  for w in wsorted_squared:
    partial_sum=[0]
    for weight in w[::-1]:
      partial_sum.append(partial_sum[-1]+weight)
    partial_sums.append(partial_sum[:0:-1])
  scores=[np.array([w[i]/(p[i]+1e-8) for i in range(len(w))]) for w,p in zip(wsorted_squared,partial_sums)]
  scores=[score[np.argsort(np.argsort((weight**2).reshape(-1)))] for score,weight in zip(scores,weights)]
  scores=np.concatenate(scores)
  masks=np.ones((sum(counts)))
  masks[scores.argsort()[:int(target_sparsity*sum(counts))]]=0.
  masks=[masks[counts_sum[layer]:counts_sum[layer+1]].reshape(shapes[layer]) for layer in range(len(shapes))]
  return scores,masks

def magnitude_layerwise(weight,target_sparsity):
  shape=weight.shape
  flat_abs=np.abs(weight.reshape(-1))
  argsort=flat_abs.argsort()
  prune_count=int(round(len(flat_abs)*target_sparsity))
  mask=np.ones(flat_abs.shape)
  mask[argsort[:prune_count]]=0
  mask=mask.reshape(shape)
  return mask

def magnitude_global(weights,target_sparsity):
  shapes=[weight.shape for weight in weights]
  counts=[np.prod(shape) for shape in shapes]
  counts_sum=[sum(counts[:layer]) for layer in range(len(shapes)+1)]
  scores=np.concatenate([np.abs(weight.reshape(-1)) for weight in weights])
  masks=np.ones((sum(counts)))
  masks[scores.argsort()[:int(target_sparsity*sum(counts))]]=0.
  masks=[masks[counts_sum[layer]:counts_sum[layer+1]].reshape(shapes[layer]) for layer in range(len(shapes))]
  return scores,masks

def erk_quotas(target_sparsity,shapes,**kwargs):
  counts=[np.prod(shape) for shape in shapes]
  coeffs=[sum(shape)/counts[i] for i,shape in enumerate(shapes)]
  k=(sum(counts)*(1-target_sparsity))/sum([count*coeff for count,coeff in zip(counts,coeffs)])
  sparsities=[1-k*coeff for coeff in coeffs]
  return redistribute_invalid_quotas(sparsities,shapes)

def bs_force_igq(areas,Lengths,target_sparsity,tolerance,f_low,f_high):
  lengths_low=[Length/(f_low/area+1) for Length,area in zip(Lengths,areas)]
  overall_sparsity_low=1-sum(lengths_low)/sum(Lengths)
  if abs(overall_sparsity_low-target_sparsity)<tolerance:
    return [1-length/Length for length,Length in zip(lengths_low,Lengths)]
  lengths_high=[Length/(f_high/area+1) for Length,area in zip(Lengths,areas)]
  overall_sparsity_high=1-sum(lengths_high)/sum(Lengths)
  if abs(overall_sparsity_high-target_sparsity)<tolerance:
    return [1-length/Length for length,Length in zip(lengths_high,Lengths)]
  force=float(f_low+f_high)/2
  lengths=[Length/(force/area+1) for Length,area in zip(Lengths,areas)]
  overall_sparsity=1-sum(lengths)/sum(Lengths)
  f_low=force if overall_sparsity<target_sparsity else f_low
  f_high=force if overall_sparsity>target_sparsity else f_high
  return bs_force_igq(areas,Lengths,target_sparsity,tolerance,f_low,f_high)

def igq_quotas(target_sparsity,shapes,**kwargs):
  counts=[np.prod(shape) for shape in shapes]
  tolerance=1./sum(counts)
  areas=[1./count for count in counts]
  Lengths=[count for count in counts]
  return bs_force_igq(areas,Lengths,target_sparsity,tolerance,0,1e20)

def redistribute_invalid_quotas(sparsities,shapes,**kwargs):
  counts=[np.prod(shape) for shape in shapes]
  for layer in range(len(sparsities)-1,-1,-1):
    sparsity=sparsities[layer]
    if sparsity<0 and layer>0:
      sparsities[layer-1]=(counts[layer-1]*sparsities[layer-1]+sparsities[layer]*counts[layer])/counts[layer-1]
      sparsities[layer]=0
    elif sparsity<0 and abs(sparsity*counts[0])>2 and layer==0:
      logging.warning(f"<pruning> unable to redistribute density quotas: {sparsities}")
  return sparsities

def uniform_plus_quotas(target_sparsity,shapes,**kwargs):
  assert len(shapes[0])>2,"<pruning> uniform+ supports convolutional networks only."
  counts=[np.prod(shape) for shape in shapes]
  sparsity=target_sparsity*sum(counts)/sum(counts[1:])
  to_distribute=max([0,(sparsity-0.8)*counts[-1]])
  additional_sparsity=to_distribute/(sum(counts[1:-1]))
  return np.concatenate([0.],[sparsity+additional_sparsity]*(len(counts)-2),[min([sparsity,0.8])])

def effective_correction_from_global_scores(model,tensors,scores,target_sparsity):
  shapes=[model.layers[layer].get_weights()[0].shape for layer in tensors]
  counts=[np.prod(shape) for shape in shapes]
  counts_sum=[sum(counts[:layer]) for layer in range(len(shapes)+1)]
  low,high=0,sum(counts)
  while high-low>1:
    middle=(high+low)//2
    middle_masks=np.concatenate([np.ones(shape).reshape(-1) for shape in shapes])
    middle_masks[scores.argsort()[:middle]]=0.
    middle_masks=[middle_masks[counts_sum[layer]:counts_sum[layer+1]].reshape(shapes[layer]) for layer in range(len(shapes))]
    middle_effective_masks=effective_masks_synflow(model,tensors,middle_masks)
    middle_effective_sparsity=get_overall_direct_sparsity(middle_effective_masks)
    if middle_effective_sparsity<=target_sparsity:
      low,high=middle,high
    else:
      low,high=low,middle
  low_masks=np.concatenate([np.ones(shape).reshape(-1) for shape in shapes])
  low_masks[scores.argsort()[:low]]=0.
  low_masks=[low_masks[counts_sum[layer]:counts_sum[layer+1]].reshape(shapes[layer]) for layer in range(len(shapes))]
  return low_masks

def effective_correction_layerwise_scores_magnitude_pruning(model,tensors,func,scores,target_sparsity):
  shapes=[model.layers[layer].get_weights()[0].shape for layer in tensors]
  counts=[np.prod(shape) for shape in shapes]
  low,high,flag=0,sum(counts),False
  low_val=func(target_sparsity=low,shapes=shapes)
  while ((np.array(low_val)<0).any() or (np.array(low_val)>1).any()) and high-low>1: # checking the lowest achievable sparsity 
    flag=True
    middle=(low+high)//2
    middle_val=func(target_sparsity=middle/sum(counts),shapes=shapes)
    middle_sparsities=np.array(middle_val)
    if (middle_sparsities<0).any() or (middle_sparsities>1).any():
      low,high=middle,high
    else:
      low,high=low,middle
  low=actual_low=high if flag else 0
  high_val=sum(counts)
  flag=False
  while ((np.array(high_val)<0).any() or (np.array(high_val)>1).any()) and high-low>1: # checking the highest achievable sparsity
    flag=True
    middle=(low+high)//2
    middle_val=func(target_sparsity=middle/sum(counts),shapes=shapes)
    middle_sparsities=np.array(middle_val)
    if (middle_sparsities<0).any() or (middle_sparsities>1).any():
      low,high=low,middle
    else:
      low,high=middle,high
  high=actual_high=low if flag else sum(counts)
  low=actual_low
  if low>=high:
    logging.error(f"<pruning> low: ({low}) >= high ({high}). target sparsity might be incompatible with the pruning method.")
  low_val=func(target_sparsity=low/sum(counts),shapes=shapes)
  low_sparsities=np.array(low_val)
  low_masks=[magnitude_layerwise(score,sparsity) for score,sparsity in zip(scores,low_sparsities)]
  high_masks=[np.zeros(shape) for shape in shapes]
  while high-low>1:
    middle=(high+low)//2
    middle_val=func(target_sparsity=middle/sum(counts),shapes=shapes)
    middle_sparsities=np.array(middle_val)
    middle_masks=[magnitude_layerwise(score,sparsity) for score,sparsity in zip(scores,middle_sparsities)]
    effective_middle_masks=effective_masks_synflow(model,tensors,middle_masks)
    effective_middle_sparsity=get_overall_direct_sparsity(effective_middle_masks)
    if effective_middle_sparsity<=target_sparsity:
      low,high=middle,high
      low_masks,high_masks=middle_masks,high_masks
    else:
      low,high=low,middle
      low_masks,high_masks=low_masks,middle_masks
  masks=low_masks
  sparsities=func(target_sparsity=target_sparsity,shapes=shapes)
  logging.info(f'<pruning> requested: ({target_sparsity:.6f})')
  direct_masks=[magnitude_layerwise(score,sparsity) for score,sparsity in zip(scores,sparsities)]
  logging.info(f'<pruning> direct pruning: overall effective sparsity: {get_overall_direct_sparsity(effective_masks_synflow(model,tensors,direct_masks)):.6f}')
  logging.info(f'<pruning> effective pruning: overall effective sparsity: {get_overall_direct_sparsity(effective_masks_synflow(model,tensors,masks)):.6f}')
  return low_masks

def uniform_quotas(target_sparsity,shapes,**kwargs):
  return np.full(len(shapes),target_sparsity)

class Pruner(object):
  def __init__(self,mode):
    self.mode=mode
  def prune(self,model,tensors,sparsity,pruning_type,**kwargs):
    shapes=[model.layers[layer].get_weights()[0].shape for layer in tensors]
    if self.mode=='lamp':
      try:
        final_weights=np.load(f'{kwargs["path_to_dense"]}/{kwargs["sample"]}_1_final_weights.npy',allow_pickle=True)
        inits=np.load(f'{kwargs["path_to_dense"]}/{kwargs["sample"]}_1_inits.npy',allow_pickle=True)
      except:
        logging.error(f'<pruning> required files in ({kwargs["path_to_dense"]}/{kwargs["sample"]}) do not exist.')
        raise FileNotFoundError
      set_weights_model(model,tensors,inits)
      scores,masks=lamp(final_weights,sparsity)
      corrected_masks=effective_correction_from_global_scores(model,tensors,scores,sparsity)
      if pruning_type=='effective':
        return corrected_masks
      elif pruning_type=='direct':
        return masks
    elif self.mode.split('/')[0]=='magnitude':
      model_name=model.name.replace('-','').lower()
      try:
        final_weights=np.load(f'{kwargs["path_to_dense"]}/{kwargs["sample"]}_1_final_weights.npy',allow_pickle=True)
        inits=np.load(f'{kwargs["path_to_dense"]}/{kwargs["sample"]}_1_inits.npy',allow_pickle=True)
      except:
        logging.error(f'<pruning> required files in ({kwargs["path_to_dense"]}/{kwargs["sample"]}) do not exist.')
        raise FileNotFoundError
      if self.mode=='magnitude/erk':
        corrected_masks=effective_correction_layerwise_scores_magnitude_pruning(model,tensors,erk_quotas,abs(final_weights),sparsity)
        sparsities=check_valid_sparsities(erk_quotas(sparsity,shapes))
        masks=[magnitude_layerwise(final_weight,s) for final_weight,s in zip(final_weights,sparsities)]
      if self.mode=='magnitude/igq':
        corrected_masks=effective_correction_layerwise_scores_magnitude_pruning(model,tensors,igq_quotas,abs(final_weights),sparsity)
        sparsities=check_valid_sparsities(igq_quotas(sparsity,shapes))
        masks=[magnitude_layerwise(final_weight,s) for final_weight,s in zip(final_weights,sparsities)]
      if self.mode=='magnitude/uniform':
        corrected_masks=effective_correction_layerwise_scores_magnitude_pruning(model,tensors,uniform_quotas,abs(final_weights),sparsity)
        sparsities=check_valid_sparsities(uniform_quotas(sparsity,shapes))
        masks=[magnitude_layerwise(final_weight,s) for final_weight,s in zip(final_weights,sparsities)]
      if self.mode=='magnitude/uniform_plus':
        corrected_masks=effective_correction_layerwise_scores_magnitude_pruning(model,tensors,uniform_plus_quotas,abs(final_weights),sparsity)
        sparsities=check_valid_sparsities(uniform_plus_quotas(sparsity,shapes))
        masks=[magnitude_layerwise(final_weight,s) for final_weight,s in zip(final_weights,sparsities)]
      if self.mode=='magnitude/global':
        scores,masks=magnitude_global(final_weights,sparsity)
        corrected_masks=effective_correction_from_global_scores(model,tensors,scores,sparsity)
      set_weights_model(model,tensors,inits)
      if pruning_type=='effective':
        return corrected_masks
      elif pruning_type=='direct':
        return masks
    elif self.mode.split('/')[0]=='random':
      if self.mode=='random/snip':
        func=lambda **kwargs: get_direct_sparsity(self.prune_snip(**kwargs))
      if self.mode=='random/synflow':
        func=lambda **kwargs: get_direct_sparsity(self.prune_synflow(**kwargs))
      if self.mode=='random/uniform':
        func=uniform_quotas
      if self.mode=='random/igq':
        func=igq_quotas
      if self.mode=='random/erk':
        func=erk_quotas
      if self.mode=='random/uniform_plus':
        func=uniform_plus_quotas
      if pruning_type=='effective':
        return self.prune_random(model,tensors,'effective',target_sparsity=sparsity,func=func,shapes=shapes,**kwargs)
      elif pruning_type=='direct':
        sparsities=check_valid_sparsities(func(target_sparsity=sparsity,shapes=shapes,model=model,tensors=tensors,pruning_type='direct',**kwargs))
        return self.prune_random(model,tensors,'direct',sparsities=sparsities)
    elif self.mode=='snip/iterative':
      return self.prune_iterative_snip(model,tensors,sparsity,pruning_type,**kwargs)
    elif self.mode=='snip':
      return self.prune_snip(model,tensors,sparsity,pruning_type,**kwargs)
    elif self.mode=='synflow':
      return self.prune_synflow(model,tensors,sparsity,pruning_type,**kwargs)
    elif self.mode=='dense':
      return [np.ones(shape) for shape in shapes]
    else:
      logging.error(f'<pruning> unknown pruner "{self.mode}" encountered.')
      raise NotImplementedError

  def prune_synflow(self,model,tensors,target_sparsity,pruning_type,train_X,train_y,**kwargs):
    shapes=[model.layers[layer].get_weights()[0].shape for layer in tensors]
    masks=[np.ones(shape) for shape in shapes]
    counts=[np.prod(shape) for shape in shapes]
    counts_sum=[sum(counts[:layer]) for layer in range(len(shapes)+1)]
    linear_model,linear_tensors=get_model(shape=train_X[0].shape,architecture=model.name.replace('-','').lower(),batchnorm=False,activation=False,pool='average',output_classes=len(train_y[0]))
    abs_inits=[abs(model.layers[layer].get_weights()[0]) for layer in tensors]
    already_pruned,weight_scores=0,np.zeros(counts_sum[-1])
    for iteration in range(1,101):
      set_weights_model(linear_model,linear_tensors,abs_inits,masks=masks)
      to_prune=int(counts_sum[-1]-counts_sum[-1]*(1-target_sparsity)**(float(iteration)/100))-already_pruned
      already_pruned+=to_prune
      with tf.GradientTape(persistent=False) as tape:
        output=linear_model(np.ones([1]+linear_model.inputs[0].shape[1:]))
        saliency=tf.reduce_sum(output)
        weights=[linear_model.layers[layer].trainable_weights[0] for layer in linear_tensors]
      gradients=tape.gradient(saliency,weights)
      scores=np.concatenate([(gradient.numpy()*abs_init*mask).reshape(-1) for gradient,abs_init,mask in zip(gradients,abs_inits,masks)])
      masks=np.concatenate([mask.reshape(-1) for mask in masks])
      indices_to_prune=scores.argsort()[np.isin(scores.argsort(),np.where(masks==1)[0])][:to_prune]
      masks[indices_to_prune]=0.
      if len(indices_to_prune)>0:
        weight_scores[indices_to_prune]=iteration*2+(scores[indices_to_prune]-np.min(scores[indices_to_prune]))/(np.max(scores[indices_to_prune])-np.min(scores[indices_to_prune])+1e-10)
      masks=[masks[counts_sum[layer]:counts_sum[layer+1]].reshape(shapes[layer]) for layer in range(len(linear_tensors))]
      if iteration==100:
        last_batch=np.where(weight_scores==0)[0]
        weight_scores[last_batch]=(iteration+1)*2+(scores[last_batch]-np.min(scores[last_batch]))/(np.max(scores[last_batch])-np.min(scores[last_batch])+1e-10)
    if pruning_type=='effective':
      corrected_masks=effective_correction_from_global_scores(model,tensors,weight_scores,target_sparsity)
      logging.info(f'<pruning> direct pruning: overall effective sparsity: {get_overall_direct_sparsity(effective_masks_synflow(model,tensors,masks)):.6f}')
      logging.info(f'<pruning> effective pruning: overall effective sparsity: {get_overall_direct_sparsity(effective_masks_synflow(model,tensors,corrected_masks)):.6f}')
      return corrected_masks
    elif pruning_type=='direct':
      return masks

  def prune_snip(self,model,tensors,target_sparsity,pruning_type,train_X,train_y,config,**kwargs):
    inits=[model.layers[layer].get_weights()[0] for layer in tensors]
    shapes=[model.layers[layer].get_weights()[0].shape for layer in tensors]
    counts=[np.prod(shape) for shape in shapes]
    counts_sum=[sum(counts[:layer]) for layer in range(len(shapes)+1)]
    choice=np.random.choice(range(len(train_X)),size=min([config['batch_size_snip']*300,len(train_X)]),replace=False)
    batch_X,batch_y=train_X[choice],train_y[choice]
    masks=np.ones(counts_sum[-1])
    gradients={layer:[] for layer in tensors}
    for minibatch_X,minibatch_y in zip(np.split(batch_X,range(128,len(batch_X),128)),np.split(batch_y,range(128,len(batch_y),128))):
      with tf.GradientTape(persistent=False) as tape:
        output=model(minibatch_X)
        weights=[model.layers[layer].trainable_weights[0] for layer in tensors]
        loss=tf.reduce_mean(tf.keras.losses.categorical_crossentropy(output,minibatch_y))
      gradients_minibatch=tape.gradient(loss,weights)
      for gradient_minibatch,layer in zip(gradients_minibatch,tensors):
        gradients[layer].append(gradient_minibatch.numpy())
    for layer in tensors:
      gradients[layer]=np.mean(gradients[layer],axis=0)
    cs=np.concatenate([abs(gradients[layer]*init*mask).reshape(-1) for layer,init,mask in zip(tensors,inits,masks)])
    masks[cs.argsort()[:int(target_sparsity*counts_sum[-1])]]=0.
    masks=[masks[counts_sum[layer]:counts_sum[layer+1]].reshape(shapes[layer]) for layer in range(len(tensors))]
    if pruning_type=='effective':
      corrected_masks=effective_correction_from_global_scores(model,tensors,cs,target_sparsity)
      logging.info(f'<pruning> direct pruning: overall effective sparsity: {get_overall_direct_sparsity(effective_masks_synflow(model,tensors,masks)):.6f}')
      logging.info(f'<pruning> effective pruning: overall effective sparsity: {get_overall_direct_sparsity(effective_masks_synflow(model,tensors,corrected_masks)):.6f}')
      return corrected_masks
    elif pruning_type=='direct':
      return masks

  def prune_iterative_snip(self,model,tensors,target_sparsity,pruning_type,train_X,train_y,config,**kwargs):
    inits=[model.layers[layer].get_weights()[0] for layer in tensors]
    shapes=[init.shape for init in inits]
    masks=[np.ones(shape) for shape in shapes]
    counts=[np.prod(shape) for shape in shapes]
    counts_sum=[sum(counts[:layer]) for layer in range(len(shapes)+1)]
    scores,already_pruned=np.zeros(counts_sum[-1]),0
    for iteration in range(1,301):
      choice=np.random.choice(range(len(train_X)),size=min([config['batch_size_snip'],len(train_X)]),replace=False)
      batch_X,batch_y=train_X[choice],train_y[choice]
      to_prune=int(counts_sum[-1]-counts_sum[-1]*(1-target_sparsity)**(float(iteration)/300))-already_pruned
      already_pruned+=to_prune
      gradients={layer:[] for layer in tensors}
      set_weights_model(model,tensors,inits,masks=masks)
      for minibatch_X,minibatch_y in zip(np.split(batch_X,range(128,len(batch_X),128)),np.split(batch_y,range(128,len(batch_y),128))):
        with tf.GradientTape(persistent=False) as tape:
          output=model(minibatch_X)
          weights=[model.layers[layer].trainable_weights[0] for layer in tensors]
          loss=tf.reduce_mean(tf.keras.losses.categorical_crossentropy(output,minibatch_y))
        gradients_minibatch=tape.gradient(loss,weights)
        for gradient_minibatch,layer in zip(gradients_minibatch,tensors):
          gradients[layer].append(gradient_minibatch.numpy())
      for layer in tensors:
        gradients[layer]=np.mean(gradients[layer],axis=0)
      cs=np.concatenate([abs(gradients[layer]*init*mask).reshape(-1) for layer,init,mask in zip(tensors,inits,masks)])
      masks=np.concatenate([mask.reshape(-1) for mask in masks])
      indices_to_prune=cs.argsort()[np.isin(cs.argsort(),np.where(masks==1)[0])][:to_prune]
      masks[indices_to_prune]=0.
      if len(indices_to_prune)>0:
        scores[indices_to_prune]=iteration*2+(cs[indices_to_prune]-np.min(cs[indices_to_prune]))/(np.max(cs[indices_to_prune])-np.min(cs[indices_to_prune])+1e-10)
      masks=[masks[counts_sum[layer]:counts_sum[layer+1]].reshape(shapes[layer]) for layer in range(len(tensors))]
    scores[np.where(scores==0)[0]]=2*(iteration+2)
    set_weights_model(model,tensors,inits)
    if pruning_type=='effective':
      corrected_masks=effective_correction_from_global_scores(model,tensors,scores,target_sparsity)
      logging.info(f'<pruning> direct pruning: overall effective sparsity: {get_overall_direct_sparsity(effective_masks_synflow(model,tensors,masks)):.6f}')
      logging.info(f'<pruning> effective pruning: overall effective sparsity: {get_overall_direct_sparsity(effective_masks_synflow(model,tensors,corrected_masks)):.6f}')
      return corrected_masks
    elif pruning_type=='direct':
      return masks

  def prune_random(self,model,tensors,pruning_type,**kwargs):
    if pruning_type=='direct':
      sparsities=kwargs["sparsities"]
      masks=[np.ones(model.layers[layer].get_weights()[0].shape) for layer in tensors]
      inds=[np.random.choice(range(len(mask.reshape(-1))),size=int(sparsities[ind]*len(mask.reshape(-1))),replace=False) for ind,mask in enumerate(masks)]
      for ind,mask in enumerate(masks):
        mask.reshape(-1)[inds[ind]]=0.
    elif pruning_type=='effective':
      shapes=[model.layers[layer].get_weights()[0].shape for layer in tensors]
      counts=[np.prod(shape) for shape in shapes]
      target_sparsity=kwargs["target_sparsity"]
      del kwargs["target_sparsity"]
      low,high,func,flag=0,sum(counts),kwargs["func"],False
      low_val=func(model=model,tensors=tensors,pruning_type='direct',target_sparsity=low/sum(counts),**kwargs)
      while ((np.array(low_val)<0).any() or (np.array(low_val)>1).any()) and high-low>1:
        flag=True
        middle=(low+high)//2
        middle_val=func(model=model,tensors=tensors,pruning_type='direct',target_sparsity=middle/sum(counts),**kwargs)
        middle_sparsities=np.array(middle_val)
        if (middle_sparsities<0).any() or (middle_sparsities>1).any():
          low,high=middle,high
        else:
          low,high=low,middle
      low=high if flag else 0
      high=sum(counts)
      low_val=func(model=model,tensors=tensors,pruning_type='direct',target_sparsity=low/sum(counts),**kwargs)
      low_sparsities=np.array(low_val)
      low_masks=self.prune_random(model,tensors,'direct',sparsities=low_sparsities)
      high_masks=[np.zeros(shape) for shape in shapes]
      while high-low>1:
        middle=(high+low)//2
        middle_val=func(model=model,tensors=tensors,pruning_type='direct',target_sparsity=middle/sum(counts),**kwargs)
        middle_sparsities=np.array(middle_val)
        middle_to_prune=[round(s*count) for s,count in zip(middle_sparsities,counts)]
        indices_to_prune=[np.random.choice(np.where(np.logical_and(low_masks[layer].reshape(-1)==1,high_masks[layer].reshape(-1)==0))[0],size=min([max([0,int(middle_to_prune[layer]-np.sum(1-low_masks[layer]))]),len(np.where(np.logical_and(low_masks[layer].reshape(-1)==1,high_masks[layer].reshape(-1)==0))[0])]),replace=False) for layer in range(len(tensors))]
        middle_masks=[np.array(low_masks[layer]) for layer in range(len(tensors))]
        for i,middle_mask in enumerate(middle_masks):
          middle_mask.reshape(-1)[indices_to_prune[i]]=0
        effective_middle_masks=effective_masks_synflow(model,tensors,middle_masks)
        effective_middle_sparsity=get_overall_direct_sparsity(effective_middle_masks)
        if effective_middle_sparsity<=target_sparsity:
          low,high=middle,high
          low_masks,high_masks=middle_masks,high_masks
        else:
          low,high=low,middle
          low_masks,high_masks=low_masks,middle_masks
      masks=low_masks
      sparsities=func(model=model,tensors=tensors,pruning_type='direct',target_sparsity=target_sparsity,**kwargs)
      direct_masks=self.prune_random(model,tensors,'direct',sparsities=sparsities)
      logging.info(f'<pruning> direct pruning: overall effective sparsity: {get_overall_direct_sparsity(effective_masks_synflow(model,tensors,direct_masks)):.6f}')
      logging.info(f'<pruning> effective pruning: overall effective sparsity: {get_overall_direct_sparsity(effective_masks_synflow(model,tensors,masks)):.6f}')
    return masks
