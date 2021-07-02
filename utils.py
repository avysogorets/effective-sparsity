import numpy as np
import logging

def set_weights_layer(model,layer,weight,mask=None):
  mask=np.ones(weight.shape) if mask is None else mask
  if len(model.layers[layer].get_weights())==1:
      model.layers[layer].set_weights([weight*mask])
  else:
      model.layers[layer].set_weights((weight*mask,model.layers[layer].get_weights()[1]))
  
def set_weights_model(model,tensors,weights,masks=None):
  masks=[np.ones(weight.shape) for weight in weights] if masks is None else masks
  for weight,mask,layer in zip(weights,masks,tensors):
    set_weights_layer(model,layer,weight,mask=mask)

def get_direct_sparsity(matrices):
  s=[round(1-float(len(np.nonzero(matrix.reshape(-1))[0]))/np.prod(matrix.shape),9) for matrix in matrices] if len(matrices)>1 else matrices[0]
  return np.array(s)

def get_overall_direct_sparsity(matrices):
  layerwise=get_direct_sparsity(matrices)
  overall_sparsity=sum(i_1*i_2 for i_1,i_2 in zip(layerwise,[np.prod(matrix.shape) for matrix in matrices]))/sum([np.prod(matrix.shape) for matrix in matrices])
  return overall_sparsity

def check_valid_sparsities(sparsities):
  if sum([int(s<=1 and s>=0) for s in sparsities])<len(sparsities): logging.error(f"<utils> invalid sparsities {sparsities} encountered.")
  assert sum([int(s<=1 and s>=0) for s in sparsities])==len(sparsities),f"<utils> invalid sparsities {sparsities} encountered."
  return np.array(sparsities)
