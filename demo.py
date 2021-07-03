import pruning
import numpy as np
import data
import math
import models
import tensorflow as tf
from effective_masks import effective_masks_synflow
from utils import get_overall_direct_sparsity
import matplotlib as mpl
import matplotlib.pyplot as plt
import os

os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
datagen,train_X,train_y,test_X,test_y=data.get_data('mnist')
model,tensors=models.get_model(shape=train_X[0].shape,architecture='lenet300100')
compressions=[10**(0.2*i) for i in range(5,26)]
sparsities=[1-1./compression for compression in compressions]
pruner_names=['random/igq'] # any random, snip, snip/iterative, or synflow.
pruners={key:pruning.Pruner(key) for key in pruner_names}
direct_compressions_direct_pruning={key:[] for key in pruner_names}
effective_compressions_direct_pruning={key:[] for key in pruner_names}
direct_compressions_effective_pruning={key:[] for key in pruner_names}
effective_compressions_effective_pruning={key:[] for key in pruner_names}
print('\n')
for sparsity in sparsities:
  kwargs_direct_pruning={'model':model,'tensors':tensors,'sparsity':sparsity,'pruning_type':'direct','train_X':train_X,'train_y':train_y,'config':{'batch_size_snip':128}}
  direct_masks_direct_pruning={key:pruners[key].prune(**kwargs_direct_pruning) for key in pruner_names}
  effective_masks_direct_pruning={key:effective_masks_synflow(model,tensors,direct_masks_direct_pruning[key]) for key in pruner_names}
  kwargs_effective_pruning={'model':model,'tensors':tensors,'sparsity':sparsity,'pruning_type':'effective','train_X':train_X,'train_y':train_y,'config':{'batch_size_snip':128}}
  direct_masks_effective_pruning={key:pruners[key].prune(**kwargs_effective_pruning) for key in pruner_names}
  effective_masks_effective_pruning={key:effective_masks_synflow(model,tensors,direct_masks_effective_pruning[key]) for key in pruner_names}
  for key in pruner_names:
    print(f'================================ [target sparsity: {sparsity:.6f}] ================================')
    print(f'<demo> {key} direct pruning    [direct sparsity: {get_overall_direct_sparsity(direct_masks_direct_pruning[key]):.6f}][effective sparsity: {get_overall_direct_sparsity(effective_masks_direct_pruning[key]):.6f}]')
    print(f'<demo> {key} effective pruning [direct sparsity: {get_overall_direct_sparsity(direct_masks_effective_pruning[key]):.6f}][effective sparsity: {get_overall_direct_sparsity(effective_masks_effective_pruning[key]):.6f}]')
    if get_overall_direct_sparsity(effective_masks_direct_pruning[key])==1: effective_compressions_direct_pruning[key].append(math.inf)
    else: effective_compressions_direct_pruning[key].append(1./(1-get_overall_direct_sparsity(effective_masks_direct_pruning[key])))
    if get_overall_direct_sparsity(effective_masks_effective_pruning[key])==1: effective_compressions_effective_pruning[key].append(math.inf)
    else: effective_compressions_effective_pruning[key].append(1./(1-get_overall_direct_sparsity(effective_masks_effective_pruning[key])))
colors={pruner_name:mpl.cm.rainbow(_) for pruner_name,_ in zip(pruner_names,np.linspace(0,1,len(pruner_names)))}
for key in pruner_names:
  plt.plot(compressions,effective_compressions_direct_pruning[key],color=colors[key],linestyle='dashed',linewidth=3)
  plt.plot(compressions,effective_compressions_effective_pruning[key],color=colors[key],linestyle='solid',linewidth=3)
legend_elements=[mpl.patches.Patch(facecolor=colors[key],label=key) for key in pruner_names]
legend_elements.append(mpl.lines.Line2D([0],[0],color='k',linestyle='dashed',label='direct pruning'))
legend_elements.append(mpl.lines.Line2D([0],[0],color='k',linestyle='solid',label='effective pruning'))
plt.title('LeNet-300-100',fontsize='medium')
plt.xlabel("Direct compression",fontsize='medium')
plt.ylabel(f"Effective compression",fontsize='medium')
plt.legend(handles=legend_elements,loc='lower right',markerscale=1,fontsize='medium')
plt.xscale('log',base=10)
plt.yscale('log',base=10)
plt.grid(zorder=0,ls='dashed',alpha=0.6)
plt.show()