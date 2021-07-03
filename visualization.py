import matplotlib as mpl
import numpy as np
import os
import math
import argparse
import matplotlib.pyplot as plt
fig,ax=plt.subplots(nrows=1,ncols=1,figsize=[6,4])

parser=argparse.ArgumentParser()
parser.add_argument('--architecture',type=str,default='lenet300100',help='network type (choose one of: lenet300100, lenet5, vgg16, vgg19, resnet18)')
parser.add_argument('--num_samples',type=int,default=3,help='number of seeds available')
parser.add_argument('--pruners_to_display',nargs='+',default=[],help='select pruners for display (choose one or more of: lamp, snip, snip/iterative, synflow, random/uniform, random/erk, random/igq, random/uniform_plus, random/synflow, magnitude/global, magnitude/uniform, magnitude/erk, magnitude/igq, magnitude/uniform_plus)')
parser.add_argument('--out_path',type=str,default='EffectiveSparsity',help='path to the folder with output files')
parser.add_argument('--pruning_type',type=str,default='effective',help='pruning type used to generate output files (choose one of: direct, effective)')
parser.add_argument('--effective_type',type=str,default='synflow',help='method to compute effective compression (choose one of: custom, synflow)')
parser.add_argument('--plot_type',type=str,default='accuracies',help='plot type (choose one of: accuracies, compressions)')
args=parser.parse_args()
if not os.path.exists(os.path.join(args.out_path,args.architecture,'figures')):
    os.makedirs(os.path.join(args.out_path,args.architecture,'figures'))

pruners=['random/erk','random/igq','magnitude/global','snip','snip/iterative','random/snip','synflow','random/synflow','random/uniform','lamp','magnitude/pistons','magnitude/uniform','magnitude/erk','magnitude/uniform_plus','random/uniform_plus']
network_names={'lenet300100':'LeNet-300-100','lenet5':'LeNet-5','vgg16':'VGG-16','vgg19':'VGG-19','resnet18':'ResNet-18'}
pruner_names=['ERK (random)','IGQ (random)','Global (magnitude)','SNIP','SNIP-iterative','SNIP (random)','SynFlow','SynFlow (random)','Uniform (random)','LAMP',"IGQ (magnitude)","Uniform (magnitude)","ERK (magnitude)","Uniform+ (magnitude)","Uniform+ (random)"]
pruner_names={pruner:name for pruner,name in zip(pruners,pruner_names)}
color_map={'random/uniform_plus':'blue','random/snip':'blue','lamp':'aqua','synflow':'orange','snip':'blue','random/uniform':'mediumorchid','snip/iterative':'cyan','random/erk':'green','random/igq':'red','magnitude/uniform_plus':'blue','magnitude/global':'gold','random/synflow':'orange','magnitude/uniform':'mediumorchid','magnitude/igq':'red','magnitude/erk':'green'}
random={'lenet300100':10,'lenet5':10,'vgg16':10,'vgg19':100,'resent18':200}
linestyle_direct='dashed'
linestyle_effective='solid'

def accuracies(args):
    """ plots accuracies against effective and direct compression
    """
    legend_elements=[]
    compressions={pruner:set([]) for pruner in args.pruners_to_display}
    load_prefix={pruner:os.path.join(args.out_path,args.architecture,pruner,args.pruning_type) for pruner in args.pruners_to_display}
    for pruner in args.pruners_to_display:
        for filename in os.listdir(load_prefix[pruner]):
            if os.path.isfile(os.path.join(load_prefix[pruner],filename)):
                compression=int(filename.split('_')[1])
                if 'accuracies.npy' in filename:
                    compressions[pruner].add(compression)
    compressions={pruner:sorted(list(compressions[pruner])) for pruner in args.pruners_to_display}
    counts=np.load(os.path.join(args.out_path,args.architecture,'dense','counts.npy'))
    accuracies={pruner:np.array([[np.load(os.path.join(load_prefix[pruner],f'{sample}_{compression}_accuracies.npy'))[-1] for compression in compressions[pruner]] for sample in range(args.num_samples)]) for pruner in args.pruners_to_display}
    direct_sparsities={pruner:np.array([[np.load(os.path.join(load_prefix[pruner],f'{sample}_{compression}_sparsities_direct.npy')) for compression in compressions[pruner]] for sample in range(args.num_samples)]) for pruner in args.pruners_to_display}
    effective_sparsities={pruner:np.array([[np.load(os.path.join(load_prefix[pruner],f'{sample}_{compression}_sparsities_effective_{args.effective_type}.npy')) for compression in compressions[pruner]] for sample in range(args.num_samples)]) for pruner in args.pruners_to_display}
    overall_direct_sparsity={pruner:np.array([[sum([sparsity*count for sparsity,count in zip(direct_sparsities[pruner][sample][level],counts)])/sum(counts) for level in range(len(compressions[pruner]))] for sample in range(args.num_samples)]) for pruner in args.pruners_to_display}
    overall_effective_sparsity={pruner:np.array([[sum([sparsity*count for sparsity,count in zip(effective_sparsities[pruner][sample][level],counts)])/sum(counts) for level in range(len(compressions[pruner]))] for sample in range(args.num_samples)]) for pruner in args.pruners_to_display}
    overall_direct_compression={pruner:1./(1-overall_direct_sparsity[pruner]) for pruner in args.pruners_to_display}
    overall_effective_compression={pruner:1./(1-overall_effective_sparsity[pruner]) for pruner in args.pruners_to_display}
    dense=np.array([np.load(os.path.join(args.out_path,args.architecture,'dense',f'{sample}_1_accuracies.npy'))[-1] for sample in range(args.num_samples)])
    dense_min,dense_mean,dense_max=np.min(dense),np.mean(dense),np.max(dense)
    ticks={pruner:10**(0.05*np.arange(20,20*np.log10(compressions[pruner][-1]),20/9)) for pruner in args.pruners_to_display}
    accuracies_direct_sorted={pruner:[accuracies[pruner][sample][overall_direct_compression[pruner][sample].argsort()] for sample in range(args.num_samples)] for pruner in args.pruners_to_display}
    overall_direct_compression={pruner:[sorted(overall_direct_compression[pruner][sample]) for sample in range(args.num_samples)] for pruner in args.pruners_to_display}
    inf_idxs_direct={pruner:[np.where(np.array(overall_direct_compression[pruner][sample])>1e20)[0][0] if sum([int(direct_compression>1e20) for direct_compression in overall_direct_compression[pruner][sample]])>0 else len(overall_direct_compression[pruner][sample]) for sample in range(args.num_samples)] for pruner in args.pruners_to_display}
    for pruner in args.pruners_to_display:
        means,mins,maxs,start_idx,end_idx=[],[],[],math.inf,len(ticks[pruner])
        for tick_idx,tick in enumerate(ticks[pruner]):
            curr_accs=[]
            for sample in range(args.num_samples):
                if len(np.where(np.array(overall_direct_compression[pruner][sample])<=tick)[0])>0:
                    start=np.where(np.array(overall_direct_compression[pruner][sample])<=tick)[0][-1]
                    start_idx=min([start_idx,tick_idx])
                    if start+1<inf_idxs_direct[pruner][sample]:
                        coeff_start=(tick-overall_direct_compression[pruner][sample][start])/(overall_direct_compression[pruner][sample][start+1]-overall_direct_compression[pruner][sample][start])
                        coeff_end=(overall_direct_compression[pruner][sample][start+1]-tick)/(overall_direct_compression[pruner][sample][start+1]-overall_direct_compression[pruner][sample][start])
                        curr_accs.append((accuracies_direct_sorted[pruner][sample][start]*coeff_end+accuracies_direct_sorted[pruner][sample][start+1]*coeff_start))
                    else:
                        curr_accs.append(random[args.architecture])
                        end_idx=min([end_idx,tick_idx])
                else:
                    curr_accs.append(random[args.architecture])
                    end_idx=min([end_idx,tick_idx])
            assert len(curr_accs)>0,"no accuracy data for given ticks."
            mins.append(np.min(curr_accs))
            means.append(np.mean(curr_accs))
            maxs.append(np.max(curr_accs))
        mins,means,maxs=np.array(mins),np.array(means),np.array(maxs)
        ax.plot(ticks[pruner][start_idx:end_idx],means[start_idx:end_idx],zorder=3,color=color_map[pruner],linewidth=2.5,linestyle=linestyle_direct)
        ax.fill_between(ticks[pruner][start_idx:end_idx],mins[start_idx:end_idx],maxs[start_idx:end_idx],color=color_map[pruner],zorder=3,alpha=0.3)
        legend_elements+=[mpl.patches.Patch(facecolor=color_map[pruner],label=pruner_names[pruner])]
    accuracies_effective_sorted={pruner:[accuracies[pruner][sample][overall_effective_compression[pruner][sample].argsort()] for sample in range(args.num_samples)] for pruner in args.pruners_to_display}
    overall_effective_compression={pruner:[sorted(overall_effective_compression[pruner][sample]) for sample in range(args.num_samples)] for pruner in args.pruners_to_display}
    inf_idxs_effective={pruner:[np.where(np.array(overall_effective_compression[pruner][sample])>1e20)[0][0] if sum([int(effective_compression>1e20) for effective_compression in overall_effective_compression[pruner][sample]])>0 else len(overall_effective_compression[pruner][sample]) for sample in range(args.num_samples)] for pruner in args.pruners_to_display}
    for pruner in args.pruners_to_display:
        means,mins,maxs,start_idx,end_idx=[],[],[],math.inf,len(ticks[pruner])
        for tick_idx,tick in enumerate(ticks[pruner]):
            curr_accs=[]
            for sample in range(args.num_samples):
                if len(np.where(np.array(overall_effective_compression[pruner][sample])<=tick)[0])>0:
                    start=np.where(np.array(overall_effective_compression[pruner][sample])<=tick)[0][-1]
                    start_idx=min([start_idx,tick_idx])
                    if start+1<inf_idxs_effective[pruner][sample]:
                        coeff_start=(tick-overall_effective_compression[pruner][sample][start])/(overall_effective_compression[pruner][sample][start+1]-overall_effective_compression[pruner][sample][start])
                        coeff_end=(overall_effective_compression[pruner][sample][start+1]-tick)/(overall_effective_compression[pruner][sample][start+1]-overall_effective_compression[pruner][sample][start])
                        curr_accs.append((accuracies_effective_sorted[pruner][sample][start]*coeff_end+accuracies_effective_sorted[pruner][sample][start+1]*coeff_start))
                    else:
                        curr_accs.append(random[args.architecture])
                        end_idx=min([end_idx,tick_idx])
                else:
                    curr_accs.append(random[args.architecture])
                    end_idx=min([end_idx,tick_idx])
            assert len(curr_accs)>0,"no accuracy data for given ticks."
            mins.append(np.min(curr_accs))
            means.append(np.mean(curr_accs))
            maxs.append(np.max(curr_accs))
        ax.scatter(overall_effective_compression[pruner][sample],accuracies_effective_sorted[pruner][sample],color=color_map[pruner],zorder=3,s=25,alpha=0.3)
        mins,means,maxs=np.array(mins),np.array(means),np.array(maxs)
        ax.plot(ticks[pruner][start_idx:end_idx],means[start_idx:end_idx],zorder=3,color=color_map[pruner],linewidth=2.5,linestyle=linestyle_effective)
        ax.fill_between(ticks[pruner][start_idx:end_idx],mins[start_idx:end_idx],maxs[start_idx:end_idx],color=color_map[pruner],zorder=3,alpha=0.3)
    ax.axhline(dense_mean,linestyle='solid',color='grey',zorder=3,linewidth=2.5)
    ax.fill_between(plt.xlim(),dense_min,dense_max,color='grey',zorder=3,alpha=0.3)
    legend_elements+=[mpl.lines.Line2D([0],[0],color='grey',linestyle='solid',label='Dense',linewidth=2.5)]
    legend_elements+=[mpl.lines.Line2D([0],[0],color='k',linestyle='solid',label='vs. effective',linewidth=2.5)]
    legend_elements+=[mpl.lines.Line2D([0],[0],color='k',linestyle='dashed',label='vs. direct',linewidth=2.5)]
    ax.set_title(network_names[args.architecture],fontsize='medium')
    ax.set_xlabel("Compression",fontsize='medium')
    ax.set_ylabel(f"Top-1 test accuracy",fontsize='medium')
    ax.tick_params(axis='both',labelsize='medium')
    ax.legend(handles=legend_elements,loc='upper right',markerscale=1,fontsize='medium')
    ax.set_xscale('log',base=10)
    ax.grid(zorder=0,ls='dashed',alpha=0.6)
    fig.tight_layout()
    plt.savefig(os.path.join(args.out_path,args.architecture,'figures','accuracies.png'),dpi=600)

def compressions(args):
    """ plots effective compression against target compression
    """
    legend_elements=[]
    compressions={pruner:set([]) for pruner in args.pruners_to_display}
    load_prefix={pruner:os.path.join(args.out_path,args.architecture,pruner,args.pruning_type) for pruner in args.pruners_to_display}
    for pruner in args.pruners_to_display:
        for filename in os.listdir(load_prefix[pruner]):
            if os.path.isfile(os.path.join(load_prefix[pruner],filename)):
                compression=int(filename.split('_')[1])
                if '.npy' in filename:
                    compressions[pruner].add(compression)
    compressions={pruner:sorted(list(compressions[pruner])) for pruner in args.pruners_to_display}
    counts=np.load(os.path.join(args.out_path,args.architecture,'dense','counts.npy'))
    effective_sparsities={pruner:np.array([[np.load(os.path.join(load_prefix[pruner],f'{sample}_{compression}_sparsities_effective_{args.effective_type}.npy')) for compression in compressions[pruner]] for sample in range(args.num_samples)]) for pruner in args.pruners_to_display}
    overall_effective_sparsity={pruner:np.array([[sum([sparsity*count for sparsity,count in zip(effective_sparsities[pruner][sample][level],counts)])/sum(counts) for level in range(len(compressions[pruner]))] for sample in range(args.num_samples)]) for pruner in args.pruners_to_display}
    overall_effective_compression={pruner:1./(1-overall_effective_sparsity[pruner]) for pruner in args.pruners_to_display}
    inf_idxs_effective={pruner:np.array([np.where(np.array(overall_effective_compression[pruner][sample])>1e20)[0][0] if sum([int(effective_compression>1e20) for effective_compression in overall_effective_compression[pruner][sample]])>0 else len(overall_effective_compression[pruner][sample]) for sample in range(args.num_samples)]) for pruner in args.pruners_to_display}
    overall_effective_compression={pruner:overall_effective_compression[pruner][inf_idxs_effective[pruner].argsort()[-args.num_samples//2:]] for pruner in args.pruners_to_display}
    overall_effective_compression_min={pruner:np.min(overall_effective_compression[pruner],axis=0) for pruner in args.pruners_to_display}
    overall_effective_compression_mean={pruner:np.mean(overall_effective_compression[pruner],axis=0) for pruner in args.pruners_to_display}
    overall_effective_compression_max={pruner:np.max(overall_effective_compression[pruner],axis=0) for pruner in args.pruners_to_display}
    inf_idxs_effective={pruner:min([np.where(np.array(overall_effective_compression[pruner][sample])>1e20)[0][0] if sum([int(effective_compression>1e20) for effective_compression in overall_effective_compression[pruner][sample]])>0 else len(overall_effective_compression[pruner][sample]) for sample in range(args.num_samples//2+int(args.num_samples>2))]) for pruner in args.pruners_to_display}
    for pruner in args.pruners_to_display:
        ax.plot(compressions[pruner][:inf_idxs_effective[pruner]],overall_effective_compression_mean[pruner][:inf_idxs_effective[pruner]],color=color_map[pruner],linewidth=2.5)
        ax.fill_between(compressions[pruner][:inf_idxs_effective[pruner]],overall_effective_compression_min[pruner][:inf_idxs_effective[pruner]],overall_effective_compression_max[pruner][:inf_idxs_effective[pruner]],color=color_map[pruner],alpha=0.3)
        legend_elements+=[mpl.patches.Patch(facecolor=color_map[pruner],label=pruner_names[pruner])]
    ax.set_title(network_names[args.architecture],fontsize='medium')
    ax.set_xlabel(f"Target compression ({args.pruning_type} pruning)",fontsize='medium')
    ax.set_ylabel(f"Effective compression",fontsize='medium')
    ax.tick_params(axis='both',labelsize='medium')
    ax.legend(handles=legend_elements,loc='lower right',markerscale=1,fontsize='medium')
    ax.set_xscale('log',base=10)
    ax.set_yscale('log',base=10)
    ax.grid(zorder=0,ls='dashed',alpha=0.6)
    fig.tight_layout()
    plt.savefig(os.path.join(args.out_path,args.architecture,'figures','compressions.png'),dpi=600)

if args.plot_type=='accuracies':
  accuracies(args)
if args.plot_type=='compressions':
  compressions(args)
print(f'<visualzation> plot saved to {os.path.join(args.out_path,args.architecture,"figures")}')
