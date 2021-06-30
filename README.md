## Effective Sparsity

This repository contains code for the experimental section of the paper *[Connectivity Matters: Neural Network Pruning Through the Lens of Effective Sparsity](https://arxiv.org/user/)*.

#### Environment
To run code from this project, first clone the repository and install dependencies by executing ```pip install -r requirements.txt```. The code was tested for Python-3.6 and Python-3.8 with TensorFlow-2.0.0.

#### Quick demonstration
Our original full-fledged experiments involved 5 different network architectures (3 of which require a GPU) and 15 pruning algorithms combined across about 30 sparsity levels and repeated 3 times for stability of results. In total, this comes to almost 7,000 networks to be trained. For demonstration purposes, we provide a quick but limited lightweight demonstration ```demo.py``` that requires no flags or arguments and is easily executable on a CPU-powered device.

#### Original experiments
To replicate our results, run ```python main.py``` with a selection of arguments (run ```python main.py --help``` for description). For example,
```python main.py --sample=0 --architecture=lenet300100 --data=mnist --pruner=snip --target_sparsity=0.99 --pruning_type=effective --train=1``` will use SNIP to prune LeNet-300-100 to 99% effective sparsity and train this subnetwork on MNIST. Alternatively, you can choose to specify target compression by passing ```--com_exp=n```, which should result in ```10^(com_exp)``` compression (direct or effective, depending on ```pruning_type```). This flag will overwrite ```--target_sparsity``` if specified. 

#### Visualizing results
The ```visualization.py``` file produces graphs out of data generated in the above step. Execute ```python visualization.py --help``` for the full list of flags and their description. To generate plots for LeNet-300-100 effectively pruned by SNIP, SynFlow, and IGQ (random) and repeated for 3 samples, run ```python visualization.py --architecture=lenet300100 --pruners_to_display=snip synflow random/igq --num_samples=3 --pruning_type=effective```. Plots are saved to ```--out_path/--architecture/figures```.
