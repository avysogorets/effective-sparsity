## Effective-Sparsity

This repository contains code for the experimental section of the paper *[Connectivity Matters: Neural Network Pruning Through the Lens of Effective Sparsity](https://arxiv.org/user/)*.

#### Environment


#### Demonstration
Running full experiments can require thousands of network training cycles and can be costly. Our experiments involved 5 different network architectures (3 of which require a GPU) and 15 pruning algorithms run across about 30 sparsity levels, all repeated 3 times for stability of results. In total, this comes to almost 7,000 networks to be trained. Thus, we provide a ```demo.py``` file as a limited but lightweight demonstration of some key results of the paper. 
