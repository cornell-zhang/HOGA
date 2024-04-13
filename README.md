HOGA
===============================

[![arXiv](https://img.shields.io/badge/arXiv-2403.01317-b31b1b.svg)](https://arxiv.org/abs/2403.01317)

HOGA is an attention model for scalable and generalizable learning on circuits. By leveraging a novel gated attention module on hop-wise features,  HOGA not only outperforms prior graph learning models on challenging circuit problems, but is also friendly to distributed training by mitigating communication overhead caused by graph dependencies. This renders HOGA applicable to industrial-scale circuit applications. More details are available in [our paper](https://arxiv.org/abs/2403.01317).

| ![HOGA.png](/figures/HOGA.png) | 
|:--:| 
| Figure1: An overview of HOGA and gated attention module. |

Requirements
------------
* python 3.9
* pytorch 1.12 (CUDA 11.3)
* torch_geometric 2.1

Datasets
------------
### Pre-processed CSA and Booth Multipliers (for Gamora experiments)
Check at: https://huggingface.co/datasets/yucx0626/Gamora-CSA-Multiplier/tree/main
### Pre-processed OpenABC-D benchmark (for OpenABC-D experiments)
Check at: https://zenodo.org/records/6399454#.YkTglzwpA5k

Note
------------
More experiments and details will be provided soon.
