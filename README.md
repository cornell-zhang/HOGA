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
The implementation of hop-wise feature generation is available [here](https://github.com/cornell-zhang/HOGA/blob/b9dc53bc5df60369a34a8c79ca6015b10946e5f3/utils.py#L23). The model (i.e., hop-wise gated attention) implementation is available [here](https://github.com/cornell-zhang/HOGA/blob/b9dc53bc5df60369a34a8c79ca6015b10946e5f3/model.py#L60). You can adjust them for your own tasks. 

Citation
------------
If you use HOGA in your research, please cite our work
published in DAC'24.

```
@inproceedings{deng2024hoga,
  title={Less is More: Hop-Wise Graph Attention for Scalable and Generalizable Learning on Circuits},
  author={Chenhui Deng and Zichao Yue and Cunxi Yu and Gokce Sarar and Ryan Carey and Rajeev Jain and Zhiru Zhang},
  booktitle={DAC},
  year={2024},
}
```

