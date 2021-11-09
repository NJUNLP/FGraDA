## FGraDA
-----
Resources and code for our paper "[FGraDA: A Dataset and Benchmark for Fine-Grained Domain Adaptation in Machine Translation](https://arxiv.org/abs/2012.15717)". This project implements several baselines used in our paper. The implementation is build upon [NJUNMT](https://github.com/whr94621/NJUNMT-pytorch).
Please cite our paper if you find this repository helpful in your research:

```latex
@article{zhu2021fgrada,
  title={FDMT: A Benchmark Dataset for Fine-grained Domain Adaptation in Machine Translation},
  author={Zhu, Wenhao and Huang, Shujian and Pu, Tong and Huang, Pingxuan and Zhang, Xu and Yu, Jian and Chen, Wei and Wang, Yanfeng and Chen, Jiajun},
  journal={arXiv preprint arXiv:2012.15717},
  year={2021}
}
```

### Requirements
* python==3.8.10
* pytorch==1.6.0
* PyYAML==5.4.1
* tensorboardX=2.4.0
* sacrebleu==2.0.0

### Instructions
We use an example to show how to run our codes.

#### Data
For convenience, We provide both raw data and pre-processed data of FGraDA, which can be download [here](https://drive.google.com/file/d/1vZjidCuBX1r_rqn2zBL-FS1Pe3TMWu73/view?usp=sharing).

#### Train Base Model
```bash
bash ../run_scripts/train.sh
```

#### Finetune Model on Parallel Data
```bash
bash ../run_scrpts/finetune.sh
```

#### Inference with Grid Beam Search
To prepare for grid beam search, you need to run ./scripts/build_constraint.py to generate the json file before runing the following script.
```bash
bash ../run_scrpts/translate_beam_search.sh
```
We recommend you to use below weight hyper-parameter to replicate results of Dict<sub>GBS</sub> and Wiki<sub>BT</sub>+Dict<sub>GBS</sub>. 
|  Model                                |  AV   | AIE  | RTN  | SP   |
|  ----                                 | ----  | ---- | ---- | ---- |
| Dict<sub>GBS</sub>                    | 0.3   | 0.35 | 0.15 | 0.35 | 
| Wiki<sub>BT</sub>+Dict<sub>GBS</sub>  | 0.4   | 0.25 | 0.05 | 0.35 |

#### Inference with Beam search
```bash 
bash ../run_scrpts/translate_grid_beam_search.sh
```