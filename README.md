# C2F-FWN
data/code repository of "C2F-FWN: Coarse-to-Fine Flow Warping Network for Spatial-Temporal Consistent Motion Transfer"  
(https://arxiv.org/abs/2012.08976)

## News
2020.12.16: Our paper is available on [[ArXiv]](https://arxiv.org/abs/2012.08976) now!  
2020.12.28: Our SoloDance Dataset is available on [[google drive]](https://drive.google.com/drive/folders/1f6NEO1onLtf-K65bpms4_alBlNh5YIVW?usp=sharing) and [[baidu pan (extraction code:gle4]](https://pan.baidu.com/s/14oitDhULAeirGaojV_VYew) now!  
2020.12.28: A preview version of our code is now available, which needs further clean-up.

## Example Results
- motion transfer videos
<p align='left'>
  <img src='imgs/motion transfer.gif' width='640'/>
</p>

- multi-source appearance attribute editing videos
<p align='left'>
  <img src='imgs/appearance control.gif' width='640'/>
</p>

- Full supplementary video:
https://youtu.be/THuQN1GXuGI

## Prerequisites
- Ubuntu
- Python 3
- NVIDIA GPU (>12GB memory) + CUDA10 cuDNN7
- PyTorch 1.0.0
### Other Dependencies
#### DConv (modified from original [[DConv]](https://github.com/chengdazhi/Deformable-Convolution-V2-PyTorch))
    cd models/dconv
    bash make.sh
#### FlowNet_v2 (directly ported from the original [[flownet2]](https://github.com/NVIDIA/flownet2-pytorch) following the steps described in [[vid2vid]](https://github.com/NVIDIA/vid2vid))
    cd models/flownet2-pytorch
    bash install.sh

## Getting Started
It's a preview version of our source code. We will clean it up in the near future.  
### Notes
1. Main functions for training and testing can be found in "train_stage1.py", "train_stage2.py", "train_stage2.py", "test_all_stages.py";
2. Data preprocessings of all the stages can be found in "data" folder;
3. Model definitions of all the stages can be found in "models" folder;
4. Training and testing options can be found in "options" folder;
5. Training and testing scripts can be found in "scripts" folder;
6. Tool functions can be found in "util" folder.

### Data Preparation
Download all the data packages from [[google drive]](https://drive.google.com/drive/folders/1f6NEO1onLtf-K65bpms4_alBlNh5YIVW?usp=sharing) or [[baidu pan (extraction code:gle4]](https://pan.baidu.com/s/14oitDhULAeirGaojV_VYew), and uncompress them.
You should create a directory named 'SoloDance' in the root (i.e., 'C2F-FWN') of this project, and then put 'train' and 'test' folders to 'SoloDance' you just created.
The structure should look like this:  
-C2F-FWN  
---SoloDance  
------train  
------test  

### Training
#### 1.Train the layout GAN of stage 1:
    bash scripts/stage1/train_1.sh
#### 2.Train our C2F-FWN of stage 2:
    bash scripts/stage2/train_2_tps_only.sh
    bash scripts/stage2/train_2.sh
#### 3.Train the composition GAN of stage 3:
    bash scripts/stage3/train_3.sh
    
### Testing all the stages together (separate testing scripts for different stages will be updated in the near future)
    bash scripts/full/test_full.sh

## Acknowledgement
A large part of the code is borrowed from [NVIDIA/vid2vid](https://github.com/NVIDIA/vid2vid). Thanks for their wonderful works.

## Citation
If you find this project useful for your research, please cite our paper using the following BibTeX entry.
```
@article{wei2020c2f,
  title={C2F-FWN: Coarse-to-Fine Flow Warping Network for Spatial-Temporal Consistent Motion Transfer},
  author={Wei, Dongxu and Xu, Xiaowei and Shen, Haibin and Huang, Kejie},
  journal={arXiv preprint arXiv:2012.08976},
  year={2020}
}
```
