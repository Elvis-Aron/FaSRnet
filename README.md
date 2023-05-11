# FaSRnet
## Introduction
 we propose a human pose estimation framework with refinements at feature and semantic levels. 
 We align auxiliary features with the features of current frame to reduce the loss caused by different feature distributions. 
 Then, an attention mechanism is used to fuse auxiliary features with current features. In term of semantic, 
 we utilize the difference information between adjacent heatmaps as auxiliary features to refine the current heatmaps.


## Environment
The code is developed using python 3.7, pytorch-11.1, and CUDA 11.3 on Ubuntu 22.04. 
For our experiments, we used 1 NVIDIA 3090 GPUs.


## Pre-trained models and related file downloads
[pretrained models](https://drive.google.com/file/d/1eD0oENp4_NKpodaTZs2P9c4TXjZ8nbdg/view?usp=sharing)  
[Related file](https://drive.google.com/file/d/1YV0-caNYlWoc_88nZLTnNXEZh_seM8CN/view?usp=share_link)  


## Data preparation
Please refer to [Data preparation](https://github.com/Pose-Group/DCPose/blob/main/docs/Installation.md)  


## Acknowledgement
Thanks for the baselines, we construct the code based on them:

[DCpose](https://github.com/Pose-Group/DCPose)  
[EDVR](https://github.com/xinntao/EDVR)  
