# Pytorch Experiments
Pytorch implementation of the IterNorm-Sigma, which is described in the following paper:

**An Investigation of Stochasticity of Batch Whitening** 

Lei Huang, Lei Zhao, Yi Zhou, Fan Zhu, Li Liu, Ling Shao

*IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2020 (accepted).*


This project also provide the pytorch implementation of Decorrelated Batch Normalization (CVPR 2018, [arXiv:1804.08450](https://arxiv.org/abs/1804.08450)), and CD-decoposition based Whitening, more details please refer to the [Torch project](https://github.com/princeton-vl/DecorrelatedBN). 

## Requirements and Dependency
* Install [PyTorch](http://torch.ch) with CUDA (for GPU). (Experiments are validated on python 3.6.8 and pytorch-nightly 1.0.0)
* (For visualization if needed), install the dependency [visdom](https://github.com/facebookresearch/visdom) by:
```Bash
pip install visdom
 ```


## Experiments
 
 #### 1.  MLP on MNIST datasets:
 
 
 #### 2.  VGG-network on Cifar-10 datasets:
 
run the scripts in the `./cifar10/experiments/vgg`. Note that the dataset root dir should be altered by setting the para '--dataset-root', and the dataset is official pytorch cifar datset. 

If the dataset is not exist, the script will download it, under the conditioning that the `dataset-root` dir is existed

 

#### 3. ImageNet experiments.

run the scripts in the `./ImageNet/experiment`. Note that resnet18 experimetns are run on one GPU, and resnet-50 are run on 2 GPU in the scripts. 

Note that the dataset root dir should be altered by setting the para '--dataset-root'.
 and the dataset style is described as:
