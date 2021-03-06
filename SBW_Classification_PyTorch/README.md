# Pytorch Experiments
This directory/project (Pytorch implementation) provides the discriminative classification experiments of the following paper:

**An Investigation of Stochasticity of Batch Whitening. Lei Huang, Lei Zhao, Yi Zhou, Fan Zhu, Li Liu, Ling Shao. CVPR 2020 (accepted)** 


We provide the Pytorch implementation of the four Batch Whitening (BW) methods: [PCA whitening](https://github.com/princeton-vl/DecorrelatedBN), [ZCA whitening](https://github.com/princeton-vl/DecorrelatedBN), [CD whitening](https://github.com/AliaksandrSiarohin/wc-gan) and [IterNorm Whitening](https://github.com/huangleiBuaa/IterNorm), and their different methods in estimation population statistics. Please refer to the `./extension/normalization/` for details. The scripts to run the experiments for Mnist, Cifar10 and ImageNet are in the corresponding directories.


## Requirements and Dependency
* Install [PyTorch](http://torch.ch) with CUDA (for GPU). (Experiments are validated on python 3.6.8 and pytorch 1.0.1)
* (For visualization if needed), install the dependency [visdom](https://github.com/facebookresearch/visdom) by:
```Bash
pip install visdom
 ```


## Experiments
 
 #### 1.  MLP on MNIST datasets:
 The scripts to reproduce the experiments are in `./Mnist/experiments/MLP_S_Train`  and `./Mnist/experiments/MLP_S_Inference`. For example, one can  run the ZCA whitening with covariance matrix estimation, on the 4 layer MLP with 256 neuron in each layer,  by following script (after `cd ./Mnist/experiments/MLP_S_Inference'): 
 ```Bash
bash execute_MLP_d4_w256_DBNSigma_NC512_lr1_b1024_s1.sh
 ```
 One can change different hyper-paramters to reproduce the experiments in the paper. 
 
 **Note: change the hyper-parameter `--dataset-root` to your Mnist dataset path. (we use the Pytorch official Mnist)**
 
 #### 2.  VGG-network on Cifar-10 datasets:
 
The scripts to reproduce the experiments are in `./Cifar10/experiments/vgg/`. For example, one can  run the ItN whitening with covariance matrix estimation,  by following script: 
  ```Bash
bash y_execute_vgg_DBNSigma_seed1.sh
 ```
**Note: 1) change the hyper-parameter `--dataset-root` to your Cifar10 dataset path (we use the Pytorch official Cifar10);
2) change the `CUDA_VISIBLE_DEVICES`, if your machine has less GPUs than the value. (e.g., you have to set the value as 0 if you have only one GPU on your machine).**

#### 3. ImageNet experiments.

The scripts to reproduce the experiments are in `./ImageNet/experiments/`. For example, one can  run the ItN whitening with covariance matrix estimation on 18 layer residual network (ARCC),  by following script: 
  ```Bash
bash resnet18_ARCC_ItNSigma.sh
 ```
**Note: 1) change the hyper-parameter `--dataset-root` to your ImageNet dataset path; 
2) change the `CUDA_VISIBLE_DEVICES`, if your machine has less GPUs than the value. (e.g., you have to set the value as 0 if you have only one GPU on your machine).**

## Acknowledgement

This pytorch implementation is based on the [IterNorm-pytorch](https://github.com/huangleiBuaa/IterNorm-pytorch)
