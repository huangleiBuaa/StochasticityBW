# GANs Experiments
This directory/project (Tensorflow implementation) provides the GANs experiments of the following paper:

**An Investigation of Stochasticity of Batch Whitening. Lei Huang, Lei Zhao, Yi Zhou, Fan Zhu, Li Liu, Ling Shao. CVPR 2020 (accepted)** 


We provide the Tensorflow implementation of the four Batch whitening methods: 
[PCA whitening](https://github.com/princeton-vl/DecorrelatedBN), 
[ZCA whitening](https://github.com/princeton-vl/DecorrelatedBN), 
[CD whitening](https://github.com/AliaksandrSiarohin/wc-gan) 
and [IterNorm Whitening](https://github.com/huangleiBuaa/IterNorm), 
and their different methods in estimation population statistics. 
Please refer to the `./gan/layers/normalization.py` for details. 


## Requirements and Dependency
* python3
* numpy
* scipy
* skimage
* pandas
* tensorflow (Experiments are validated on tensorflow-gpu 1.12 on Linux)

## Experiments
 The scripts to run the experiments are under `.scripts/`, and all the experiments are on cifar10.
 #### 1.  Stability Experiments:
 The scripts to reproduce the experiments are in `./scripts/stability/`. 
 For example, one can train GAN using hinge loss, 
 dcgan architecture, 
 spectral normalized discriminator, 
 and iter-norm with scale&shift. 
 ```Bash
bash y_execute_script_exp2_1_A_cifar10_dcgan_sn1_ItN_ucs.sh
 ```
 
 Note:  change the `CUDA_VISIBLE_DEVICES`, if your machine has less 8 GPU. (e.g., you set the vale as 0 if you have only one GPU on your machine).

 #### 2.  On Larger Architectures:
   The scripts to reproduce the experiments are in `./scripts/larger_arch/`.
   For example, one can train GAN with larger architecture using 
    dcgan, 
    zca whitening with group size of 64. 
 ```Bash
bash exp3_cifar10_dcgan_unsup_Gzca_uconv_GW64.sh
 ```
 
 Note:  change the `CUDA_VISIBLE_DEVICES`, if your machine has less 8 GPU. (e.g., you set the vale as 0 if you have only one GPU on your machine).

 #### 3.  Speed Test
   Speed test for larger architecture models are in `./scripts/speed/`,
   just run the script with the corresponding name.



## Acknowledgement
The implementation is modified from:

Siarohin, A., Sangineto, E., Sebe, N.: Whitening and coloring transform for GANs.
In: ICLR (2019), https://openreview.net/forum?id=S1x2Fj0qKQ

