# Experiments on GANs

### Requirments
* python3
* numpy
* scipy
* skimage
* pandas
* tensorflow >= 1.5.0

The commands to reproduce experiments are under the "scripts" folder, for example:

```CUDA_VISIBLE_DEVICES=0 scripts/Exp2_Stability/hinge/y_execute_script_exp2_1_A_cifar10_dcgan_sn1_ItN.sh```

will train GAN using hinge loss on cifar10 dataset, with dcgan architecture, spectral normalized discriminator in unconditional case, and using iter-norm do to the whitening.


The dataset is downloaded and trained at the same time.


### Reference
Siarohin, A., Sangineto, E., Sebe, N.: Whitening and coloring transform for GANs.
In: ICLR (2019), https://openreview.net/forum?id=S1x2Fj0qKQ

