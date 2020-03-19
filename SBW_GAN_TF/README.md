# Experiments on GANs

### Requirments
* python3
* numpy
* scipy
* skimage
* pandas
* tensorflow >= 1.5.0


For the commands reproducing experiments from the paper check scripts folder.

All scripts has the following name: (name of the dataset) + (architecture type (resnet or dcgan)) +
(discriminator normalization (sn or wgan_gp)) + (conditional of unconditional) + (if conditional use soft assigment (sa)).

For example:

```CUDA_VISIBLE_DEVICES=0 scripts/cifar10_resnet_sn_uncond.sh```

will train GAN for cifar10 dataset, with resnet architecture, spectral normalized discriminator in unconditional case.


All dataset are downloaded and trained at the same time.


### Reference
Siarohin, A., Sangineto, E., Sebe, N.: Whitening and coloring transform for GANs.
In: ICLR (2019), https://openreview.net/forum?id=S1x2Fj0qKQ

