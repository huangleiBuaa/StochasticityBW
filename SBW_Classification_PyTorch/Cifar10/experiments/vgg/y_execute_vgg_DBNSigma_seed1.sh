#!/usr/bin/env bash
cd "$(dirname $0)/../.."
methods=(DBNSigma DBN)
seeds=(1)
NCs=(512 256 128 64 32 16)
Count=1
l=${#methods[@]}
n=${#seeds[@]}
m=${#NCs[@]}
for ((a=0;a<$l;++a))
do
    for ((i=0;i<$n;++i))
    do
        for ((j=0;j<$m;++j))
        do
CUDA_VISIBLE_DEVICES=${Count} python3 cifar10.py \
-a=vgg \
--batch-size=256 \
--epochs=160 \
-oo=sgd \
-oc=momentum=0.9 \
-wd=0 \
--lr=0.1 \
--lr-method=steps \
--lr-steps=60,120 \
--lr-gamma=0.2 \
--dataset-root=/raid/Lei_Data/dataset/cifar10-pytorch/ \
--norm=${methods[$a]} \
--norm-cfg=T=5,num_channels=${NCs[$j]} \
--seed=${seeds[$i]} \
$@
        done
    done
done
