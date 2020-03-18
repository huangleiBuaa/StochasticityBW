#!/usr/bin/env bash
cd "$(dirname $0)/.."
CUDA_VISIBLE_DEVICES=2,3 python3 imagenet.py \
-a=resnet50 \
--arch-cfg=last=True \
--batch-size=256 \
--epochs=100 \
-oo=sgd \
-oc=momentum=0.9 \
-wd=1e-4 \
--lr=0.1 \
--lr-method=cos \
--lr-step=100 \
--lr-gamma=0.00001 \
--dataset-root=/raid/Lei_Data/imageNet/input_torch/ \
--dataset=folder \
--norm=ItN \
--norm-cfg=T=5,num_channels=64 \
--seed=1 \
$@
