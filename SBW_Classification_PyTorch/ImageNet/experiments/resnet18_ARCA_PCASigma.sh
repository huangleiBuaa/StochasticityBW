#!/usr/bin/env bash
cd "$(dirname $0)/.."
CUDA_VISIBLE_DEVICES=7 python3 imagenet.py \
-a=resnet18 \
--batch-size=256 \
--epochs=100 \
-oo=sgd \
-oc=momentum=0.9 \
-wd=1e-4 \
--lr=0.1 \
--lr-method=step \
--lr-steps=30 \
--lr-gamma=0.1 \
--dataset-root=/raid/Lei_Data/imageNet/input_torch/ \
--dataset=folder \
--norm=PCASigma \
--norm-cfg=T=5,num_channels=16 \
--seed=1 \
$@
