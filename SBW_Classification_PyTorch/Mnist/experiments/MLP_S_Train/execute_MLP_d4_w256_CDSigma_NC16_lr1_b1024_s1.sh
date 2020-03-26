#!/usr/bin/env bash
cd "$(dirname $0)/../.." 
python3 mnist.py \
-a=MLP \
--width=256 \
--depth=4 \
--batch-size=1024 \
--epochs=30 \
-oo=sgd \
-oc=momentum=0 \
-wd=0 \
--lr=1 \
--lr-method=step \
--lr-step=100 \
--lr-gamma=0.2 \
--dataset-root=/home/ubuntu/leihuang/pytorch_work/data/ \
--norm=CDSigma \
--norm-cfg=num_channels=16,momentum=0.1,affine=False,dim=2 \
--seed=1 \
