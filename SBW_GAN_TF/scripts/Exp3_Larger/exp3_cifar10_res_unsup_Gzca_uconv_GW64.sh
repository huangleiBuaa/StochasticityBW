#!/usr/bin/env bash
cd "$(dirname $0)/.." 
name=$(basename $0)
base_name=${name%.*}
name="${base_name}"
CUDA_VISIBLE_DEVICES=0 python run.py --name $name --dataset cifar10 --arc res\
 --generator_adversarial_objective hinge --discriminator_adversarial_objective hinge\
 --discriminator_filters 128 --generator_filters 256\
 --discriminator_spectral 1\
 --generator_block_norm d --generator_block_coloring uconv --generator_last_norm d --generator_last_coloring uconv\
 --g_decomposition zca --g_iter_num 7 --g_whitten_m 64 --g_coloring_m 0 --g_instance_norm 0\
 --discriminator_norm n --discriminator_coloring n\
 --d_decomposition zca --d_iter_num 7 --d_whitten_m 64 --d_coloring_m 64 --d_instance_norm 0\
 --gradient_penalty_weight 0\
 --lr_decay_schedule linear --generator_lr 2e-4 --discriminator_lr 2e-4 \
 --beta1 0 --beta2 0.9\
 --number_of_epochs 50 --batch_size 64\
 --training_ratio 5 --generator_batch_multiple 2 \
