import os


if __name__ == "__main__":
    os.system(
    'python speed.py --dataset cifar10 --arc dcgan\
    --generator_adversarial_objective hinge --discriminator_adversarial_objective hinge\
    --discriminator_filters 512 --generator_filters 512\
    --discriminator_spectral 1\
    --generator_block_norm d --generator_block_coloring uconv --generator_last_norm d --generator_last_coloring uconv\
    --g_decomposition iter_norm --g_iter_num 5 --g_whitten_m 0 --g_coloring_m 0 --g_instance_norm 0\
    --discriminator_norm n --discriminator_coloring n\
    --d_decomposition iter_norm --d_iter_num 5 --d_whitten_m 0 --d_coloring_m 0 --d_instance_norm 0\
    --gradient_penalty_weight 0\
    --lr_decay_schedule linear --generator_lr 2e-4 --discriminator_lr 2e-4 \
    --beta1 0 --beta2 0.9\
    --number_of_epochs 100 --batch_size 64\
    --training_ratio 1 --generator_batch_multiple 1')
