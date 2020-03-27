#!/bin/bash
cd "$(dirname $0)/.." 
Gmethods=(cholesky_wm)
Gm=(0)
LRs=(2e-4 1e-4 1e-3)
BatchSize=(64)
seeds=(1)
Count=7
dataset="cifar10"
arc="dcgan"
generator_adversarial_objective="hinge"
discriminator_adversarial_objective="hinge"
discriminator_filters=256
generator_filters=256
generator_block_norm="d"
#generator_block_coloring="uconv"
generator_block_coloring="ucs"
generator_last_norm="d"
#generator_last_coloring="uconv"
generator_last_coloring="ucs"
g_iter_num=5
g_instance_norm=0
#discriminator_spectral=1
discriminator_spectral=1
discriminator_norm="n"
discriminator_coloring="n"
d_iter_num=5
d_instance_norm=0
gradient_penalty_weight=0
lr_decay_schedule="linear"
beta1=0.5
beta2=0.999
training_ratio=1
number_of_epochs=100
generator_batch_multiple=1

l=${#Gmethods[@]}
n=${#LRs[@]}
m=${#Gm[@]}
t=${#BatchSize[@]}
f=${#seeds[@]}

for ((a=0;a<$l;++a))
do 
   for ((i=0;i<$n;++i))
   do 
      for ((j=0;j<$m;++j))
      do	
        for ((k=0;k<$t;++k))
        do
          for ((b=0;b<$f;++b))
          do
                name="exp2_1_${dataset}_${arc}_D_sn${discriminator_spectral}_G${Gmethods[$a]}_${generator_block_coloring}_Gins${g_instance_norm}_GW${Gm[$j]}_f${discriminator_filters}_bs${BatchSize[$k]}_${LRs[$i]}_s${seeds[$b]}_C${Count}"
   	            echo "${name}"
CUDA_VISIBLE_DEVICES=${Count} python run.py --name $name --dataset ${dataset} --arc ${arc}\
 --generator_adversarial_objective ${generator_adversarial_objective} --discriminator_adversarial_objective ${discriminator_adversarial_objective}\
 --discriminator_filters ${discriminator_filters} --generator_filters ${generator_filters}\
 --discriminator_spectral ${discriminator_spectral}\
 --generator_block_norm ${generator_block_norm} --generator_block_coloring ${generator_block_coloring} --generator_last_norm ${generator_last_norm} --generator_last_coloring ${generator_last_coloring}\
 --g_decomposition ${Gmethods[$a]} --g_iter_num ${g_iter_num} --g_whitten_m ${Gm[$j]} --g_coloring_m 0 --g_instance_norm ${g_instance_norm}\
 --discriminator_norm ${discriminator_norm} --discriminator_coloring ${discriminator_coloring}\
 --d_decomposition ${Gmethods[$a]} --d_iter_num ${g_iter_num} --d_whitten_m ${Gm[$j]} --d_coloring_m ${Gm[$j]} --d_instance_norm ${d_instance_norm}\
 --gradient_penalty_weight ${gradient_penalty_weight}\
 --lr_decay_schedule ${lr_decay_schedule} --generator_lr ${LRs[$i]} --discriminator_lr ${LRs[$i]}\
 --beta1 ${beta1} --beta2 ${beta2}\
 --number_of_epochs ${number_of_epochs} --batch_size ${BatchSize[$k]}\
 --training_ratio ${training_ratio} --generator_batch_multiple ${generator_batch_multiple}
           done
         done
      done
   done
done
