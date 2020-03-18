#!/bin/bash
depths=(4)
methods=(DBN DBNSigma QR QRSigma)
NCs=(512)
lrs=(1)
seeds=(1)
T=5
NormMM=0.1
affine="False"
width=256
arc="MLP"
#batch_size=59999
batch_size=1024
epochs=120
oo="sgd"
momentum=0
wd=0
lrmethod="step"
lrstep=100
lrgamma=0.2
datasetroot="/home/ubuntu/leihuang/pytorch_work/data/"

l=${#depths[@]}
n=${#methods[@]}
m=${#NCs[@]}
t=${#lrs[@]}
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
                baseString="execute_${arc}_d${depths[$a]}_w${width}_${methods[$i]}_NC${NCs[$j]}_lr${lrs[$k]}_b${batch_size}_s${seeds[$b]}"
                fileName="${baseString}.sh"
   	            echo "${baseString}"
                touch "${fileName}"
                echo  "#!/usr/bin/env bash
cd \"\$(dirname \$0)/../..\" 
python3 mnist.py \\
-a=${arc} \\
--width=${width} \\
--depth=${depths[$a]} \\
--batch-size=${batch_size} \\
--epochs=${epochs} \\
-oo=${oo} \\
-oc=momentum=${momentum} \\
-wd=${wd} \\
--lr=${lrs[$k]} \\
--lr-method=${lrmethod} \\
--lr-step=${lrstep} \\
--lr-gamma=${lrgamma} \\
--dataset-root=${datasetroot} \\
--norm=${methods[$i]} \\
--norm-cfg=num_channels=${NCs[$j]},momentum=${NormMM},affine=${affine},dim=2 \\
--seed=${seeds[$b]} \\" >> ${fileName}
                echo  "nohup bash ${fileName} >output_${baseString}.out 2>&1 &" >> z_bash_excute.sh
           done
         done
      done
   done
done
