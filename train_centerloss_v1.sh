#!/bin/bash
mkdir log
time=`date +"%m-%d-%H-%M"`

two_steps=200

first_step=160
second_step=240
total_epoch=270
num_centers=64
lr=0.02
dropout=0.8
optim=SGD
sampling_method=tsn
timesteps=25
log_id=17
lossweight=0.01
prefix=ucf101_bnin_${optim}_t${timesteps}_k${num_centers}_lr${lr}_d${dropout}_e${first_step}${second_step}${total_epoch}_f${two_steps}_soft_${sampling_method}_lossweight${lossweight}

srun --mpi=pmi2 --gres=gpu:8 -n1 -p SenseMediaA --ntasks-per-node=1 --job-name=activa python /mnt/lustre/xuyoujiang/action/seqvlad-pytorch/main_with_centerloss.py ucf101 RGB ./data/ucf101_splits/rgb/train_split1.txt ./data/ucf101_splits/rgb/test_split1.txt \
   --arch BNInception \
   --timesteps ${timesteps} --num_centers ${num_centers} --redu_dim 512 \
   --gd 20 --lr ${lr} --lr_steps ${first_step} ${second_step} --epochs ${total_epoch} \
   -b 64 -j 8 --dropout ${dropout} \
   --snapshot_pref models/rgb/${prefix} \
   --sources /mnt/lustre/xuyoujiang/data/action/ucf-101/UCF-101-frames \
   --resume ucf101_bninception_rgb_model_best.pth.tar \
   --resume_type tsn --two_steps ${two_steps} \
   --activation softmax \
   --optim ${optim} \
   --sampling_method ${sampling_method} \
   --lossweight ${lossweight} \
   2>&1 | tee ./log/logid${log_id}_${time}.txt
