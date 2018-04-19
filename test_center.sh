#!/bin/bash

num_centers=64
test_segments=1
seqvlad_type=seqvlad
#seqvlad_type=bidirect
#seqvlad_type=unshare_bidirect
timesteps=15
pref=${seqvlad_type}_t${timesteps}_"d0.8_e160240270_centerloss0.02"
srun --mpi=pmi2 --gres=gpu:1 -n1 -p SenseMediaA --job-name=test python test_models_centers.py ucf101 RGB ./data/ucf101_splits/rgb/test_split1.txt \
    models/rgb/ucf101_bnin_SGD_t15_k64_lr0.02_d0.8_e160240270_f200_soft_tsn_lossweight0.01_rgb_model_best.pth.tar \
    --arch BNInception \
    --save_scores seqvlad_rgb_k${num_centers}_s${test_segments}_${pref} \
    --num_centers ${num_centers} \
    --timesteps ${timesteps} \
    --redu_dim 512 \
    --sources ../../data/action/ucf-101/UCF-101-frames/ \
    --activation softmax \
    --seqvlad_type ${seqvlad_type} \
    --test_segments ${test_segments}
