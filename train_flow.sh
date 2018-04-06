#!/bin/bash

python main.py ucf101 Flow ./data/ucf101_splits/flow/train_split1.txt ./data/ucf101_splits/flow/test_split1.txt \
   --arch BNInception \
   --timesteps 3 --num_centers 4 --redu_dim 512 \
   --gd 20 --lr 0.01 --lr_steps 190 300 --epochs 340 \
   -b 3 -j 8 --dropout 0.7 \
   --snapshot_pref ucf101_bninception_ --flow_pref flow_ \
   --sources /data/UCF-101-opticalflow/ucf101_flow_img_tvl1_gpu/ 
