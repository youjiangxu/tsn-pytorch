#!/bin/bash

python main.py ucf101 RGB ./data/ucf101_splits/rgb/train_split1.txt ./data/ucf101_splits/rgb/test_split1.txt \
   --arch BNInception \
   --timesteps 3 --num_centers 4 --redu_dim 512 --with_relu \
   --gd 20 --lr 0.001 --lr_steps 30 60 --epochs 80 \
   -b 2 -j 8 --dropout 0.8 \
   --snapshot_pref ucf101_bninception_ \
   --sources /data/UCF-101-frames/ \
   --activation softmax \
   --optim Adam \
    --two_steps 30 \
