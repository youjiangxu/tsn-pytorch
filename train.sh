#!/bin/bash

python main.py ucf101 RGB ../ActionVLAD/data/ucf101/train_test_lists/train_split1.txt ../ActionVLAD/data/ucf101/train_test_lists/test_split1.txt \
   --arch BNInception --num_segments 1 \
   --gd 20 --lr 0.001 --lr_steps 30 60 --epochs 80 \
   -b 1 -j 8 --dropout 0.8 \
   --snapshot_pref ucf101_bninception_ --source /data/UCF-101-frames 
