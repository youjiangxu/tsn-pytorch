#!/bin/bash
#!/bin/bash
python test_models.py ucf101 RGB ./data/ucf101_splits/rgb/test_split1.txt \
    ucf101_bnin_tsnPretrained_twosteps_lr0.02_d0.8_e90_f150_onlySeqvlad_rgb_model_best.pth.tar \
    --arch BNInception \
    --save_scores twosteps_tsn_pretrain \
    --num_centers 64 \
    --timesteps 10 \
    --redu_dim 512 \
    --sources /data/UCF-101-frames/ \
    --with_relu \
    --activation softmax 


