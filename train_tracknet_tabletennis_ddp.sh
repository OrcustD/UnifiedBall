#!/bin/bash

# Set environment variables
export CUDA_VISIBLE_DEVICES=2,3,4,5,6,7

# Run the training script with DDP
nohup python -m torch.distributed.launch --nproc_per_node=6 train.py \
    --data_type UniBall \
    --data_dir data/tabletennis \
    --model_name TrackNet \
    --seq_len 3 \
    --epochs 50 \
    --batch_size 50 \
    --bg_mode concat \
    --alpha -1 \
    --save_dir exp_tt_010902 \
    --verbose \
    --last_only \
    --vis_step 50 \
    --sigma 2.5 \
    --heatmap_mode gaussian \
    > exp_tt_0109.log 2>&1 &