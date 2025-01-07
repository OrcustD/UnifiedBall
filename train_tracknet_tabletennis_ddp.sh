#!/bin/bash

# Set environment variables
export CUDA_VISIBLE_DEVICES=1,2,3,4

# Run the training script with DDP
nohup python -m torch.distributed.launch --nproc_per_node=4 --nproc_per_node=4 train.py \
    --data_type UniBall \
    --data_dir data/tabletennis \
    --model_name TrackNet \
    --seq_len 4 \
    --epochs 300 \
    --batch_size 50 \
    --bg_mode concat \
    --alpha -1 \
    --save_dir exp_tt_0106 \
    --verbose \
    --last_only \
    > exp_tt_0106.log 2>&1 &