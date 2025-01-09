#!/bin/bash

# Define the arguments
DATA_TYPE="UniBall"
DATA_DIR="data/tabletennis"
MODEL_NAME="TrackNet"
SEQ_LEN="4"
EPOCHS="300"
BATCH_SIZE="10"
BG_MODE="concat"
ALPHA="-1"
SAVE_DIR=$1
VERBOSE="--verbose"
DEBUG="--debug"
LAST_ONLY="--last_only"
RESUME_TRAINING="--resume_training"

# Set the environment variable
export CUDA_VISIBLE_DEVICES="0"

# Run the Python command
python eval.py \
    --data_type $DATA_TYPE \
    --data_dir $DATA_DIR \
    --model_name $MODEL_NAME \
    --seq_len $SEQ_LEN \
    --epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --bg_mode $BG_MODE \
    --alpha $ALPHA \
    --save_dir $SAVE_DIR \
    $VERBOSE \
    $DEBUG \
    $LAST_ONLY \
    $RESUME_TRAINING