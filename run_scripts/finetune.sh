#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES=0

echo "Using GPU $CUDA_VISIBLE_DEVICES..."

python -m src.bin.train \
    --model_name transformer \
    --config_path /your/path/to/config \
    --log_path /your/path/to/save/log \
    --saveto /your/path/to/save/model \
    --use_gpu \
    --shared_dir /tmp \
    --pretrain_path /your/path/to/pretrained/model \
    --reload