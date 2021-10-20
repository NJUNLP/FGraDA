#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES=0,1,2,3

echo "Using GPU $CUDA_VISIBLE_DEVICES..."

python -m src.distributed.launch \
    --nproc_per_node=4 \
    --nnodes=1 --node_rank=0 \
    --master_addr="127.0.0.1" --master_port=1234 \
    src.bin.train \
    --model_name transformer \
    --config_path /your/path/to/config \
    --log_path /your/path/to/save/log \
    --saveto /your/path/to/save/model \
    --use_gpu \
    --shared_dir /tmp \
    --reload