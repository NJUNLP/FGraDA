#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES=0

echo "Using GPU $CUDA_VISIBLE_DEVICES..."

domain=target_domain
data_dir=/your/path/to/data
save_dir=/your/path/to/save/results
model_path=/your/path/to/model
config_path=/your/path/to/config

mkdir -p save_dir

src_tok=$data_dir/$domain.tok.zh
tgt_detok=$data_dir/$domain.en

hyp_tok=$save_dir/$domain.tok.en
hyp_detok=$save_dir/$domain.en

python -m src.bin.translate \
    --model_name transformer \
    --source_path $src_tok \
    --model_path $model_path \
    --config_path $config_path \
    --beam_size 5 \
    --alpha 0 \
    --keep_n 1 \
    --saveto $hyp_tok \
    --use_gpu


scripts=/your/path/to/mosesdecoder/scripts
cat $hyp_tok.0 | $scripts/recaser/detruecase.perl | $scripts/tokenizer/detokenizer.perl -q -l en > $hyp_detok

sacrebleu -l zh-en -lc $tgt_detok < $hyp_detok