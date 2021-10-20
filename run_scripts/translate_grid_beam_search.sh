# domain: auto, education, network, phone
domain=auto
weight=your_weight

data_dir=/your/path/to/data
save_dir=/your/path/to/save/results
model_path=/your/path/to/model
config_path=/your/path/to/config
constraint_path=$data_dir/$domain.json

mkdir -p save_dir

src_tok=$data_dir/$domain.tok.zh
tgt_detok=$data_dir/$domain.en

hyp_tok=$save_dir/$domain.tok.en
hyp_detok=$save_dir/$domain.en

python -m src.bin.GBS_translate \
    --model_name transformer \
    --source_path $src_tok \
    --model_path $model_path \
    --config_path $config_path \
    --constraint_path $constraint_path \
    --beam_size 5 \
    --alpha 0 \
    --keep_n 1 \
    --weight $weight \
    --saveto $hyp_tok \
    --use_gpu

scripts=/fsa/home/hsj_zhuwh/tools/mosesdecoder/scripts
cat $hyp_tok.0 | $scripts/recaser/detruecase.perl | $scripts/tokenizer/detokenizer.perl -q -l en > $hyp_detok

sacrebleu -l zh-en -lc $tgt_detok < $hyp_detok