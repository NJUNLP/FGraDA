tok=$1
lang=$2
detok=$3
scripts=../src/metric/scripts

cat $tok | $scripts/recaser/detruecase.perl | $scripts/tokenizer/detokenizer.perl -q -l $lang > $detok