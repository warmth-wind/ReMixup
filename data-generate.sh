#!/usr/bin/env bash
set -x
set -e
#---------------------------------------

src=en
tgt=fr

DATA_PATH=data/${src}-${tgt}/
MODEL_PATH=train/${src}-${tgt}
#export CUDA_VISIBLE_DEVICES=0

python3 scripts/average_checkpoints.py \
    --inputs ${MODEL_PATH}/ckpt \
    --num-epoch-checkpoints 10  \
    --output ${MODEL_PATH}/ckpt/model.pt

python3 fairseq_cli/generate.py $DATA_PATH \
    --path ${MODEL_PATH}/ckpt/model.pt \
    --source-lang $src --target-lang $tgt \
    --num-workers 12 \
    --batch-size 128 \
    --beam 5 --remove-bpe >> ${MODEL_PATH}/result.gen

bash scripts/compound_split_bleu.sh ${MODEL_PATH}/result.gen
