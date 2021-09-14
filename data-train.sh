#!/usr/bin/env bash
set -x
set -e
#---------------------------------------

src=en
tgt=fr

DATA_PATH=data/${src}-${tgt}/
MODEL_PATH=train/${src}-${tgt}
mkdir -p $MODEL_PATH
nvidia-smi

python3 train.py $DATA_PATH \
    --user-dir examples/translation_rdrop/translation_rdrop_src/ \
    --task rdrop_translation \
    --arch transformer_iwslt_de_en \
    --share-all-embeddings \
    --optimizer adam --lr 0.001 -s $src -t $tgt \
    --label-smoothing 0.1 --dropout 0.3 --max-tokens 4096 \
    --lr-scheduler inverse_sqrt --weight-decay 0.0001 \
    --criterion reg_label_smoothed_cross_entropy \
    --log-interval 100 --tensorboard-logdir $MODEL_PATH/log --log-format simple \
    --reg-alpha 5 \
    --no-progress-bar --keep-last-epochs 10 \
    --seed 64 \
    --eval-bleu \
    --patience 15 \
    --update-freq 4 \
    --dropout 0.3 \
    --eval-bleu-args '{"beam": 5, "max_len_a": 1.2, "max_len_b": 10}' \
    --eval-bleu-detok moses \
    --eval-bleu-remove-bpe \
    --eval-bleu \
    --best-checkpoint-metric bleu --maximize-best-checkpoint-metric \
    --max-update 300000 --warmup-updates 4000 --warmup-init-lr 1e-07 --adam-betas '(0.9,0.98)' \
    --save-dir ${MODEL_PATH}/ckpt  \
    --fp16 2>&1 | tee -a $MODEL_PATH/train.log \
