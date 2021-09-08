#!/usr/bin/env bash

TEXT=data/en-fr-100k/tmp
fairseq-preprocess --source-lang en --target-lang fr \
    --joined-dictionary \
    --trainpref $TEXT/train100k --validpref $TEXT/valid --testpref $TEXT/test\
    --destdir data/en-fr-100k \
    --workers 20
