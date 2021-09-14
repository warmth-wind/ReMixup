#!/usr/bin/env bash

TEXT=data/en-fr/tmp
fairseq-preprocess --source-lang en --target-lang fr \
    --joined-dictionary \
    --trainpref $TEXT/train --validpref $TEXT/valid --testpref $TEXT/test\
    --destdir data/en-fr \
    --workers 20
