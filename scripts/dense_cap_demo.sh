#!/usr/bin/env bash
set -x
set -e

time python ./tools/demo.py \
    --ckpt '/home/joe/git/densecap/output/dc_fixed/vg_1.2_train' \
    --cfg '/home/joe/git/densecap/scripts/dense_cap_config.yml' \
    --vocab '/home/joe/git/visual_genome/1.2/vocabulary.txt' \
    --set TEST.USE_BEAM_SEARCH False EMBED_DIM 512
