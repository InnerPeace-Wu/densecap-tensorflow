#!/usr/bin/env bash

set -x
set -e

export PYTHONUNBUFFERED='True'

DATASET='visual_genome_1.2'
NET='res50'

case $DATASET in
   visual_genome)
    TRAIN_IMDB="vg_1.0_train"
    TEST_IMDB="vg_1.0_val"
    PT_DIR="dense_cap"
    ITERS=2000
#    FINETUNE_AFTER1=200000
#    FINETUNE_AFTER2=100000
#    ITERS1=400000
#    ITERS2=300000
    ;;
  visual_genome_1.2)
    TRAIN_IMDB="vg_1.2_train"
    TEST_IMDB="vg_1.2_val"
    PT_DIR="dense_cap"
    ITERS=1000
#    FINETUNE_AFTER1=200000
#    FINETUNE_AFTER2=100000
#    ITERS1=400000
#    ITERS2=300000
    ;;
  *)
    echo "No dataset given"
    exit
    ;;
esac

LOG="tests/logs/${NET}_${TRAIN_IMDB}_test.txt.`date +'%Y-%m-%d_%H-%M-%S'`"

exec &> >(tee -a "$LOG")
echo Logging output to "$LOG"

time python ./tools/train_net.py \
    --weights /home/joe/git/slim_models/${NET}.ckpt \
    --imdb ${TRAIN_IMDB} \
    --imdbval ${TEST_IMDB} \
    --iters ${ITERS} \
    --cfg scripts/dense_cap_config.yml \
    --net ${NET} \


