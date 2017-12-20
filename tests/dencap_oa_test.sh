#!/usr/bin/env bash

# This script is used for my own experiments, just ignore it.
# Run with:
#       bash scripts/dense_cap_train.sh [dataset] [net] [ckpt_to_init] [data_dir] [step]

set -x
set -e

export PYTHONUNBUFFERED='True'

DATASET='visual_genome_1.2'
NET='res50'
ckpt_path='/home/joe/git/slim_models'
data_dir='/home/joe/git/visual_genome'
step=$1

case $DATASET in
   visual_genome)
    TRAIN_IMDB="vg_1.0_train"
    TEST_IMDB="vg_1.0_val"
    PT_DIR="dense_cap"
    FINETUNE_AFTER1=200000
    FINETUNE_AFTER2=100000
    ITERS1=400000
    ITERS2=300000
    ;;
  visual_genome_1.2)
    TRAIN_IMDB="vg_1.2_train"
    TEST_IMDB="vg_1.2_val"
    PT_DIR="dense_cap"
    FINETUNE_AFTER1=200000
    FINETUNE_AFTER2=100000
    ITERS1=400000
    ITERS2=300000
    ;;
  *)
    echo "No dataset given"
    exit
    ;;
esac

if [ -d '/valohai/outputs' ]; then
    ckpt_path='/valohai/inputs/resnet'
    data_dir='/valohai/inputs/visual_genome'
    LOG="/valohai/outputs/s${step}_${NET}_${TRAIN_IMDB}.txt.`date +'%Y-%m-%d_%H-%M-%S'`"
else
    LOG="logs/s${step}_${NET}_${TRAIN_IMDB}.txt.`date +'%Y-%m-%d_%H-%M-%S'`"
fi

exec &> >(tee -a "$LOG")
echo Logging output to "$LOG"

FIRST_ITERS=80000
if [ ${step} -lt '2' ]
then
time python ./tools/train_net.py \
    --weights ${ckpt_path}/${NET}.ckpt \
    --imdb ${TRAIN_IMDB} \
    --imdbval ${TEST_IMDB} \
    --iters 50000 \
    --cfg scripts/dense_cap_config.yml \
    --data_dir ${data_dir} \
    --net ${NET} \
    --set TRAIN_GLOVE False EXP_DIR dc_fixed CONTEXT_FUSION False RESNET.FIXED_BLOCKS 3 KEEP_AS_GLOVE_DIM False LOSS.CLS_W 1. LOSS.BBOX_W 0.2 LOSS.RPN_BBOX_W 1. LOSS.RPN_CLS_W 0.5
    # --set EXP_DIR dc_fixed CONTEXT_FUSION False RESNET.FIXED_BLOCKS 3

# mkdir output/dc_fixed
# cp -r output/Densecap/ output/dc_dc_fixed
fi

NEW_WIGHTS=output/dc_fixed/${TRAIN_IMDB}
if [ ${step} -lt '3' ]
then
time python ./tools/train_net.py \
    --weights ${NEW_WIGHTS} \
    --imdb ${TRAIN_IMDB} \
    --iters 30000 \
    --imdbval ${TEST_IMDB} \
    --cfg scripts/dense_cap_config.yml \
    --data_dir ${data_dir} \
    --net ${NET} \
    --set TRAIN_GLOVE True EXP_DIR dc_tune_vec CONTEXT_FUSION False RESNET.FIXED_BLOCKS 3 KEEP_AS_GLOVE_DIM False
# TRAIN.LEARNING_RATE 0.0005
# --iters `expr ${FINETUNE_AFTER1} - ${FIRST_ITERS}` \

# mkdir output/dc_tune_vec
# cp -r output/Densecap/ output/dc_tune_vec
fi

#NEW_WIGHTS=output/dc_tune_vec/${TRAIN_IMDB}
if [ ${step} -lt '4' ]
then
time python ./tools/train_net.py \
    --weights ${NEW_WIGHTS} \
    --imdb ${TRAIN_IMDB} \
    --imdbval ${TEST_IMDB} \
    --iters `expr ${ITERS1} - ${FINETUNE_AFTER1}` \
    --cfg scripts/dense_cap_config.yml \
    --data_dir ${data_dir} \
    --net ${NET} \
    --set EXP_DIR dc_tune_conv CONTEXT_FUSION False RESNET.FIXED_BLOCKS 1

# mkdir output/dc_tune_conv
# cp -r output/Densecap/ output/dc_tune_conv
fi

NEW_WIGHTS=output/dc_tune_conv/${TRAIN_IMDB}
if [ ${step} -lt '5' ]
then
time python ./tools/train_net.py \
    --weights ${NEW_WIGHTS} \
    --imdb ${TRAIN_IMDB} \
    --imdbval ${TEST_IMDB} \
    --iters ${FINETUNE_AFTER2} \
    --cfg scripts/dense_cap_config.yml \
    --data_dir ${data_dir} \
    --net ${NET} \
    --set TRAIN_GLOVE True EXP_DIR dc_context CONTEXT_FUSION True RESNET.FIXED_BLOCKS 3
# mkdir output/dc_context
# cp -r output/Densecap/ output/dc_context
# --iters `expr ${FINETUNE_AFTER1} - ${FIRST_ITERS}`
fi

NEW_WIGHTS=output/dc_context/${TRAIN_IMDB}
if [ ${step} -lt '6' ]
then
time python ./tools/train_net.py \
    --weights ${NEW_WIGHTS} \
    --imdb ${TRAIN_IMDB} \
    --imdbval ${TEST_IMDB} \
    --iters `expr ${ITERS2} - ${FINETUNE_AFTER2}` \
    --cfg scripts/dense_cap_config.yml \
    --data_dir ${data_dir} \
    --net ${NET} \
    --set TRAIN_GLOVE True EXP_DIR dc_tune_context CONTEXT_FUSION True RESNET.FIXED_BLOCKS 1
fi
