#!/usr/bin/env bash

# Run with:
#       bash scripts/dense_cap_train.sh [dataset] [net] [ckpt_to_init] [data_dir] [step]

set -x
set -e

export PYTHONUNBUFFERED='True'

DATASET=$1
NET=$2
ckpt_path=$3
data_dir=$4
step=$5

# For my own experiment usage, just ignore it.
if [ -d '/home/joe' ]; then
    DATASET='visual_genome_1.2'
    NET='res50'
    ckpt_path="experiments/random_fixconv_i85k_171219/dc_fixed_1219/vg_1.2_train"
    # ckpt_path="experiments/rd_fixconv_i165k_171221/dc_conv_fixed/vg_1.2_train"
    # ckpt_path='/home/joe/git/slim_models/res50.ckpt'
    data_dir='/home/joe/git/visual_genome'
fi

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

# This is for valohai computing platform, one can just ignore it.
if [ -d '/valohai/outputs' ]; then
    ckpt_path='/valohai/inputs/resnet'
    data_dir='/valohai/inputs/visual_genome'
    LOG="/valohai/outputs/s${step}_${NET}_${TRAIN_IMDB}.txt.`date +'%Y-%m-%d_%H-%M-%S'`"
else
    LOG="logs/s${step}_${NET}_${TRAIN_IMDB}.txt.`date +'%Y-%m-%d_%H-%M-%S'`"
fi

exec &> >(tee -a "$LOG")
echo Logging output to "$LOG"

# First step, freeze conv nets weights
if [ ${step} -lt '2' ]
then
time python ./tools/train_net.py \
    --weights ${ckpt_path} \
    --imdb ${TRAIN_IMDB} \
    --imdbval ${TEST_IMDB} \
    --iters ${FINETUNE_AFTER1}\
    --cfg scripts/dense_cap_config.yml \
    --data_dir ${data_dir} \
    --net ${NET} \
    --set EXP_DIR dc_conv_fixed CONTEXT_FUSION False RESNET.FIXED_BLOCKS 3
fi

# Step2: Finetune convnets
NEW_WIGHTS=output/dc_conv_fixed/${TRAIN_IMDB}
if [ ${step} -lt '3' ]
then
time python ./tools/train_net.py \
    --weights ${NEW_WIGHTS} \
    --imdb ${TRAIN_IMDB} \
    --iters `expr ${ITERS1} - ${FINETUNE_AFTER1}` \
    --imdbval ${TEST_IMDB} \
    --cfg scripts/dense_cap_config.yml \
    --data_dir ${data_dir} \
    --net ${NET} \
    --set EXP_DIR dc_tune_conv CONTEXT_FUSION False RESNET.FIXED_BLOCKS 1 TRAIN.LEARNING_RATE 0.00025
fi

# Step3: train with contex fusion
NEW_WIGHTS=output/dc_tune_conv/${TRAIN_IMDB}
if [ ${step} -lt '4' ]
then
time python ./tools/train_net.py \
    --weights ${NEW_WIGHTS} \
    --imdb ${TRAIN_IMDB} \
    --imdbval ${TEST_IMDB} \
    --iters ${FINETUNE_AFTER2} \
    --cfg scripts/dense_cap_config.yml \
    --data_dir ${data_dir} \
    --net ${NET} \
    --set EXP_DIR dc_context CONTEXT_FUSION True RESNET.FIXED_BLOCKS 3 TRAIN.LEARNING_RATE 0.000125
fi

# Step4: finetune context fusion
NEW_WIGHTS=output/dc_context/${TRAIN_IMDB}
if [ ${step} -lt '5' ]
then
time python ./tools/train_net.py \
    --weights ${NEW_WIGHTS} \
    --imdb ${TRAIN_IMDB} \
    --imdbval ${TEST_IMDB} \
    --iters `expr ${ITERS2} - ${FINETUNE_AFTER2}` \
    --cfg scripts/dense_cap_config.yml \
    --data_dir ${data_dir} \
    --net ${NET} \
    --set EXP_DIR dc_tune_context CONTEXT_FUSION True RESNET.FIXED_BLOCKS 1 TRAIN.LEARNING_RATE 0.0000625
fi
