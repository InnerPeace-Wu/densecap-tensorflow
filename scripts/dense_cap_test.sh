#!/usr/bin/env bash

# --------------------------------------------------------
# DenseCap-Tensorflow
# Written by InnerPeace
# This file is adapted from Ross Linjie's work
# --------------------------------------------------------

# TODO: change the test procedure.
set -x
set -e

GPU_ID=0
CKPT=$1
TEST_IMDB=$2


# Fro valohai platform, maybe out of date.
if [ -d '/valohai/outputs' ]; then
    CKPT="./output/Densecap_res50_context_all/vg_1.2_train"
fi

# For my own experiment, just ignore it.
if [ -d '/home/joe' ]; then
    CKPT="/home/joe/git/densecap/output/dc_tune_context/vg_1.2_train"
    TEST_IMDB="vg_1.2_test"
fi

LOG="logs/test_log.txt.`date +'%Y-%m-%d_%H-%M-%S'`"
exec &> >(tee -a "$LOG")
echo Logging output to "$LOG"

time python ./tools/test_net.py  \
  --ckpt ${CKPT} \
  --imdb ${TEST_IMDB} \
  --cfg scripts/dense_cap_config.yml \
  --set ALL_TEST True
