#!/usr/bin/env bash

# --------------------------------------------------------
# DenseCap-Tensorflow
# Written by InnerPeace
# This file is adapted from Ross Linjie's work
# --------------------------------------------------------

# TODO: change the test procedure.
GPU_ID=0
CKPT="/home/joe/git/densecap/output/dense_cap/vg_1.2_train"
TEST_IMDB="vg_1.2_test"
PT_DIR="dense_cap"
time python ./tools/test_net.py  \
  --ckpt ${CKPT} \
  --imdb ${TEST_IMDB} \
  --cfg scripts/dense_cap_config.yml \
