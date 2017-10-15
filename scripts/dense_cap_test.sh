#!/usr/bin/env bash

# --------------------------------------------------------
# DenseCap-Tensorflow
# Written by InnerPeace
# This file is adapted from Ross Linjie's work
# --------------------------------------------------------

# TODO: change the test procedure.
GPU_ID=0
NET_FINAL=models/dense_cap/dense_cap_late_fusion_sum.caffemodel
TEST_IMDB="vg_1.0_test"
PT_DIR="dense_cap"
time ./lib/tools/test_net.py --gpu ${GPU_ID} \
  --def_feature models/${PT_DIR}/vgg_region_global_feature.prototxt \
  --def_recurrent models/${PT_DIR}/test_cap_pred_context.prototxt \
  --def_embed models/${PT_DIR}/test_word_embedding.prototxt \
  --net ${NET_FINAL} \
  --imdb ${TEST_IMDB} \
  --cfg models/${PT_DIR}/dense_cap.yml \
