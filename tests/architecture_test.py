# --------------------------------------------------------
# DenseCap-Tensorflow
# Written by InnerPeace
# This file is adapted from Linjie's work
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from lib.config import cfg
import tensorflow as tf
from lib.nets.resnet_v1 import resnetv1
from tests.roidata_test import get_data_test
import numpy as np

def architecture_test():
    blob = get_data_test()
    if cfg.LIMIT_RAM:
        phrases = blob['gt_phrases']
    else:
        phrases = None

    net = resnetv1(50)
    net._build_network()

    tfconfig = tf.ConfigProto(allow_soft_placement=True)
    tfconfig.gpu_options.allow_growth = True

    with tf.Session(config=tfconfig) as sess:
        pass


if __name__ == '__main__':
    architecture_test()

