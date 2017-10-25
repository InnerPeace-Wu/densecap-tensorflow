# --------------------------------------------------------
# DenseCap-Tensorflow
# Written by InnerPeace
# This file is adapted from Linjie's work
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from lib.config import cfg
from lib.layers.sentence_data_layer import sentence_data_layer
from tests.roidata_test import get_data_test
import numpy as np


def sentence_data_layer_test():
    data = get_data_test()
    if cfg.LIMIT_RAM:
        phrases = data['gt_phrases']
    else:
        phrases = None

    labels = np.arange(1382, 1385)
    sentence_data_layer(labels, split='pre', gt_phrases=phrases)


if __name__ == '__main__':

    sentence_data_layer_test()
