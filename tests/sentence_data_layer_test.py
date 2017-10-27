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
    phrases = data['gt_phrases']

    labels = data['gt_boxes'][:3, 4]
    sentence_data_layer(labels, phrases)


if __name__ == '__main__':

    sentence_data_layer_test()
