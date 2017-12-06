# --------------------------------------------------------
# DenseCap-Tensorflow
# Written by InnerPeace
# This file is adapted from Xinlei's work
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np


def GlobalRoILayer(im_info):
    """
    Set up the global RoI
    """
    return np.array([0., 0., 0., im_info[1] - 1, im_info[0] - 1])
