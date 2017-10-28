# --------------------------------------------------------
# DenseCap-Tensorflow
# Written by InnerPeace
# This file is adapted from Linjie's work
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from lib.config import cfg
from lib.fast_rcnn.bbox_transform import bbox_transform_inv, clip_boxes


# compute the new bboxes shifted by offset from rois
def compute_rois_offset(rois, offset, im_info=None):
    """Compute bounding-box offset for region of interests"""

    assert rois.shape[1] == 4
    assert offset.shape[1] == 4

    if cfg.TRAIN.BBOX_NORMALIZE_TARGETS_PRECOMPUTED:
        # Optionally normalize targets by a precomputed mean and stdev -- reverse the transformation
        offset_unnorm = offset * np.array(cfg.TRAIN.BBOX_NORMALIZE_STDS) + \
                        np.array(cfg.TRAIN.BBOX_NORMALIZE_MEANS)
    else:
        offset_unnorm = offset.copy()
    rois_offset = bbox_transform_inv(rois, offset_unnorm)
    if not im_info is None:
        rois_offset = clip_boxes(rois_offset, im_info[:2])
    return rois_offset
