# --------------------------------------------------------
# DenseCap-Tensorflow
# Written by InnerPeace
# This file is adapted from Ross Girshick's work
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

"""functions for LIMIT_RAM version"""

# import sys
# sys.path.append("..")

import numpy as np
from lib.config import cfg


def pre_roidb(roidb):
    """Enrich the imdb's roidb by adding some derived quantities that
    are useful for training. This function precomputes the maximum
    overlap, taken over ground-truth boxes, between each ROI and
    each ground-truth box. The class with maximum overlap is also
    recorded.
    """
    # need gt_overlaps as a dense array for argmax
    gt_overlaps = roidb['gt_overlaps'].toarray()
    # max overlap with gt over classes (columns)
    max_overlaps = gt_overlaps.max(axis=1)
    # gt class that had the max overlap
    max_classes = gt_overlaps.argmax(axis=1)
    roidb['max_classes'] = max_classes
    roidb['max_overlaps'] = max_overlaps
    # sanity checks
    # max overlap of 0 => class should be zero (background)
    zero_inds = np.where(max_overlaps == 0)[0]
    assert all(max_classes[zero_inds] == 0)
    # max overlap > 0 => class should not be zero (must be a fg class)
    # nonzero_inds = np.where(max_overlaps > 0)[0]
    # assert all(max_classes[nonzero_inds] != 0)
    return roidb


def is_valid_limitRam(entry):
    # Valid images have:
    #   (1) At least one foreground RoI OR
    #   (2) At least one background RoI
    overlaps = entry['max_overlaps']
    # find boxes with sufficient overlap
    fg_inds = np.where(overlaps >= cfg.TRAIN.FG_THRESH)[0]
    # Select background RoIs as those within [BG_THRESH_LO, BG_THRESH_HI)
    bg_inds = np.where((overlaps < cfg.TRAIN.BG_THRESH_HI) &
                       (overlaps >= cfg.TRAIN.BG_THRESH_LO))[0]
    # image is only valid if such boxes exist
    valid = len(fg_inds) > 0 or len(bg_inds) > 0
    return valid


def flip_image(roidb):
    """flip image and change the name for reading later"""

    boxes = roidb['boxes'].copy()
    oldx1 = boxes[:, 0].copy()
    oldx2 = boxes[:, 2].copy()
    boxes[:, 0] = roidb['width'] - oldx2 - 1
    boxes[:, 2] = roidb['width'] - oldx1 - 1
    assert (boxes[:, 2] >= boxes[:, 0]).all()
    entry = {'boxes': boxes,
             'gt_overlaps': roidb['gt_overlaps'],
             'gt_classes': roidb['gt_classes'],
             'flipped': True,
             'gt_phrases': roidb['gt_phrases'],
             'width': roidb['width'],
             'height': roidb['height'],
             'image': roidb['image'],
             'image_id': '%s_flip' % roidb['image_id']}

    return entry
