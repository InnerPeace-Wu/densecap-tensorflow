# --------------------------------------------------------
# Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick and Xinlei Chen
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from lib.config import cfg
from lib.fast_rcnn.bbox_transform import bbox_transform_inv, clip_boxes
from lib.fast_rcnn.nms_wrapper import nms


def proposal_layer(rpn_cls_prob, rpn_bbox_pred, im_info, cfg_key, _feat_stride, anchors, num_anchors):
    """A simplified version compared to fast/er RCNN
     For details please see the technical report
  """
    if type(cfg_key) == bytes:
        cfg_key = cfg_key.decode('utf-8')
    pre_nms_topN = cfg[cfg_key].RPN_PRE_NMS_TOP_N
    post_nms_topN = cfg[cfg_key].RPN_POST_NMS_TOP_N
    nms_thresh = cfg[cfg_key].RPN_NMS_THRESH

    # Get the scores and bounding boxes
    scores = rpn_cls_prob[:, :, :, num_anchors:]
    rpn_bbox_pred = rpn_bbox_pred.reshape((-1, 4))
    scores = scores.reshape((-1, 1))
    proposals = bbox_transform_inv(anchors, rpn_bbox_pred)
    if cfg.DEBUG_ALL:
        print ('number of proposals before clip boxes to image board: {}'.format(
            proposals.shape[0]
        ))
    proposals = clip_boxes(proposals, im_info[:2])

    # remove predicted boxes with either height or width < threshold
    # (NOTE: convert min_size to input image scale stored in im_info[2])
    if cfg.FILTER_SMALL_BOX:
        min_size = cfg[cfg_key].RPN_MIN_SIZE
        keep = _filter_boxes(proposals, min_size * im_info[2])
        proposals = proposals[keep, :]
        scores = scores[keep]

    # Pick the top region proposals
    order = scores.ravel().argsort()[::-1]
    if pre_nms_topN > 0:
        order = order[:pre_nms_topN]
    proposals = proposals[order, :]
    scores = scores[order]

    # Non-maximal suppression
    if cfg.DEBUG_ALL:
        print("number of proposals before nms: {}".format(proposals.shape[0]))
    keep = nms(np.hstack((proposals, scores)), nms_thresh)
    if cfg.DEBUG_ALL:
        print("number of proposals after nms: {}".format(len(keep)))

    # Pick th top region proposals after NMS
    if post_nms_topN > 0:
        keep = keep[:post_nms_topN]
    proposals = proposals[keep, :]
    scores = scores[keep]

    # Only support single image as input
    batch_inds = np.zeros((proposals.shape[0], 1), dtype=np.float32)
    blob = np.hstack((batch_inds, proposals.astype(np.float32, copy=False)))

    return blob, scores


def _filter_boxes(boxes, min_size):
    """Remove all boxes with any side smaller than min_size."""

    ws = boxes[:, 2] - boxes[:, 0] + 1
    hs = boxes[:, 3] - boxes[:, 1] + 1
    keep = np.where((ws >= min_size) & (hs >= min_size))[0]
    return keep
