# --------------------------------------------------------
# DenseCap-Tensorflow
# Written by InnerPeace
# This file is adapted from Linjie's work and Xinlei's work
# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

"""Compute minibatch blobs for training a DenseCap network."""

import numpy as np
import numpy.random as npr
import cv2
from six.moves import cPickle, xrange
from lib.config import cfg
from lib.utils.blob import prep_im_for_blob, im_list_to_blob


def get_minibatch(roidb):
    """Given a roidb, construct a minibatch sampled from it."""

    if cfg.LIMIT_RAM:
        num_images = 1  # one image per minibatch
    else:
        num_images = len(roidb)

    # Sample random scales to use for each image in this batch
    random_scale_inds = npr.randint(0, high=len(cfg.TRAIN.SCALES),
                                    size=num_images)
    assert (cfg.TRAIN.BATCH_SIZE % num_images == 0), \
        'num_images ({}) must divide BATCH_SIZE ({})'. \
            format(num_images, cfg.TRAIN.BATCH_SIZE)

    # Get the input image blob, formatted for caffe
    im_blob, im_scales, roidb = _get_image_blob(roidb, random_scale_inds)

    blobs = {'data': im_blob}

    if cfg.TRAIN.HAS_RPN:
        assert len(im_scales) == 1, "Single batch only"
        assert len(roidb) == 1, "Single batch only"
        # gt boxes: (x1, y1, x2, y2, cls)
        gt_inds = np.where(roidb[0]['gt_classes'] != 0)[0]
        gt_boxes = np.empty((len(gt_inds), 5), dtype=np.float32)
        gt_boxes[:, 0:4] = roidb[0]['boxes'][gt_inds, :] * im_scales[0]
        gt_boxes[:, 4] = roidb[0]['gt_classes'][gt_inds]
        # TODO: add "gt_phrases"
        blobs['gt_phrases'] = _process_gt_phrases(roidb[0]['gt_phrases'])
        blobs['gt_boxes'] = gt_boxes
        blobs['im_info'] = np.array(
            # TODO: for blob format stick to tf_faster_rcnn version
            # [[im_blob.shape[2], im_blob.shape[3], im_scales[0]]],
            # [[im_blob.shape[1], im_blob.shape[2], im_scales[0]]],
            # make it shape [3,]
            [im_blob.shape[1], im_blob.shape[2], im_scales[0]],
            dtype=np.float32)
        # if cfg.LIMIT_RAM:
        #     blobs['gt_phrases'] = roidb[0]['gt_phrases']
    else:  # not using RPN
        raise NotImplementedError

    return blobs


def _process_gt_phrases(phrases):
    """processing gt phrases for blob"""
    num_regions = len(phrases)
    gt_phrases = np.zeros((num_regions, cfg.MAX_WORDS), dtype=np.int32)
    for ix, phra in enumerate(phrases):
        l = len(phra)
        gt_phrases[ix, :l] = phra

    return gt_phrases


def _get_image_blob(roidb, scale_inds):
    """Builds an input blob from the images in the roidb at the specified
    scales.
    """
    num_images = len(scale_inds)
    processed_ims = []
    im_scales = []
    if cfg.LIMIT_RAM:
        # roidb is the pickle file path
        assert num_images == 1, "LIMIT_RAM version, it has to be one image."
        with open(roidb, 'rb') as f:
            roidb = [cPickle.load(f)]

    for i in xrange(num_images):
        im = cv2.imread(roidb[i]['image'])
        if roidb[i]['flipped']:
            im = im[:, ::-1, :]
        target_size = cfg.TRAIN.SCALES[scale_inds[i]]
        im, im_scale = prep_im_for_blob(im, cfg.PIXEL_MEANS, target_size,
                                        cfg.TRAIN.MAX_SIZE)
        im_scales.append(im_scale)
        processed_ims.append(im)

    # Create a blob to hold the input images
    blob = im_list_to_blob(processed_ims)

    return blob, im_scales, roidb

