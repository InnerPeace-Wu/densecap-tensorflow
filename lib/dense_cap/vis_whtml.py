# ----------------------------------------------
# DenseCap
# Written by InnerPeace
# ----------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cv2
import os
import numpy as np
from six.moves import xrange


def vis_whtml(im_path, im, captions, dets, pre_results=dict(),
              thresh=0.5, save_path='./vis/data'):
    print("visualizing with pretty html...")
    if not os.path.exists(save_path):
        os.mkdirs(save_path)

    im_name = im_path.split('/')[-1][:-4]
    box_xywh = []
    box_caps = []
    scores = []
    for i in xrange(dets.shape[0]):
        if dets[i, -1] > thresh:
            box_xywh.append(box2xywh(dets[i, :4].tolist()))
            box_caps.append(captions[i])
            scores.append(float(dets[i, -1]))

    # save image
    im_new = np.copy(im)
    cv2.imwrite("%s/%s.jpg" % (save_path, im_name), im_new)
    result = {"img_name": "%s.jpg" % im_name,
              "scores": scores,
              "captions": box_caps,
              "boxes": box_xywh}
    pre_results["results"] = pre_results.get("results", []) + [result]

    return pre_results


def box2xywh(box):
    xywh = []
    xywh.extend(box[:2])
    for i in xrange(2):
        xywh.append(box[i+2] - box[i])

    return xywh
