# --------------------------------------------------------
# DenseCap-Tensorflow
# Written by InnerPeace
# This file is adapted from Linjie's work
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

"""This python layer accepts region ids as input and 
retrieves region sentense for them."""

from six.moves import cPickle
from lib.config import cfg
from collections import Counter
import numpy as np
import six
from six.moves import xrange

# TODO: disable debug and clear stuff
DEBUG = True


def sentence_data_layer(labels, roi_phrases, time_steps=12, mode='concat'):
    all_modes = ('repeat', 'concat')
    assert (mode in all_modes), "Wrong type of mode which should be 'repeat' or 'concat'"

    if DEBUG:
        print('length of labels, i.e. number of regions: {}'.format(len(roi_phrases)))

    # all_regions is a dict from region id to caption stream
    assert len(labels.shape) == 2, 'Pleace check the shape of "label"'

    num_regions = labels.shape[0]
    if mode == 'repeat':
        input_sentence = np.zeros((num_regions, time_steps), dtype=np.float32)
    elif mode == 'concat':
        input_sentence = np.zeros((num_regions, time_steps - 1), dtype=np.float32)

    target_sentence = np.zeros((num_regions, time_steps), dtype=np.float32)
    cont_sentence = np.zeros((num_regions, time_steps), dtype=np.float32)
    cont_bbox = np.zeros((num_regions, time_steps), dtype=np.float32)
    for i in xrange(num_regions):
        stream = get_streams(roi_phrases[i], int(labels[i]), time_steps, mode)
        input_sentence[i, :] = stream['input_sentence']
        target_sentence[i, :] = stream['target_sentence']
        cont_sentence[i, :] = stream['cont_sentence']
        cont_bbox[i, :] = stream['cont_bbox']

    if DEBUG:
        print('sentence data layer input (first 3)')
        for ix, l in enumerate(labels[:3]):
            print(l[0], roi_phrases[ix])
        print('sentence data layer output (first 3)')
        print('input sentence')
        print(input_sentence[:3, :])
        print('target sentence')
        print(target_sentence[:3, :])
        print('cont sentence')
        print(cont_sentence[:3, :])
        print('cont bbox')
        print(cont_bbox[:3, :])

    return input_sentence, target_sentence, cont_sentence, cont_bbox


def get_streams(phrases, region_id, time_steps=12, mode='concat'):

    if mode == 'repeat':
        # Image features repeated at each time step
        if region_id > 0:
            stream = phrases[:np.sum(phrases > 0)]
            stream = stream.tolist()
            pad = time_steps - (len(stream) + 1)
            out = {}
            out['cont_sentence'] = [0] + [1] * len(stream) + [0] * pad
            out['input_sentence'] = [1] + stream + [0] * pad
            out['target_sentence'] = stream + [2] + [0] * pad
            # only make prediction at the last time step for bbox
            out['cont_bbox'] = [0] * len(stream) + [1] + [0] * pad

            for key, val in six.iteritems(out):
                if len(val) > time_steps:
                    out[key] = val[:time_steps]
        else:
            # negative sample, no phrase related
            out = {}
            out['cont_sentence'] = [0] * time_steps
            out['input_sentence'] = [0] * time_steps
            out['target_sentence'] = [0] * time_steps
            out['cont_bbox'] = [0] * time_steps

    elif mode == 'concat':
        # Image feature concatenated to the first time step
        if region_id > 0:
            # stream = phrases[region_id]
            stream = phrases[:np.sum(phrases > 0)]
            stream = stream.tolist()
            pad = time_steps - (len(stream) + 2)
            out = {}
            out['cont_sentence'] = [0] + [1] * (len(stream) + 1) + [0] * pad
            out['input_sentence'] = [1] + stream + [0] * pad
            out['target_sentence'] = [1] + stream + [2] + [0] * pad
            # only make prediction at the last time step for bbox
            out['cont_bbox'] = [0] * (len(stream) + 1) + [1] + [0] * pad

            for key, val in six.iteritems(out):
                if len(val) > time_steps:
                    out[key] = val[:time_steps]
        else:
            # negative sample, no phrase related
            out = {}
            out['cont_sentence'] = [0] * time_steps
            out['input_sentence'] = [0] * (time_steps - 1)
            out['target_sentence'] = [0] * time_steps
            out['cont_bbox'] = [0] * time_steps
    else:
        # Global feature and region feature concatenated to the first time step
        if region_id > 0:
            stream = phrases[region_id]
            stream = stream.tolist()
            pad = time_steps - (len(stream) + 3)
            out = {}
            out['cont_sentence'] = [0] + [1] * (len(stream) + 2) + [0] * pad
            out['input_sentence'] = [1] + stream + [0] * pad
            out['target_sentence'] = [1, 1] + stream + [2] + [0] * pad
            # only make prediction at the last time step for bbox
            out['cont_bbox'] = [0] * (len(stream) + 2) + [1] + [0] * pad

            for key, val in out.iteritems():
                if len(val) > time_steps:
                    out[key] = val[:time_steps]
        else:
            # negative sample, no phrase related
            out = {}
            out['cont_sentence'] = [0] * time_steps
            out['input_sentence'] = [0] * (time_steps - 2)
            out['target_sentence'] = [0] * time_steps
            out['cont_bbox'] = [0] * time_steps

    return out
