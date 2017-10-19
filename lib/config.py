# --------------------------------------------------------
# DenseCap-Tensorflow
# Written by InnerPeace
# This file is adapted from Ross Girshick's work
# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

""" config system for Densecap

THis file specifies default config for Densecap. One can change the value of the file by
writing a config file(in yaml) and use cfg_from_file(yaml_file) to load and override the
default options.
"""

import os
import os.path as osp
from os.path import join as pjoin
import numpy as np
# from distutils import spawn
# run: pip install easydict
from easydict import EasyDict as edict

__C = edict()
# get config by:
#   from lib.config import cfg
cfg = __C

#
# Training options
#

__C.TRAIN = edict()

# Training using proposal
__C.TRAIN.PROPOSAL_METHOD = 'gt'

# Use horizontally-flipped images during training?
__C.TRAIN.USE_FLIPPED = True

# Make minibatches from images that have similar aspect ratios (i.e. both
# tall and thin or both short and wide) in order to avoid wasting computation
# on zero-padding.
__C.TRAIN.ASPECT_GROUPING = True

# images to use per minibatch, we use 1 in default
__C.TRAIN.IMS_PER_BATCH = 1

# Scales to use during training (can list multiple scales)
# Each scale is the pixel size of an image's shortest side
__C.TRAIN.SCALES = (600,)

# Max pixel size of the longest side of a scaled input image
__C.TRAIN.MAX_SIZE = 720

# Minibatch size (number of regions of interest [ROIs])
__C.TRAIN.BATCH_SIZE = 256

# Fraction of minibatch that is labeled foreground (i.e. class > 0)
__C.TRAIN.FG_FRACTION = 0.25

# Overlap threshold for a ROI to be considered foreground (if >= FG_THRESH)
__C.TRAIN.FG_THRESH = 0.5

# Overlap threshold for a ROI to be considered background (class = 0 if
# overlap in [LO, HI))
__C.TRAIN.BG_THRESH_HI = 0.5
__C.TRAIN.BG_THRESH_LO = 0.1

# Use RPN to detect objects
__C.TRAIN.HAS_RPN = True

# Train bounding-box regressors
__C.TRAIN.BBOX_REG = True

# Normalize the targets (subtract empirical mean, divide by empirical stddev)
__C.TRAIN.BBOX_NORMALIZE_TARGETS = True

# Normalize the targets using "precomputed" (or made up) means and stdevs
# (BBOX_NORMALIZE_TARGETS must also be True)
__C.TRAIN.BBOX_NORMALIZE_TARGETS_PRECOMPUTED = False
__C.TRAIN.BBOX_NORMALIZE_MEANS = (0.0, 0.0, 0.0, 0.0)
__C.TRAIN.BBOX_NORMALIZE_STDS = (0.1, 0.1, 0.2, 0.2)


#
# MISC
#

# Root directory of project
__C.ROOT_DIR = osp.abspath(pjoin(osp.dirname(__file__), '..'))

# Data directory
# __C.DATA_DIR = osp.abspath(pjoin(__C.ROOT_DIR, 'data'))
__C.DATA_DIR = '/home/joe/git/visual_genome'

# Log directory
__C.LOG_DIR = osp.abspath(pjoin(__C.ROOT_DIR, 'logs'))

# Cache directory
__C.CACHE_DIR = __C.DATA_DIR

# Dataset splits directory
__C.SPLIT_DIR = osp.abspath(pjoin(__C.ROOT_DIR, 'info'))

# Place outputs under an experiment directory
__C.EXP_DIR = 'default'

# Limited memory which is less than 16G and unable to read the whole
# region description JSON file
__C.LIMIT_RAM = True

# For reproducibility
__C.RNG_SEED = 3

# Pixel mean values (BGR order) as a (1, 1, 3) array
# We use the same pixel mean for all networks even though it's not exactly what
# they were trained with
__C.PIXEL_MEANS = np.array([[[102.9801, 115.9465, 122.7717]]])


#
# Functions
#

def get_output_dir(imdb, weights_filename):
    """Return the directory where experimental artifacts are placed.
    If the directory does not exist, it is created.

    A canonical path is built using the name from an imdb and a network
    (if not None).
    """
    outdir = osp.abspath(osp.join(__C.ROOT_DIR, 'output', __C.EXP_DIR, imdb.name))
    if weights_filename is not None:
        outdir = osp.join(outdir, weights_filename)
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    return outdir


def get_output_tb_dir(imdb, weights_filename):
    """Return the directory where tensorflow summaries are placed.
    If the directory does not exist, it is created.

    A canonical path is built using the name from an imdb and a network
    (if not None).
    """
    outdir = osp.abspath(osp.join(__C.ROOT_DIR, 'tensorboard', __C.EXP_DIR, imdb.name))
    if weights_filename is None:
        weights_filename = 'default'
    outdir = osp.join(outdir, weights_filename)
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    return outdir


def _merge_a_into_b(a, b):
    """Merge config dictionary a into config dictionary b, clobbering the
    options in b whenever they are also specified in a.
    """
    if type(a) is not edict:
        return

    for k, v in a.iteritems():
        # a must specify keys that are in b
        if not b.has_key(k):
            raise KeyError('{} is not a valid config key'.format(k))

        # the types must match, too
        old_type = type(b[k])
        if old_type is not type(v):
            if isinstance(b[k], np.ndarray):
                v = np.array(v, dtype=b[k].dtype)
            else:
                raise ValueError(('Type mismatch ({} vs. {}) '
                                  'for config key: {}').format(type(b[k]),
                                                               type(v), k))

        # recursively merge dicts
        if type(v) is edict:
            try:
                _merge_a_into_b(a[k], b[k])
            except:
                print('Error under config key: {}'.format(k))
                raise
        else:
            b[k] = v


def cfg_from_file(filename):
    """Load a config file and merge it into the default options."""
    import yaml
    with open(filename, 'r') as f:
        yaml_cfg = edict(yaml.load(f))

    _merge_a_into_b(yaml_cfg, __C)


def cfg_from_list(cfg_list):
    """Set config keys via list (e.g., from command line)."""
    from ast import literal_eval
    assert len(cfg_list) % 2 == 0
    for k, v in zip(cfg_list[0::2], cfg_list[1::2]):
        key_list = k.split('.')
        d = __C
        for subkey in key_list[:-1]:
            assert d.has_key(subkey)
            d = d[subkey]
        subkey = key_list[-1]
        assert d.has_key(subkey)
        try:
            value = literal_eval(v)
        except:
            # handle the case when v is a string literal
            value = v
        assert type(value) == type(d[subkey]), \
            'type {} does not match original type {}'.format(
                type(value), type(d[subkey]))
        d[subkey] = value


if __name__ == '__main__':
    print(cfg.ROOT_DIR)
