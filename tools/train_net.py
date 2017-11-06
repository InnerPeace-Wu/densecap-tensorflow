# --------------------------------------------------------
# DenseCap-Tensorflow
# Written by InnerPeace
# This file is adapted from Linjie's work
# --------------------------------------------------------
# Train a dense captioning model
# Code adapted from faster R-CNN project
# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

"""Train a dense caption model"""

import _init_paths
# import os
# from os.path import join as pjoin
import sys
# sys.path.append("..")
# import time
import six
import argparse
import numpy as np
import tensorflow as tf

from lib.config import cfg, cfg_from_file, cfg_from_list, get_output_dir, get_output_tb_dir
from lib.datasets.factory import get_imdb
import lib.datasets.imdb
from lib.dense_cap.train import get_training_roidb, train_net
from lib.nets.vgg16 import vgg16
from lib.nets.resnet_v1 import resnetv1
import pprint

# set up log in bash file
# import logging
# logging.basicConfig(level=logging.INFO)


def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Train a Dense Caption network')
    parser.add_argument('--device', dest='device', help='device to use',
                        default='gpu', type=str)
    parser.add_argument('--device_id', dest='device_id', help='device id to use',
                        default=0, type=int)
    parser.add_argument('--tag', dest='tag',
                        help='tag of the model',
                        default=None, type=str)
    parser.add_argument('--iters', dest='max_iters',
                        help='number of iterations to train',
                        default=40000, type=int)
    parser.add_argument('--weights', dest='pretrained_model',
                        help='initialize with pretrained model weights',
                        default=None, type=str)
    parser.add_argument('--cfg', dest='cfg_file',
                        help='optional config file',
                        default=None, type=str)
    parser.add_argument('--imdb', dest='imdb_name',
                        help='dataset to train on',
                        default='vg_1.2_train', type=str)
    parser.add_argument('--imdbval', dest='imdbval_name',
                        help='dataset to validation on',
                        default='vg_1.2_val', type=str)
    parser.add_argument('--rand', dest='randomize',
                        help='randomize (do not use a fixed seed)',
                        action='store_true')
    # TODO: add inception
    parser.add_argument('--net', dest='net',
                        help='vgg16, res50, res101, res152',
                        default=None, type=str)
    parser.add_argument('--set', dest='set_cfgs',
                        help='set config keys', default=None,
                        nargs=argparse.REMAINDER)

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()
    return args


def combined_roidb(imdb_names):
    # for now: imdb_names='vg_1.2_train'
    def get_roidb(imdb_name):
        imdb = get_imdb(imdb_name)
        print('Loaded dataset `{:s}` for training'.format(imdb.name))
        imdb.set_proposal_method(cfg.TRAIN.PROPOSAL_METHOD)
        print('Set proposal method: {:s}'.format(cfg.TRAIN.PROPOSAL_METHOD))
        roidb = get_training_roidb(imdb)
        return roidb

    roidbs = [get_roidb(s) for s in imdb_names.split('+')]
    roidb = roidbs[0]
    if len(roidbs) > 1:
        for r in roidbs[1:]:
            roidb.extend(r)
        imdb = lib.datasets.imdb.imdb(imdb_names)
    else:
        imdb = get_imdb(imdb_names)
    return imdb, roidb


def get_roidb_limit_ram(imdb_name):
    """
    Note: we need to run get_training_roidb sort of funcs later
    for now, it only supports single roidb.
    """

    imdb = get_imdb(imdb_name)
    roidb = imdb.roidb

    assert isinstance(roidb, six.string_types), \
        "for limit ram vision, roidb should be a path."

    return imdb, roidb


def main():
    args = parse_args()

    # c_time = time.strftime('%m%d_%H%M', time.localtime())
    # if not os.path.exists(cfg.LOG_DIR):
    #     os.makedirs(cfg.LOG_DIR)
    # file_handler = logging.FileHandler(pjoin(cfg.LOG_DIR,
    #                                          args.network_name + '_%s.txt' % c_time))
    # logging.getLogger().addHandler(file_handler)

    print('------ called with args: -------')
    pprint.pprint(args)

    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file)
    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs)

    print("runing with LIMIT_RAM: {}".format(cfg.LIMIT_RAM))

    if not args.randomize:
        # fix the random seeds (numpy and caffe) for reproducibility
        np.random.seed(cfg.RNG_SEED)
        tf.set_random_seed(cfg.RNG_SEED)

    if not cfg.LIMIT_RAM:
        imdb, roidb = combined_roidb(args.imdb_name)
    else:
        imdb, roidb = get_roidb_limit_ram(args.imdb_name)

    output_dir = get_output_dir(imdb, args.tag)
    print("output will be saved to `{:s}`".format(output_dir))

    # tensorboard directory where the summaries are saved during training
    tb_dir = get_output_tb_dir(imdb, args.tag)
    print('TensorFlow summaries will be saved to `{:s}`'.format(tb_dir))

    # also add validation set, but with no flipping image
    orgflip = cfg.TRAIN.USE_FLIPPED
    cfg.TRAIN.USE_FLIPPED = False
    if not cfg.LIMIT_RAM:
        _, valroidb = combined_roidb(args.imdbval_name)
    else:
        _, valroidb = get_roidb_limit_ram(args.imdbval_name)
    cfg.TRAIN.USE_FLIPPED = orgflip

    # load network
    if args.net == 'vgg16':
        net = vgg16()
    elif args.net == 'res50':
        net = resnetv1(num_layers=50)
    elif args.net == 'res101':
        net = resnetv1(num_layers=101)
    elif args.net == 'res152':
        net = resnetv1(num_layers=152)
    else:
        raise NotImplementedError

    # TODO: "imdb" may not be useful during training
    train_net(net, imdb, roidb, valroidb, output_dir, tb_dir,
              pretrained_model=args.pretrained_model,
              max_iters=args.max_iters)


if __name__ == '__main__':
    main()
