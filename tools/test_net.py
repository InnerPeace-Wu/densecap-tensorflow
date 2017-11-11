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

"""Test a dense caption model"""
import _init_paths
from lib.dense_cap.test import test_net
from lib.config import cfg, cfg_from_file, cfg_from_list
from lib.datasets.factory import get_imdb
import argparse
import pprint
import time
import os
import sys
import tensorflow as tf
from lib.nets.vgg16 import vgg16
from lib.nets.resnet_v1 import resnetv1


def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Test a Fast R-CNN network')
    parser.add_argument('--device', dest='device', help='device to use',
                        default='gpu', type=str)
    parser.add_argument('--device_id', dest='device_id', help='device id to use',
                        default=0, type=int)
    parser.add_argument('--tag', dest='tag',
                        help='tag of the model',
                        default=None, type=str)
    parser.add_argument('--ckpt', dest='ckpt',
                        help='initialize with pretrained model weights',
                        default=None, type=str)
    parser.add_argument('--cfg', dest='cfg_file',
                        help='optional config file',
                        default=None, type=str)
    parser.add_argument('--imdb', dest='imdb_name',
                        help='dataset to test on',
                        default='vg_1.2_test', type=str)
    # TODO: delete extra options
    # parser.add_argument('--iters', dest='max_iters',
    #                     help='number of iterations to train',
    #                     default=40000, type=int)
    # parser.add_argument('--imdbval', dest='imdbval_name',
    #                     help='dataset to validation on',
    #                     default='vg_1.2_val', type=str)
    # parser.add_argument('--rand', dest='randomize',
    #                     help='randomize (do not use a fixed seed)',
    #                     action='store_true')
    # TODO: add inception
    parser.add_argument('--net', dest='net',
                        help='vgg16, res50, res101, res152',
                        default='res50', type=str)
    parser.add_argument('--vis', dest='vis', help='visualize detections',
                        action='store_true')
    parser.add_argument('--use_box_at', dest='use_box_at',
                        help='use predicted box at this time step, default to the last',
                        default=-1, type=int)
    parser.add_argument('--set', dest='set_cfgs',
                        help='set config keys', default=None,
                        nargs=argparse.REMAINDER)

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()

    print('Called with args:')
    print(args)

    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file)
    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs)

    cfg.GPU_ID = args.device_id

    print('Using config:')
    pprint.pprint(cfg)

    imdb = get_imdb(args.imdb_name)
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

    net.create_architecture("TEST", num_classes=1, tag='pre')
    # read checkpoint file
    if args.ckpt:
        ckpt = tf.train.get_checkpoint_state(args.ckpt)
    else:
        raise ValueError("NO checkpoint found in {}".format(args.ckpt))

    # set config
    tfconfig = tf.ConfigProto(allow_soft_placement=True)
    tfconfig.gpu_options.allow_growth = True

    # init session
    saver = tf.train.Saver()
    with tf.Session(config=tfconfig) as sess:
        print('Restored from {}'.format(ckpt.model_checkpoint_path))
        saver.restore(sess, ckpt.model_checkpoint_path)

        test_net(sess, net, imdb,
                 vis=args.vis, use_box_at=args.use_box_at)
