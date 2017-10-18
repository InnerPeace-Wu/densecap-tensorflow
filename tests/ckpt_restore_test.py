# ----------------------------------------------
# DenseCap
# Written by InnerPeace
# This file is adapted from Xinlei's work
# ----------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.python import pywrap_tensorflow
import tensorflow.contrib.slim as slim

from tensorflow.contrib.slim import arg_scope
from tensorflow.contrib.slim.python.slim.nets import resnet_utils
from tensorflow.contrib.slim.python.slim.nets import resnet_v1
from tensorflow.contrib.slim.python.slim.nets.resnet_v1 import resnet_v1_block
import numpy as np

from lib.config import cfg


def resnet_arg_scope(is_training=True,
                     batch_norm_decay=0.997,
                     batch_norm_epsilon=1e-5,
                     batch_norm_scale=True):
    batch_norm_params = {
        'is_training': False,
        'decay': batch_norm_decay,
        'epsilon': batch_norm_epsilon,
        'scale': batch_norm_scale,
        'trainable': False,
        'updates_collections': tf.GraphKeys.UPDATE_OPS
    }

    with arg_scope(
            [slim.conv2d],
            # weights_regularizer=slim.l2_regularizer(cfg.TRAIN.WEIGHT_DECAY),
            weights_regularizer=None,
            weights_initializer=slim.variance_scaling_initializer(),
            trainable=is_training,
            activation_fn=tf.nn.relu,
            normalizer_fn=slim.batch_norm,
            normalizer_params=batch_norm_params):
        with arg_scope([slim.batch_norm], **batch_norm_params) as arg_sc:
            return arg_sc


class resnetv1():
    def __init__(self, num_layers=50):
        # Network.__init__(self)
        self._feat_stride = [16, ]
        self._feat_compress = [1. / float(self._feat_stride[0]), ]
        self._num_layers = num_layers
        self._scope = 'resnet_v1_%d' % num_layers
        self._decide_blocks()

    # Do the first few layers manually, because 'SAME' padding can behave inconsistently
    # for images of different sizes: sometimes 0, sometimes 1
    def _build_base(self):
        with tf.variable_scope(self._scope, self._scope):
            net = resnet_utils.conv2d_same(self._image, 64, 7, stride=2, scope='conv1')
            net = tf.pad(net, [[0, 0], [1, 1], [1, 1], [0, 0]])
            net = slim.max_pool2d(net, [3, 3], stride=2, padding='VALID', scope='pool1')

        return net

    def _image_to_head(self, is_training, reuse=None):
        assert (0 <= cfg.RESNET.FIXED_BLOCKS <= 3)
        # Now the base is always fixed during training
        with slim.arg_scope(resnet_arg_scope(is_training=False)):
            net_conv = self._build_base()
        if cfg.RESNET.FIXED_BLOCKS > 0:
            with slim.arg_scope(resnet_arg_scope(is_training=False)):
                net_conv, _ = resnet_v1.resnet_v1(net_conv,
                                                  self._blocks[0:cfg.RESNET.FIXED_BLOCKS],
                                                  global_pool=False,
                                                  include_root_block=False,
                                                  reuse=reuse,
                                                  scope=self._scope)
        if cfg.RESNET.FIXED_BLOCKS < 3:
            with slim.arg_scope(resnet_arg_scope(is_training=is_training)):
                net_conv, _ = resnet_v1.resnet_v1(net_conv,
                                                  self._blocks[cfg.RESNET.FIXED_BLOCKS:-1],
                                                  global_pool=False,
                                                  include_root_block=False,
                                                  reuse=reuse,
                                                  scope=self._scope)

        self._act_summaries.append(net_conv)
        self._layers['head'] = net_conv

        return net_conv

    def _decide_blocks(self):
        # choose different blocks for different number of layers
        if self._num_layers == 50:
            self._blocks = [resnet_v1_block('block1', base_depth=64, num_units=3, stride=2),
                            resnet_v1_block('block2', base_depth=128, num_units=4, stride=2),
                            # use stride 1 for the last conv4 layer
                            resnet_v1_block('block3', base_depth=256, num_units=6, stride=1),
                            resnet_v1_block('block4', base_depth=512, num_units=3, stride=1)]

        elif self._num_layers == 101:
            self._blocks = [resnet_v1_block('block1', base_depth=64, num_units=3, stride=2),
                            resnet_v1_block('block2', base_depth=128, num_units=4, stride=2),
                            # use stride 1 for the last conv4 layer
                            resnet_v1_block('block3', base_depth=256, num_units=23, stride=1),
                            resnet_v1_block('block4', base_depth=512, num_units=3, stride=1)]

        elif self._num_layers == 152:
            self._blocks = [resnet_v1_block('block1', base_depth=64, num_units=3, stride=2),
                            resnet_v1_block('block2', base_depth=128, num_units=8, stride=2),
                            # use stride 1 for the last conv4 layer
                            resnet_v1_block('block3', base_depth=256, num_units=36, stride=1),
                            resnet_v1_block('block4', base_depth=512, num_units=3, stride=1)]

        else:
            # other numbers are not supported
            raise NotImplementedError

    def get_variables_to_restore(self, variables, var_keep_dic):
        variables_to_restore = []

        for v in variables:
            # exclude the first conv layer to swap RGB to BGR
            if v.name == (self._scope + '/conv1/weights:0'):
                self._variables_to_fix[v.name] = v
                continue
            if v.name.split(':')[0] in var_keep_dic:
                print('Variables restored: %s' % v.name)
                variables_to_restore.append(v)

        return variables_to_restore

    def fix_variables(self, sess, pretrained_model):
        print('Fix Resnet V1 layers..')
        with tf.variable_scope('Fix_Resnet_V1') as scope:
            with tf.device("/cpu:0"):
                # fix RGB to BGR
                conv1_rgb = tf.get_variable("conv1_rgb", [7, 7, 3, 64], trainable=False)
                restorer_fc = tf.train.Saver({self._scope + "/conv1/weights": conv1_rgb})
                restorer_fc.restore(sess, pretrained_model)

                sess.run(tf.assign(self._variables_to_fix[self._scope + '/conv1/weights:0'],
                                   tf.reverse(conv1_rgb, [2])))


def get_variables_in_checkpoint_file(file_name):
    try:
        reader = pywrap_tensorflow.NewCheckpointReader(file_name)
        var_to_shape_map = reader.get_variable_to_shape_map()
        return var_to_shape_map
    except Exception as e:  # pylint: disable=broad-except
        print(str(e))
        if "corrupted compressed block contents" in str(e):
            print("It's likely that your checkpoint file has been compressed "
                  "with SNAPPY.")


def main():
    ckpt_path = '/home/joe/git/slim_models/resnet_v1_50.ckpt'
    var_keep_dic = get_variables_in_checkpoint_file(ckpt_path)
    for key in var_keep_dic:
        print("tensor_name: ", key)


if __name__ == '__main__':
    main()

