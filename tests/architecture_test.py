# --------------------------------------------------------
# DenseCap-Tensorflow
# Written by InnerPeace
# This file is adapted from Linjie's work
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from lib.config import cfg
import tensorflow as tf
from lib.nets.resnet_v1 import resnetv1
from tests.roidata_test import get_data_test
import six
import numpy as np


def architecture_test():
    blob = get_data_test()
    tf.reset_default_graph()
    net = resnetv1(50)
    # net._build_network()
    net.create_architecture(mode='TEST', tag='pre')

    for n in tf.get_default_graph().as_graph_def().node:
        print(n.name)

    tfconfig = tf.ConfigProto(allow_soft_placement=True)
    tfconfig.gpu_options.allow_growth = True

    feed_dict = {net._image: blob['data'],
                 net._im_info: blob['im_info'],
                 net._gt_boxes: blob['gt_boxes'],
                 net._gt_phrases: blob['gt_phrases']}
    output = net._for_debug
    output.update({
        "image": net._image,
        "im_info": net._im_info,
        "gt_boxes": net._gt_boxes,
        "gt_phrases": net._gt_phrases
    })

    with tf.Session(config=tfconfig) as sess:
        init = tf.global_variables_initializer()
        sess.run(init)
        out = sess.run('DenseCap_ResNet50/Prediction/lstm/cap_init_state:0', feed_dict=feed_dict)
        print(out.shape)
        # out = sess.run(output, feed_dict=feed_dict)

        # for k, v in six.iteritems(out):
        #     print("name: {}               ==> {}".format(k, v.shape))
        #     # print("shape: {}".format(v.shape))
        #     if k == 'labels':
        #         # print(v)
        #         # print("first 5 example:")
        #         print(v[:5])
        #     if k == 'loss' or k == 'total_loss':
        #         print(k, v)


if __name__ == '__main__':
    architecture_test()
