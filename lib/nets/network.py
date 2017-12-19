# --------------------------------------------------------
# DenseCap-Tensorflow
# Written by InnerPeace
# This file is adapted from Linjie's work
# --------------------------------------------------------
# Tensorflow Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Xinlei Chen
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.contrib.slim import losses
from tensorflow.contrib.slim import arg_scope
import tensorflow.contrib.rnn as rnn

import numpy as np
import pdb

from lib.layers.snippets import generate_anchors_pre
from lib.layers.proposal_layer import proposal_layer
from lib.layers.proposal_top_layer import proposal_top_layer
from lib.layers.anchor_target_layer import anchor_target_layer
from lib.layers.proposal_target_layer import proposal_target_layer
from lib.layers.proposal_target_single_class_layer import proposal_target_single_class_layer
from lib.layers.sentence_data_layer import sentence_data_layer
from lib.utils.visualization import draw_bounding_boxes, draw_densecap

from lib.config import cfg


class Network(object):
    def __init__(self):
        self._predictions = {}
        self._losses = {}
        self._anchor_targets = {}
        self._proposal_targets = {}
        self._layers = {}
        self._gt_image = None
        self._act_summaries = []
        self._score_summaries = {}
        self._train_summaries = []
        self._event_summaries = {}
        self._variables_to_fix = {}

        self._image = tf.placeholder(tf.float32, shape=[1, None, None, 3])
        self._im_info = tf.placeholder(tf.float32, shape=[3])
        # add global roi
        if cfg.CONTEXT_FUSION:
            self._global_roi = tf.placeholder(tf.float32, shape=[1, 5])
        self._gt_boxes = tf.placeholder(tf.float32, shape=[None, 5])
        # add 2 for: <SOS> and <EOS>
        self._gt_phrases = tf.placeholder(tf.int32, shape=[None, cfg.MAX_WORDS])

        self._anchor_scales = cfg.ANCHOR_SCALES
        self._num_scales = len(self._anchor_scales)

        self._anchor_ratios = cfg.ANCHOR_RATIOS
        self._num_ratios = len(self._anchor_ratios)

        self._num_anchors = self._num_scales * self._num_ratios

        if cfg.DEBUG_ALL:
            self._for_debug = {}
            self._tag = 'pre'
            self._mode = 'TRAIN'
            self._num_classes = 1

    def _add_gt_image(self):
        # add back mean
        image = self._image + cfg.PIXEL_MEANS
        # BGR to RGB (opencv uses BGR)
        resized = tf.image.resize_bilinear(image, tf.to_int32(self._im_info[:2] / self._im_info[2]))
        self._gt_image = tf.reverse(resized, axis=[-1])

    def _add_gt_image_summary(self):
        # use a customized visualization function to visualize the boxes
        if self._gt_image is None:
            self._add_gt_image()
        image = tf.py_func(draw_bounding_boxes,
                           [self._gt_image, self._gt_boxes, self._im_info, self._gt_phrases],
                           tf.float32, name="gt_boxes")

        return tf.summary.image('GROUND_TRUTH', image)

    def _add_image_summary(self):
        # add back mean
        image = self._image + cfg.PIXEL_MEANS
        # BGR to RGB (opencv uses BGR)
        resized = tf.image.resize_bilinear(image, tf.to_int32(self._im_info[:2] / self._im_info[2]))
        gt_image = tf.reverse(resized, axis=[-1])
        img_wcap = tf.py_func(draw_densecap, [gt_image, self._predictions['cls_prob'],
            self._predictions['rois'], self._im_info, self._predictions['cap_probs'],
            self._predictions['bbox_pred']], tf.float32, name='image_summary')

        return tf.summary.image('TEMP_OUT', img_wcap)

    def _add_act_summary(self, tensor):
        tf.summary.histogram('ACT/' + tensor.op.name + '/activations', tensor)
        tf.summary.scalar('ACT/' + tensor.op.name + '/zero_fraction',
                          tf.nn.zero_fraction(tensor))

    def _add_score_summary(self, key, tensor):
        tf.summary.histogram('SCORE/' + tensor.op.name + '/' + key + '/scores', tensor)

    def _add_train_summary(self, var):
        tf.summary.histogram('TRAIN/' + var.op.name, var)

    def _reshape_layer(self, bottom, num_dim, name):
        input_shape = tf.shape(bottom)
        with tf.variable_scope(name) as scope:
            # change the channel to the caffe format
            to_caffe = tf.transpose(bottom, [0, 3, 1, 2])
            # then force it to have channel 2
            reshaped = tf.reshape(to_caffe,
                                  tf.concat(axis=0, values=[[1, num_dim, -1], [input_shape[2]]]))
            # then swap the channel back
            to_tf = tf.transpose(reshaped, [0, 2, 3, 1])
            return to_tf

    def _softmax_layer(self, bottom, name):
        if name.startswith('rpn_cls_prob_reshape'):
            input_shape = tf.shape(bottom)
            bottom_reshaped = tf.reshape(bottom, [-1, input_shape[-1]])
            reshaped_score = tf.nn.softmax(bottom_reshaped, name=name)
            return tf.reshape(reshaped_score, input_shape)
        return tf.nn.softmax(bottom, name=name)

    def _proposal_top_layer(self, rpn_cls_prob, rpn_bbox_pred, name):
        with tf.variable_scope(name) as scope:
            rois, rpn_scores = tf.py_func(proposal_top_layer,
                                          [rpn_cls_prob, rpn_bbox_pred, self._im_info,
                                           self._feat_stride, self._anchors, self._num_anchors],
                                          [tf.float32, tf.float32], name="proposal_top")
            rois.set_shape([cfg.TEST.RPN_TOP_N, 5])
            rpn_scores.set_shape([cfg.TEST.RPN_TOP_N, 1])

        return rois, rpn_scores

    def _proposal_layer(self, rpn_cls_prob, rpn_bbox_pred, name):
        with tf.variable_scope(name) as scope:
            rois, rpn_scores = tf.py_func(proposal_layer,
                                          [rpn_cls_prob, rpn_bbox_pred, self._im_info, self._mode,
                                           self._feat_stride, self._anchors, self._num_anchors],
                                          [tf.float32, tf.float32], name="proposal")
            rois.set_shape([None, 5])
            rpn_scores.set_shape([None, 1])

            if cfg.DEBUG_ALL:
                self._for_debug['proposal_rois'] = rois
                self._for_debug['proposal_rpn_scores'] = rpn_scores

        return rois, rpn_scores

    # Only use it if you have roi_pooling op written in tf.image
    def _roi_pool_layer(self, bootom, rois, name):
        with tf.variable_scope(name) as scope:
            return tf.image.roi_pooling(bootom, rois,
                                        pooled_height=cfg.POOLING_SIZE,
                                        pooled_width=cfg.POOLING_SIZE,
                                        spatial_scale=1. / 16.)[0]

    def _crop_pool_layer(self, bottom, rois, name):
        with tf.variable_scope(name) as scope:
            batch_ids = tf.squeeze(tf.slice(rois, [0, 0], [-1, 1], name="batch_id"), [1])
            # Get the normalized coordinates of bounding boxes
            bottom_shape = tf.shape(bottom)
            height = (tf.to_float(bottom_shape[1]) - 1.) * np.float32(self._feat_stride[0])
            width = (tf.to_float(bottom_shape[2]) - 1.) * np.float32(self._feat_stride[0])
            x1 = tf.slice(rois, [0, 1], [-1, 1], name="x1") / width
            y1 = tf.slice(rois, [0, 2], [-1, 1], name="y1") / height
            x2 = tf.slice(rois, [0, 3], [-1, 1], name="x2") / width
            y2 = tf.slice(rois, [0, 4], [-1, 1], name="y2") / height
            # Won't be back-propagated to rois anyway, but to save time
            bboxes = tf.stop_gradient(tf.concat([y1, x1, y2, x2], axis=1))
            pre_pool_size = cfg.POOLING_SIZE * 2
            crops = tf.image.crop_and_resize(bottom, bboxes, tf.to_int32(batch_ids),
                                             [pre_pool_size, pre_pool_size],
                                             name="crops")

        # slim.max_pool2d has stride 2 in default
        return slim.max_pool2d(crops, [2, 2], padding='SAME')

    def _dropout_layer(self, bottom, name, ratio=0.5):
        return tf.nn.dropout(bottom, ratio, name=name)

    def _anchor_target_layer(self, rpn_cls_score, name):
        with tf.variable_scope(name) as scope:
            rpn_labels, rpn_bbox_targets, rpn_bbox_inside_weights, rpn_bbox_outside_weights = tf.py_func(
                anchor_target_layer,
                [rpn_cls_score, self._gt_boxes, self._im_info, self._feat_stride, self._anchors, self._num_anchors],
                [tf.float32, tf.float32, tf.float32, tf.float32],
                name="anchor_target")

            rpn_labels.set_shape([1, 1, None, None])
            rpn_bbox_targets.set_shape([1, None, None, self._num_anchors * 4])
            rpn_bbox_inside_weights.set_shape([1, None, None, self._num_anchors * 4])
            rpn_bbox_outside_weights.set_shape([1, None, None, self._num_anchors * 4])

            rpn_labels = tf.to_int32(rpn_labels, name="to_int32")
            self._anchor_targets['rpn_labels'] = rpn_labels
            self._anchor_targets['rpn_bbox_targets'] = rpn_bbox_targets
            self._anchor_targets['rpn_bbox_inside_weights'] = rpn_bbox_inside_weights
            self._anchor_targets['rpn_bbox_outside_weights'] = rpn_bbox_outside_weights

            self._score_summaries.update(self._anchor_targets)

        if cfg.DEBUG_ALL:
            self._for_debug['rpn_labels'] = rpn_labels
            self._for_debug['rpn_bbox_targets'] = rpn_bbox_targets
            self._for_debug['rpn_bbox_inside_weights'] = rpn_bbox_inside_weights
            self._for_debug['rpn_bbox_outside_weights'] = rpn_bbox_outside_weights

        return rpn_labels

    # TODO: about to delete
    def _proposal_target_layer(self, rois, roi_scores, name):
        with tf.variable_scope(name) as scope:
            rois, roi_scores, labels, bbox_targets, bbox_inside_weights, bbox_outside_weights = tf.py_func(
                proposal_target_layer,
                [rois, roi_scores, self._gt_boxes, self._num_classes],
                [tf.float32, tf.float32, tf.float32, tf.float32, tf.float32, tf.float32],
                name="proposal_target")

            rois.set_shape([cfg.TRAIN.BATCH_SIZE, 5])
            roi_scores.set_shape([cfg.TRAIN.BATCH_SIZE])
            labels.set_shape([cfg.TRAIN.BATCH_SIZE, 1])
            bbox_targets.set_shape([cfg.TRAIN.BATCH_SIZE, self._num_classes * 4])
            bbox_inside_weights.set_shape([cfg.TRAIN.BATCH_SIZE, self._num_classes * 4])
            bbox_outside_weights.set_shape([cfg.TRAIN.BATCH_SIZE, self._num_classes * 4])

            self._proposal_targets['rois'] = rois
            self._proposal_targets['labels'] = tf.to_int32(labels, name="to_int32")
            self._proposal_targets['bbox_targets'] = bbox_targets
            self._proposal_targets['bbox_inside_weights'] = bbox_inside_weights
            self._proposal_targets['bbox_outside_weights'] = bbox_outside_weights

            self._score_summaries.update(self._proposal_targets)

            return rois, roi_scores

    # TODO: roi_scores are not needed. About to delete
    def _proposal_target_single_class_layer(self, rois, roi_scores, name):

        with tf.variable_scope(name) as scope:
            rois, roi_scores, labels, bbox_targets, \
                bbox_inside_weights, bbox_outside_weights, clss, phrases = tf.py_func(
                    proposal_target_single_class_layer,
                    [rois, roi_scores, self._gt_boxes, self._gt_phrases],
                    [tf.float32, tf.float32, tf.float32, tf.float32,
                     tf.float32, tf.float32, tf.int32, tf.int32],
                    name="proposal_target_single_class")

            rois.set_shape([None, 5])
            roi_scores.set_shape([None])
            labels.set_shape([None, 1])
            phrases.set_shape([None, cfg.MAX_WORDS])
            bbox_targets.set_shape([None, 4])
            bbox_inside_weights.set_shape([None, 4])
            bbox_outside_weights.set_shape([None, 4])
            clss.set_shape([None, 1])

            # phrases = tf.to_int32(phrases, name='to_int32')
            labels = tf.to_int32(labels, name="to_int32")
            # clss = tf.to_int32(clss, name="to_int32")

            self._proposal_targets['rois'] = rois
            self._proposal_targets['labels'] = labels
            self._proposal_targets['clss'] = clss
            self._proposal_targets['phrases'] = phrases
            self._proposal_targets['bbox_targets'] = bbox_targets
            self._proposal_targets['bbox_inside_weights'] = bbox_inside_weights
            self._proposal_targets['bbox_outside_weights'] = bbox_outside_weights

            self._score_summaries.update(self._proposal_targets)

            if cfg.DEBUG_ALL:
                self._for_debug['rois'] = rois
                self._for_debug['labels'] = tf.to_int32(labels, name="to_int32")
                self._for_debug['clss'] = tf.to_int32(clss, name="to_int32")
                self._for_debug['phrases'] = phrases
                self._for_debug['bbox_targets'] = bbox_targets
                self._for_debug['bbox_inside_weights'] = bbox_inside_weights
                self._for_debug['bbox_outside_weights'] = bbox_outside_weights

            return rois, labels, phrases

    def _sentence_data_layer(self, name,
                             time_steps=cfg.TIME_STEPS, mode=cfg.CONTEXT_MODE):

        num_regions = self._roi_labels.shape[0]
        with tf.variable_scope(name) as scope:
            input_sentence, target_sentence, cont_sentence, cont_bbox = \
                tf.py_func(sentence_data_layer,
                           [self._roi_labels, self._roi_phrases, time_steps, mode],
                           [tf.float32, tf.float32, tf.float32, tf.float32],
                           name='sentence_data')
            if cfg.CONTEXT_MODE == 'concat':
                input_sentence.set_shape([num_regions, cfg.TIME_STEPS - 1])
            elif cfg.CONTEXT_MODE == 'repeat':
                input_sentence.set_shape([num_regions, cfg.TIME_STEPS])

            target_sentence.set_shape([num_regions, cfg.TIME_STEPS])
            cont_sentence.set_shape([num_regions, cfg.TIME_STEPS])
            cont_bbox.set_shape([num_regions, cfg.TIME_STEPS])

            input_sentence = tf.to_int32(input_sentence, name='to_int32')
            target_sentence = tf.to_int32(target_sentence, name='to_int32')

            self._sentence_data = {}
            self._sentence_data['input_sentence'] = input_sentence
            self._sentence_data['target_sentence'] = target_sentence
            self._sentence_data['cont_sentence'] = cont_sentence
            self._sentence_data['cont_bbox'] = cont_bbox

            self._score_summaries.update(self._sentence_data)

            if cfg.DEBUG_ALL:
                self._for_debug['input_sentence'] = input_sentence
                self._for_debug['target_sentence'] = target_sentence
                self._for_debug['cont_sentence'] = cont_sentence
                self._for_debug['cont_bbox'] = cont_bbox

        return input_sentence

    def _embed_caption_layer(self, fc7, input_sentence, initializer, is_training):
        """
        comput image context feature and embed input sentence,
        do 'concat' or 'repeat' image feature
        """
        region_features = slim.fully_connected(fc7, cfg.EMBED_DIM,
                                              weights_initializer=initializer,
                                              trainable=is_training,
                                              activation_fn=None, scope='region_features')
        if cfg.CONTEXT_FUSION:
            # global_feature [1, cfg.EMBED_DIM(512)]
            global_feature, region_features = tf.split(region_features, [1, -1], axis=0)
            batch_size = tf.shape(region_features)[0]
            # global_feature_rep [batch_size(256), cfg.EMBED_DIM(512)]
            global_feature_rep = tf.tile(global_feature, [batch_size, 1])
            gfeat_lstm_cell = rnn.BasicLSTMCell(cfg.EMBED_DIM, forget_bias=1.0,
                state_is_tuple=True)
        else:
            batch_size = tf.shape(region_features)[0]

        with tf.variable_scope('seq_embedding'), tf.device("/cpu:0"):
            if cfg.INIT_BY_GLOVE and is_training:
                glove_path = cfg.DATA_DIR + "/glove.trimmed.{}.npz".format(cfg.GLOVE_DIM)
                glove = np.load(glove_path)['glove'].astype(np.float32)
                print("load pre-trained glove from {}, with shape: {}".format(glove_path,
                    glove.shape))
                if not cfg.KEEP_AS_GLOVE_DIM:
                    g_mean = np.mean(glove, axis=1)
                    g_std = np.std(glove, axis=1)
                    expand_glove = np.random.normal(g_mean, g_std,
                         (cfg.EMBED_DIM, cfg.VOCAB_SIZE + 3))
                    expand_glove[:glove.shape[1], :] = glove.T
                    embed_initializer = tf.constant_initializer(expand_glove.T)
                else:
                    embed_initializer = tf.constant_initializer(glove)
                    assert cfg.EMBED_DIM == cfg.GLOVE_DIM
            else:
                embed_initializer = initializer

            self._embedding = tf.get_variable("embedding",
                                              # 0,1,2 for pad sos eof respectively.
                                              [cfg.VOCAB_SIZE + 3, cfg.EMBED_DIM],
                                              initializer=embed_initializer,
                                              trainable=is_training,
                                              dtype=tf.float32
                                              )
            print("Shape of embedding is {}".format(self._embedding.shape))
            # independent decoder and encoder for word representation.
            # self._inverse_embed = tf.get_variable('inverse_embed',
                                                  # [cfg.EMBED_DIM, cfg.VOCAB_SIZE + 3],
                                                  # initializer=initializer)
            embed_input_sentence = tf.nn.embedding_lookup(self._embedding,
                                                          input_sentence)

        location_lstm_cell = rnn.BasicLSTMCell(cfg.EMBED_DIM, forget_bias=1.0, state_is_tuple=True)
        caption_lstm_cell = rnn.BasicLSTMCell(cfg.EMBED_DIM, forget_bias=1.0, state_is_tuple=True)

        # add dropout in rnn
        # cell = rnn.DropoutWrapper(caption_lstm_cell,
        #                           input_keep_prob=prob,
        #                           output_keep_prob=prob,)

        with tf.variable_scope("lstm") as lstm_scope:
            # Feed the image embeddings to set the intial LSTM state
            cap_zero_state = caption_lstm_cell.zero_state(
                batch_size=batch_size, dtype=tf.float32)
            loc_zero_state = location_lstm_cell.zero_state(
                batch_size=batch_size, dtype=tf.float32)
            with tf.variable_scope('cap_lstm'):
                _, cap_init_state = caption_lstm_cell(region_features, cap_zero_state)
            with tf.variable_scope('loc_lstm'):
                _, loc_init_state = location_lstm_cell(region_features, loc_zero_state)

            if cfg.CONTEXT_FUSION:
                gfeat_zero_state = gfeat_lstm_cell.zero_state(
                    batch_size=batch_size, dtype=tf.float32)
                with tf.variable_scope('gfeat_lstm'):
                    # NOTE: gfeat_init_state [batch_size(256), cfg.EMBED_DIM(512)]
                    _, gfeat_init_state = gfeat_lstm_cell(global_feature_rep,
                        gfeat_zero_state)

            # Allow the LSTM variable to be reused
            lstm_scope.reuse_variables()

            if self._mode == 'TRAIN':
                if cfg.CONTEXT_MODE == 'concat':
                    im_context = tf.expand_dims(region_features, axis=1)
                    im_concat_words = tf.concat([im_context, embed_input_sentence],
                        axis=1)
                    if cfg.CONTEXT_FUSION:
                        global_feature_rep = tf.expand_dims(global_feature_rep, axis=1)
                        global_concat_words = tf.concat([global_feature_rep,
                            embed_input_sentence], axis=1)
                else:
                    raise NotImplementedError

                # TODO(wu) need to check again about the sequence length
                sequence_length = self._sequence_length(input_sentence) + 1
                cap_outputs, cap_states = tf.nn.dynamic_rnn(caption_lstm_cell, im_concat_words,
                                                            sequence_length=sequence_length,
                                                            dtype=tf.float32,
                                                            scope='captoin_lstm')

                loc_outputs, loc_states = tf.nn.dynamic_rnn(location_lstm_cell, im_concat_words,
                                                            sequence_length=sequence_length,
                                                            dtype=tf.float32,
                                                            scope='location_lstm')
                if cfg.CONTEXT_FUSION:
                    gfeat_outputs, gfeat_states = tf.nn.dynamic_rnn(gfeat_lstm_cell,
                        global_concat_words,
                        sequence_length=sequence_length,
                        dtype=tf.float32,
                        scope='gfeat_lstm')
                    # for now, it only support mode "sum"
                    if cfg.CONTEXT_FUSION_MODE == "sum":
                        cap_outputs = gfeat_outputs + cap_outputs
                    else:
                        raise NotImplementedError

                # OUT OF MEMORY ON GPU
                # inv_embedding = tf.tile(tf.expand_dims(tf.transpose(self._embedding),
                #                                        [0]),
                #                         [tf.shape(caption_outputs)[0], 1, 1])
                # predict_caption = tf.matmul(caption_outputs, inv_embedding)

                # cap_logits = tf.matmul(tf.reshape(cap_outputs, [-1, cfg.EMBED_DIM]),
                                       # tf.transpose(self._embedding))
                # cap_logits = tf.reshape(predict_cap_reshape,
                #                         [-1, cfg.TIME_STEPS, cfg.VOCAB_SIZE + 3])

                # problematic to always slice output of the last slice
                # loc_out_slice = tf.slice(loc_outputs, [0, cfg.TIME_STEPS - 1, 0], [-1, 1, -1])
                # loc_out_slice = tf.squeeze(loc_out_slice, [1])
                # BETTER SOLUTION
                cont_bbox = tf.expand_dims(self._sentence_data['cont_bbox'], axis=2)
                loc_out_slice = tf.reduce_sum(loc_outputs * cont_bbox, axis=1)

            elif self._mode == 'TEST':
                # In inference or test mode, use concatenated states for convenient feeding
                tf.concat(values=cap_init_state, axis=1, name='cap_init_state')
                tf.concat(values=loc_init_state, axis=1, name='loc_init_state')

                # placeholder for feeding a batch of concatnated states.
                cap_state_feed = tf.placeholder(dtype=tf.float32,
                                                shape=[None, sum(caption_lstm_cell.state_size)],
                                                name='cap_state_feed')
                loc_state_feed = tf.placeholder(dtype=tf.float32,
                                                shape=[None, sum(location_lstm_cell.state_size)],
                                                name='loc_state_feed')
                cap_state_tuple = tf.split(value=cap_state_feed, num_or_size_splits=2, axis=1)
                loc_state_tuple = tf.split(value=loc_state_feed, num_or_size_splits=2, axis=1)

                # Run a single LSTM step
                seq_embedding = tf.squeeze(embed_input_sentence, axis=[1])
                cap_outputs, cap_state_tuple = caption_lstm_cell(
                    inputs=seq_embedding,
                    state=cap_state_tuple
                )
                loc_outputs, loc_state_tuple = location_lstm_cell(
                    inputs=seq_embedding,
                    state=loc_state_tuple
                )

                # Concatenate the resulting state
                tf.concat(values=cap_state_tuple, axis=1, name='cap_state')
                tf.concat(values=loc_state_tuple, axis=1, name='loc_state')
                loc_out_slice = loc_outputs
                # NOTE CONTEXT FUSION
                if cfg.CONTEXT_FUSION:
                    tf.concat(values=gfeat_init_state, axis=1, name='gfeat_init_state')
                    gfeat_state_feed = tf.placeholder(dtype=tf.float32,
                        shape=[None, sum(gfeat_lstm_cell.state_size)],
                        name='gfeat_state_feed')
                    gfeat_state_tuple = tf.split(value=gfeat_state_feed, num_or_size_splits=2,
                        axis=1)
                    gfeat_outputs, gfeat_state_tuple = gfeat_lstm_cell(
                        inputs=seq_embedding,
                        state=gfeat_state_tuple)
                    tf.concat(values=gfeat_state_tuple, axis=1, name='gfeat_state')
                    if cfg.CONTEXT_FUSION_MODE == "sum":
                        cap_outputs += gfeat_outputs

            else:
                raise NotImplementedError

            # caption logits
            cap_logits = tf.matmul(tf.reshape(cap_outputs, [-1, cfg.EMBED_DIM]),
                tf.transpose(self._embedding), name='cap_logits')
            # cap_logits = tf.matmul(tf.reshape(cap_outputs, [-1, cfg.EMBED_DIM]),
                # self._inverse_embed, name='cap_logits')
            cap_probs = tf.nn.softmax(cap_logits, name='cap_probs')

        bbox_pred = slim.fully_connected(loc_out_slice, 4,
                                         weights_initializer=initializer,
                                         trainable=is_training,
                                         activation_fn=None, scope='bbox_pred')

        self._predictions['bbox_pred'] = bbox_pred
        self._predictions['cap_probs'] = cap_probs
        self._predictions['predict_caption'] = cap_logits

        if cfg.DEBUG_ALL:
            self._for_debug['embedding'] = self._embedding
            self._for_debug['embed_input_sentence'] = embed_input_sentence
            self._for_debug['fc8'] = region_features
            # self._for_debug['im_concat_words'] = im_concat_words
            self._for_debug['captoin_outputs'] = cap_outputs
            self._for_debug['loc_outputs'] = loc_outputs
            self._for_debug['loc_out_slice'] = loc_out_slice
            self._for_debug['bbox_pred'] = bbox_pred
            self._for_debug['predict_caption'] = cap_logits

    def _sequence_length(self, input_sentence):
        return tf.reduce_sum(tf.cast(tf.cast(input_sentence, tf.bool), tf.int32), axis=1)

    def _anchor_component(self):
        with tf.variable_scope('ANCHOR_' + self._tag) as scope:
            # just to get the shape right
            height = tf.to_int32(tf.ceil(self._im_info[0] / np.float32(self._feat_stride[0])))
            width = tf.to_int32(tf.ceil(self._im_info[1] / np.float32(self._feat_stride[0])))
            anchors, anchor_length = tf.py_func(generate_anchors_pre,
                                                [height, width,
                                                 self._feat_stride, self._anchor_scales, self._anchor_ratios],
                                                [tf.float32, tf.int32], name="generate_anchors")
            anchors.set_shape([None, 4])
            anchor_length.set_shape([])
            self._anchors = anchors
            self._anchor_length = anchor_length

        if cfg.DEBUG_ALL:
            self._for_debug['anchors'] = anchors

    def _build_network(self, is_training=True):
        # select initializers
        if cfg.TRAIN.WEIGHT_INITIALIZER == 'truncated':
            initializer = tf.truncated_normal_initializer(mean=0.0, stddev=0.01)
            # initializer_bbox = tf.truncated_normal_initializer(mean=0.0, stddev=0.001)
        elif cfg.TRAIN.WEIGHT_INITIALIZER == 'normal':
            initializer = tf.random_normal_initializer(mean=0.0, stddev=0.01)
            # initializer_bbox = tf.random_normal_initializer(mean=0.0, stddev=0.001)
        else:
            initializer = tf.contrib.layers.xavier_initializer()
            # initializer_bbox = tf.contrib.layers.xavier_initializer()

        net_conv = self._image_to_head(is_training)
        with tf.variable_scope(self._scope + '/Extraction'):
            # build the anchors for the image
            self._anchor_component()
            # region proposal network
            rois = self._region_proposal(net_conv, is_training, initializer)
            # region of interest pooling
            if cfg.POOLING_MODE == 'crop':
                pool5 = self._crop_pool_layer(net_conv, rois, "pool5")
            else:
                raise NotImplementedError

            if self._mode == 'TRAIN':
                # sentence data layer
                input_sentence = self._sentence_data_layer('sentence_data')
            elif self._mode == 'TEST':
                input_feed = tf.placeholder(dtype=tf.int32,
                                            shape=[None],
                                            name='input_feed')
                input_sentence = tf.expand_dims(input_feed, 1)
            else:
                raise NotImplementedError

        fc7 = self._head_to_tail(pool5, is_training)
        with tf.variable_scope(self._scope + '/Prediction'):
            # add context fusion
            if cfg.CONTEXT_FUSION:
                # global feature after "head_to_tail" is dumped
                _, fc7_1 = tf.split(fc7, [1, -1], axis=0)
            else:
                fc7_1 = fc7
            # region classification
            cls_prob = self._region_classification(fc7_1, is_training,
                                                   initializer)
            self._embed_caption_layer(fc7, input_sentence, initializer, is_training)

        self._score_summaries.update(self._predictions)

        if cfg.DEBUG_ALL:
            self._for_debug['pool5'] = pool5
            self._for_debug['cls_prob'] = cls_prob

        return rois, cls_prob

    def _smooth_l1_loss(self, bbox_pred, bbox_targets, bbox_inside_weights, bbox_outside_weights, sigma=1.0, dim=[1]):
        sigma_2 = sigma ** 2
        box_diff = bbox_pred - bbox_targets
        in_box_diff = bbox_inside_weights * box_diff
        abs_in_box_diff = tf.abs(in_box_diff)
        smoothL1_sign = tf.stop_gradient(tf.to_float(tf.less(abs_in_box_diff, 1. / sigma_2)))
        in_loss_box = tf.pow(in_box_diff, 2) * (sigma_2 / 2.) * smoothL1_sign \
            + (abs_in_box_diff - (0.5 / sigma_2)) * (1. - smoothL1_sign)
        out_loss_box = bbox_outside_weights * in_loss_box
        loss_box = tf.reduce_mean(tf.reduce_sum(
            out_loss_box,
            axis=dim
        ))
        return loss_box

    def _add_losses(self, sigma_rpn=3.0):
        with tf.variable_scope('LOSS_' + self._tag) as scope:
            # RPN, class loss
            rpn_cls_score = tf.reshape(self._predictions['rpn_cls_score_reshape'], [-1, 2])
            rpn_label = tf.reshape(self._anchor_targets['rpn_labels'], [-1])
            rpn_select = tf.where(tf.not_equal(rpn_label, -1))
            rpn_cls_score = tf.reshape(tf.gather(rpn_cls_score, rpn_select), [-1, 2])
            rpn_label = tf.reshape(tf.gather(rpn_label, rpn_select), [-1])
            rpn_cross_entropy = tf.reduce_mean(
                tf.nn.sparse_softmax_cross_entropy_with_logits(logits=rpn_cls_score,
                                                               labels=rpn_label))

            # RPN, bbox loss
            rpn_bbox_pred = self._predictions['rpn_bbox_pred']
            rpn_bbox_targets = self._anchor_targets['rpn_bbox_targets']
            rpn_bbox_inside_weights = self._anchor_targets['rpn_bbox_inside_weights']
            rpn_bbox_outside_weights = self._anchor_targets['rpn_bbox_outside_weights']
            rpn_loss_box = self._smooth_l1_loss(rpn_bbox_pred, rpn_bbox_targets, rpn_bbox_inside_weights,
                                                rpn_bbox_outside_weights,
                                                sigma=sigma_rpn, dim=[1, 2, 3])

            # class loss
            cls_score = self._predictions["cls_score"]
            label = tf.reshape(self._proposal_targets["clss"], [-1])
            clss_cross_entropy = tf.reduce_mean(
                tf.nn.sparse_softmax_cross_entropy_with_logits(logits=cls_score, labels=label))

            # bbox loss
            bbox_pred = self._predictions['bbox_pred']
            bbox_targets = self._proposal_targets['bbox_targets']
            # bbox_inside_weights = self._proposal_targets['bbox_inside_weights']
            # bbox_outside_weights = self._proposal_targets['bbox_outside_weights']
            loss_box = self._smooth_l1_loss(bbox_pred, bbox_targets,
                                            1., 1.)

            # caption loss
            # shape [None*12, 10003]
            target_sentence = self._sentence_data['target_sentence']
            predict_caption = self._predictions['predict_caption']
            target_sentence = tf.reshape(target_sentence, [-1])
            cap_mask = tf.reshape(tf.to_float(tf.cast(target_sentence, tf.bool)), [-1])
            captoin_cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=predict_caption,
                                                                                   labels=target_sentence)
            caption_loss = tf.div(tf.reduce_sum(tf.multiply(captoin_cross_entropy, cap_mask)),
                                  tf.reduce_sum(cap_mask), name='caption_loss')

            self._losses['clss_cross_entropy'] = clss_cross_entropy
            self._losses['loss_box'] = loss_box
            self._losses['rpn_cross_entropy'] = rpn_cross_entropy
            self._losses['rpn_loss_box'] = rpn_loss_box
            self._losses['caption_loss'] = caption_loss

            loss = cfg.LOSS.CAP_W * caption_loss \
                + cfg.LOSS.CLS_W * clss_cross_entropy \
                + cfg.LOSS.BBOX_W * loss_box \
                + cfg.LOSS.RPN_CLS_W * rpn_cross_entropy \
                + cfg.LOSS.RPN_BBOX_W * rpn_loss_box
            regularization_loss = tf.add_n(tf.losses.get_regularization_losses(), 'regu')
            self._losses['total_loss'] = loss + regularization_loss

            self._event_summaries.update(self._losses)

            if cfg.DEBUG_ALL:
                self._for_debug['loss'] = loss
                self._for_debug['total_loss'] = loss

        return loss

    def _region_proposal(self, net_conv, is_training, initializer):
        rpn = slim.conv2d(net_conv, cfg.RPN_CHANNELS, [3, 3], trainable=is_training,
                          weights_initializer=initializer,
                          scope="rpn_conv/3x3")
        self._act_summaries.append(rpn)
        rpn_cls_score = slim.conv2d(rpn, self._num_anchors * 2, [1, 1], trainable=is_training,
                                    weights_initializer=initializer,
                                    padding='VALID', activation_fn=None, scope='rpn_cls_score')
        # change it so that the score has 2 as its channel size
        rpn_cls_score_reshape = self._reshape_layer(rpn_cls_score, 2, 'rpn_cls_score_reshape')
        rpn_cls_prob_reshape = self._softmax_layer(rpn_cls_score_reshape, "rpn_cls_prob_reshape")
        # rpn_cls_pred = tf.argmax(tf.reshape(rpn_cls_score_reshape, [-1, 2]), axis=1, name="rpn_cls_pred")
        rpn_cls_prob = self._reshape_layer(rpn_cls_prob_reshape, self._num_anchors * 2, "rpn_cls_prob")
        rpn_bbox_pred = slim.conv2d(rpn, self._num_anchors * 4, [1, 1], trainable=is_training,
                                    weights_initializer=initializer,
                                    padding='VALID', activation_fn=None, scope='rpn_bbox_pred')
        if is_training:
            rois, roi_scores = self._proposal_layer(rpn_cls_prob, rpn_bbox_pred, "rois")
            rpn_labels = self._anchor_target_layer(rpn_cls_score, "anchor")
            # Try to have a deterministic order for the computing graph, for reproducibility
            with tf.control_dependencies([rpn_labels]):
                # rois, _ = self._proposal_target_layer(rois, roi_scores, "rpn_rois")
                rois, labels, phrases = self._proposal_target_single_class_layer(rois, roi_scores, "rpn_rois")

                self._roi_labels = labels
                self._roi_phrases = phrases
        else:
            if cfg.TEST.MODE == 'nms':
                rois, _ = self._proposal_layer(rpn_cls_prob, rpn_bbox_pred, "rois")
            elif cfg.TEST.MODE == 'top':
                rois, _ = self._proposal_top_layer(rpn_cls_prob, rpn_bbox_pred, "rois")
            else:
                raise NotImplementedError

        self._predictions["rpn_cls_score"] = rpn_cls_score
        self._predictions["rpn_cls_score_reshape"] = rpn_cls_score_reshape
        self._predictions["rpn_cls_prob"] = rpn_cls_prob
        # self._predictions["rpn_cls_pred"] = rpn_cls_pred
        self._predictions["rpn_bbox_pred"] = rpn_bbox_pred
        self._predictions["rois"] = rois

        # NOTE: add context feature
        if cfg.CONTEXT_FUSION:
            rois = tf.concat((self._global_roi, rois), axis=0)
            print("Using context fusion, with shape of rois: {}".format(rois.shape))

        if cfg.DEBUG_ALL:
            self._for_debug['rpn'] = rpn
            self._for_debug['rpn_cls_score'] = rpn_cls_score
            self._for_debug['rpn_cls_prob'] = rpn_cls_prob
            self._for_debug['rpn_cls_prob_reshape'] = rpn_cls_prob_reshape
            self._for_debug['rpn_cls_score_reshape'] = rpn_cls_score_reshape
            self._for_debug['rpn_bbox_pred'] = rpn_bbox_pred
        return rois

    # TODO: clear stuff
    def _region_classification(self, fc7, is_training, initializer):
        # predict two class: fg or bg
        cls_score = slim.fully_connected(fc7, self._num_classes + 1,
                                         weights_initializer=initializer,
                                         trainable=is_training,
                                         activation_fn=None, scope='cls_score')
        cls_prob = self._softmax_layer(cls_score, "cls_prob")
        # cls_pred = tf.argmax(cls_score, axis=1, name="cls_pred")
        # bbox_pred = slim.fully_connected(fc7, self._num_classes * 4,
        #                                  weights_initializer=initializer_bbox,
        #                                  trainable=is_training,
        #                                  activation_fn=None, scope='bbox_pred')

        self._predictions["cls_score"] = cls_score
        # self._predictions["cls_pred"] = cls_pred
        self._predictions["cls_prob"] = cls_prob
        # self._predictions["bbox_pred"] = bbox_pred

        return cls_prob

    def _image_to_head(self, is_training, reuse=None):
        raise NotImplementedError

    def _head_to_tail(self, pool5, is_training, reuse=None):
        raise NotImplementedError

    def create_architecture(self, mode, num_classes=1, tag=None,
                            ):
        self._tag = tag

        self._num_classes = num_classes
        self._mode = mode

        training = mode == 'TRAIN'
        testing = mode == 'TEST'

        assert tag != None

        # handle most of the regularizers here
        weights_regularizer = tf.contrib.layers.l2_regularizer(cfg.TRAIN.WEIGHT_DECAY)
        if cfg.TRAIN.BIAS_DECAY:
            biases_regularizer = weights_regularizer
        else:
            biases_regularizer = tf.no_regularizer

        # list as many types of layers as possible, even if they are not used now
        with arg_scope([slim.conv2d, slim.conv2d_in_plane,
                        slim.conv2d_transpose, slim.separable_conv2d, slim.fully_connected],
                       weights_regularizer=weights_regularizer,
                       biases_regularizer=biases_regularizer,
                       biases_initializer=tf.constant_initializer(0.0)):
            rois, cls_prob = self._build_network(training)

        layers_to_output = {'rois': rois}

        for var in tf.trainable_variables():
            self._train_summaries.append(var)

        if testing:
            # stds = np.tile(np.array(cfg.TRAIN.BBOX_NORMALIZE_STDS), (self._num_classes))
            stds = np.array(cfg.TRAIN.BBOX_NORMALIZE_STDS)
            # means = np.tile(np.array(cfg.TRAIN.BBOX_NORMALIZE_MEANS), (self._num_classes))
            means = np.array(cfg.TRAIN.BBOX_NORMALIZE_MEANS)
            self._predictions["bbox_pred"] *= stds
            self._predictions["bbox_pred"] += means
        else:
            self._add_losses()
            layers_to_output.update(self._losses)

            val_summaries = []
            with tf.device("/cpu:0"):
                # val_summaries.append(self._add_gt_image_summary())
                val_summaries.append(self._add_image_summary())
                for key, var in self._event_summaries.items():
                    val_summaries.append(tf.summary.scalar(key, var))
                for key, var in self._score_summaries.items():
                    self._add_score_summary(key, var)
                for var in self._act_summaries:
                    self._add_act_summary(var)
                for var in self._train_summaries:
                    self._add_train_summary(var)

            self._summary_op = tf.summary.merge_all()
            self._summary_op_val = tf.summary.merge(val_summaries)

        layers_to_output.update(self._predictions)

        return layers_to_output

    def get_variables_to_restore(self, variables, var_keep_dic):
        raise NotImplementedError

    def fix_variables(self, sess, pretrained_model):
        raise NotImplementedError

    # Extract the head feature maps, for example for vgg16 it is conv5_3
    # only useful during testing mode
    def extract_head(self, sess, image):
        feed_dict = {self._image: image}
        feat = sess.run(self._layers["head"], feed_dict=feed_dict)
        return feat

    # only useful during testing mode
    def test_image(self, sess, image, im_info):
        feed_dict = {self._image: image,
                     self._im_info: im_info}

        cls_score, cls_prob, bbox_pred, rois = sess.run([self._predictions["cls_score"],
                                                         self._predictions['cls_prob'],
                                                         self._predictions['bbox_pred'],
                                                         self._predictions['rois']],
                                                        feed_dict=feed_dict)
        return cls_score, cls_prob, bbox_pred, rois

    def feed_image(self, sess, image, im_info):
        feed_dict = {self._image: image,
                     self._im_info: im_info}
        fetch_list = [
            '%s/Prediction/lstm/cap_init_state:0' % self._scope,
            '%s/Prediction/lstm/loc_init_state:0' % self._scope,
            self._predictions['cls_prob'],
            self._predictions['rois']]
        if cfg.CONTEXT_FUSION:
            feed_dict.update({self._global_roi: np.array([[0., 0., 0., im_info[1] - 1,
                        im_info[0] - 1]], dtype=np.float32)})
            fetch_list.append('%s/Prediction/lstm/gfeat_init_state:0' % self._scope)

        fetch = sess.run(fetch_list, feed_dict=feed_dict)

        return fetch

    def inference_step(self, sess, input_feed, cap_state_feed, loc_state_feed,
                        gfeat_state_feed=None):
        feed_dict = {'%s/Extraction/input_feed:0' % self._scope: input_feed,
                     '%s/Prediction/lstm/cap_state_feed:0' % self._scope: cap_state_feed,
                     '%s/Prediction/lstm/loc_state_feed:0' % self._scope: loc_state_feed}
        fetch_list = ['%s/Prediction/lstm/cap_probs:0' % self._scope,
                   self._predictions['bbox_pred'],
                   '%s/Prediction/lstm/cap_state:0' % self._scope,
                   '%s/Prediction/lstm/loc_state:0' % self._scope]
        if cfg.CONTEXT_FUSION:
            feed_dict.update({'%s/Prediction/lstm/gfeat_state_feed:0' % self._scope:
                              gfeat_state_feed})
            fetch_list.append('%s/Prediction/lstm/gfeat_state:0' % self._scope)
        fetch = sess.run(fetches=fetch_list, feed_dict=feed_dict)

        return fetch

    def get_summary(self, sess, blobs):
        feed_dict = self._feed_dict(blobs)

        summary = sess.run(self._summary_op_val, feed_dict=feed_dict)

        return summary

    def train_step(self, sess, blobs, train_op):
        feed_dict = self._feed_dict(blobs)

        rpn_loss_cls, rpn_loss_box, loss_cls, \
            loss_box, caption_loss, loss, \
            _ = sess.run([self._losses["rpn_cross_entropy"],
                          self._losses['rpn_loss_box'],
                          self._losses['clss_cross_entropy'],
                          self._losses['loss_box'],
                          self._losses['caption_loss'],
                          self._losses['total_loss'],
                          train_op],
                         feed_dict=feed_dict)
        return rpn_loss_cls, rpn_loss_box, loss_cls, \
            loss_box, caption_loss, loss

    def train_step_with_summary(self, sess, blobs, train_op):
        feed_dict = self._feed_dict(blobs)
        rpn_loss_cls, rpn_loss_box, loss_cls, \
            loss_box, caption_loss, loss, \
            summary, _ = sess.run([self._losses["rpn_cross_entropy"],
                                   self._losses['rpn_loss_box'],
                                   self._losses['clss_cross_entropy'],
                                   self._losses['loss_box'],
                                   self._losses['caption_loss'],
                                   self._losses['total_loss'],
                                   self._summary_op,
                                   train_op],
                                  feed_dict=feed_dict)

        return rpn_loss_cls, rpn_loss_box, loss_cls, \
            loss_box, caption_loss, loss, summary

    def train_step_no_return(self, sess, blobs, train_op):
        feed_dict = self._feed_dict(blobs)

        sess.run([train_op], feed_dict=feed_dict)

    def _feed_dict(self, blobs):
        feed_dict = {self._image: blobs['data'], self._im_info: blobs['im_info'],
                     self._gt_boxes: blobs['gt_boxes'],
                     self._gt_phrases: blobs['gt_phrases']}
        if cfg.CONTEXT_FUSION:
            feed_dict.update({self._global_roi:
                np.array([[0., 0., 0., blobs['im_info'][1] - 1,
                          blobs['im_info'][0] - 1]], dtype=np.float32)})

        return feed_dict
