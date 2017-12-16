# --------------------------------------------------------
# DenseCap-Tensorflow
# Written by InnerPeace
# This file is adapted from Linjie's work
#             and Google's im2txt project
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pdb
import math
from lib.dense_cap.caption_generator import *
import numpy as np
from lib.config import cfg
import tensorflow as tf
from six.moves import xrange


def beam_search(sess, net, blobs, im_scales):
    # (TODO wu) for now it only works with "concat" mode
    # get initial states and rois
    if cfg.CONTEXT_FUSION:
        cap_state, loc_state, scores, \
            rois, gfeat_state = net.feed_image(sess,
                                               blobs['data'],
                                               blobs['im_info'][0])
        all_states = np.concatenate((cap_state, loc_state, gfeat_state), axis=1)
    else:
        cap_state, loc_state, scores, rois = net.feed_image(sess, blobs['data'],
                                                            blobs['im_info'][0])
        all_states = np.concatenate((cap_state, loc_state), axis=1)

    # proposal boxes
    boxes = rois[:, 1:5] / im_scales[0]
    proposal_n = rois.shape[0]

    all_partial_caps = []
    all_complete_caps = []
    beam_size = cfg.TEST.BEAM_SIZE
    for i in xrange(proposal_n):
        init_beam = Caption(sentence=[cfg.VOCAB_START_ID],
                            state=all_states[i],
                            box_pred=[],
                            logprob=0.0,
                            score=0.0,
                            metadata=[""])
        partial_cap = TopN(beam_size)
        partial_cap.push(init_beam)
        complete_cap = TopN(beam_size)
        all_partial_caps.append(partial_cap)
        all_complete_caps.append(complete_cap)

    for j in xrange(cfg.TIME_STEPS - 1):
        all_candidates_len = []
        flag = False
        for i in xrange(proposal_n):
            partial_cap = all_partial_caps[i]
            size = partial_cap.size()
            all_candidates_len.append(size)
            if not size:
                continue
            partial_cap_list = partial_cap.get_data()
            input_feed_i = [c.sentence[-1] for c in partial_cap_list]
            state_feed_i = [c.state for c in partial_cap_list]
            if not flag:
                flag = True
                input_feed = np.array(input_feed_i)
                state_feed = np.array(state_feed_i)
            else:
                input_feed = np.concatenate((input_feed, np.array(input_feed_i)))
                state_feed = np.concatenate((state_feed, np.array(state_feed_i)))

        if cfg.CONTEXT_FUSION:
            cap_feed, loc_feed, gfeat_feed = np.split(state_feed, 3, axis=1)
            cap_probs, new_bbox_pred, new_cap_state, new_loc_state, \
                new_gfeat_state = net.inference_step(sess, input_feed,
                                                     cap_feed, loc_feed, gfeat_feed)
            new_state = np.concatenate((new_cap_state, new_loc_state, new_gfeat_state),
                                       axis=1)
        else:
            cap_feed, loc_feed = np.split(state_feed, 2, axis=1)
            cap_probs, new_bbox_pred, new_cap_state, \
                new_loc_state = net.inference_step(sess, input_feed,
                                                   cap_feed, loc_feed)
            new_state = np.concatenate((new_cap_state, new_loc_state), axis=1)

        count = 0
        for k in xrange(proposal_n):
            l = all_candidates_len[k]
            if l == 0:
                continue
            partial_cap = all_partial_caps[k]
            complete_cap = all_complete_caps[k]
            partial_cap_list = partial_cap.extract()
            partial_cap.reset()
            softmax_k = cap_probs[count: count + l]
            states_k = new_state[count: count + l]
            bbox_pred_k = new_bbox_pred[count: count + l]
            count += l
            for i, par_cap in enumerate(partial_cap_list):
                word_probs = softmax_k[i]
                state = states_k[i]
                bbox_pred = bbox_pred_k[i]
                # For this partial caption, get the beam_size most probable next words.
                words_and_probs = list(enumerate(word_probs))
                words_and_probs.sort(key=lambda x: -x[1])
                words_and_probs = words_and_probs[0: beam_size]
                # Each next word gives a new partial caption
                for w, p in words_and_probs:
                    if p < 1e-12:
                        continue  # Avoid log(0)
                    sentence = par_cap.sentence + [w]
                    logprob = par_cap.logprob + math.log(p)
                    sc = logprob
                    box_pred = par_cap.box_pred
                    box_pred.append(bbox_pred)
                    if w == cfg.VOCAB_END_ID:
                        if cfg.TEST.LN_FACTOR > 0:
                            sc /= len(sentence) ** cfg.TEST.LN_FACTOR
                        beam = Caption(sentence, state, box_pred, logprob, sc)
                        complete_cap.push(beam)
                    else:
                        beam = Caption(sentence, state, box_pred, logprob, sc)
                        partial_cap.push(beam)
    captions = []
    box_offsets = np.zeros((proposal_n, 4), dtype=np.float32)
    for i in xrange(proposal_n):
        complete_cap = all_complete_caps[i]
        if not complete_cap.size():
            complete_cap = all_partial_caps[i]
        caps_i = complete_cap.extract(sort=True)
        captions.append(caps_i[0].sentence)
        box_offsets[i] = caps_i[0].box_pred[-1]

    return scores, box_offsets, captions, boxes
