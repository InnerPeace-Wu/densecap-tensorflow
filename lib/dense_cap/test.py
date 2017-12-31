# --------------------------------------------------------
# Evaluate a dense captioning model using mean AP
# Code adapted from faster R-CNN project
# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""Test a Fast R-CNN network on an imdb (image database)."""

from lib.config import cfg, get_output_dir
from lib.fast_rcnn.bbox_transform import clip_boxes, bbox_transform_inv
import argparse
from lib.utils.timer import Timer
from six.moves import xrange
import numpy as np
import math
import cv2
from lib.fast_rcnn.nms_wrapper import nms
import json
from lib.utils.blob import im_list_to_blob
import os
import sys
# sys.path.append('lib/')
from lib.utils.bbox_utils import region_merge, get_bbox_coord
from lib.pycocoevalcap.vg_eval import VgEvalCap
from lib.dense_cap.beam_search import beam_search
from lib.dense_cap.vis_whtml import vis_whtml

COCO_EVAL_PATH = 'coco-caption/'
sys.path.append(COCO_EVAL_PATH)

eps = 1e-10
DEBUG = False


def _get_image_blob(im):
    """Converts an image into a network input.

    Arguments:
        im (ndarray): a color image in BGR order

    Returns:
        blob (ndarray): a data blob holding an image pyramid
        im_scale_factors (list): list of image scales (relative to im) used
            in the image pyramid
    """
    im_orig = im.astype(np.float32, copy=True)
    im_orig -= cfg.PIXEL_MEANS

    im_shape = im_orig.shape
    im_size_min = np.min(im_shape[0:2])
    im_size_max = np.max(im_shape[0:2])

    processed_ims = []
    im_scale_factors = []

    for target_size in cfg.TEST.SCALES:
        im_scale = float(target_size) / float(im_size_min)
        # Prevent the biggest axis from being more than MAX_SIZE
        if np.round(im_scale * im_size_max) > cfg.TEST.MAX_SIZE:
            im_scale = float(cfg.TEST.MAX_SIZE) / float(im_size_max)
        im = cv2.resize(im_orig, None, None, fx=im_scale, fy=im_scale,
                        interpolation=cv2.INTER_LINEAR)
        im_scale_factors.append(im_scale)
        processed_ims.append(im)

    # Create a blob to hold the input images
    blob = im_list_to_blob(processed_ims)

    return blob, np.array(im_scale_factors)


# TODO: NO_RPN option, not used
def _get_rois_blob(im_rois, im_scale_factors):
    """Converts RoIs into network inputs.

    Arguments:
        im_rois (ndarray): R x 4 matrix of RoIs in original image coordinates
        im_scale_factors (list): scale factors as returned by _get_image_blob

    Returns:
        blob (ndarray): R x 5 matrix of RoIs in the image pyramid
    """
    rois, levels = _project_im_rois(im_rois, im_scale_factors)
    rois_blob = np.hstack((levels, rois))
    return rois_blob.astype(np.float32, copy=False)


# TODO: NO_RPN option, not used
def _project_im_rois(im_rois, scales):
    """Project image RoIs into the image pyramid built by _get_image_blob.

    Arguments:
        im_rois (ndarray): R x 4 matrix of RoIs in original image coordinates
        scales (list): scale factors as returned by _get_image_blob

    Returns:
        rois (ndarray): R x 4 matrix of projected RoI coordinates
        levels (list): image pyramid levels used by each projected RoI
    """
    im_rois = im_rois.astype(np.float, copy=False)

    if len(scales) > 1:
        widths = im_rois[:, 2] - im_rois[:, 0] + 1
        heights = im_rois[:, 3] - im_rois[:, 1] + 1

        areas = widths * heights
        scaled_areas = areas[:, np.newaxis] * (scales[np.newaxis, :] ** 2)
        diff_areas = np.abs(scaled_areas - 224 * 224)
        levels = diff_areas.argmin(axis=1)[:, np.newaxis]
    else:
        levels = np.zeros((im_rois.shape[0], 1), dtype=np.int)

    rois = im_rois * scales[levels]

    return rois, levels


def _get_blobs(im, rois):
    """Convert an image and RoIs within that image into network inputs."""
    blobs = {'data': None, 'rois': None}
    blobs['data'], im_scale_factors = _get_image_blob(im)

    # TODO: NO_RPN option, not used
    if not cfg.TEST.HAS_RPN:
        blobs['rois'] = _get_rois_blob(rois, im_scale_factors)
    return blobs, im_scale_factors

# TODO: caffe version, not used


def _greedy_search(embed_net, recurrent_net, forward_args, optional_args, proposal_n, max_timestep=15,
                   pred_bbox=True, use_box_at=-1):
    """Do greedy search to find the regions and captions"""
    # Data preparation

    pred_captions = [None] * proposal_n
    pred_logprobs = [0.0] * proposal_n
    pred_bbox_offsets = np.zeros((proposal_n, 4))

    forward_args['cont_sentence'] = np.zeros((1, proposal_n))

    if 'global_features' in optional_args:
        forward_args['global_features'] = optional_args['global_features'].reshape(
            *(forward_args['input_features'].shape))
    # reshape blobs
    for k, v in forward_args.iteritems():
        if DEBUG:
            print 'shape of %s is ' % k
            print v.shape
        recurrent_net.blobs[k].reshape(*(v.shape))

    recurrent_net.forward(**forward_args)
    forward_args['cont_sentence'][:] = 1

    input_sentence = np.zeros((1, proposal_n))  # start with EOS
    embed_net.blobs['input_sentence'].reshape(1, proposal_n)

    for step in xrange(max_timestep):

        embed_out = embed_net.forward(input_sentence=input_sentence)
        forward_args['input_features'] = embed_out['embedded_sentence']
        # another lstm for global features
        if 'global_features' in recurrent_net.blobs:
            forward_args['global_features'] = embed_out['embedded_sentence']
        if 'fusion_features' in recurrent_net.blobs:
            forward_args['fusion_features'] = embed_out['embedded_sentence']
        blobs_out = recurrent_net.forward(**forward_args)

        word_probs = blobs_out['probs'].copy()
        bbox_pred = blobs_out['bbox_pred'] if pred_bbox else None
        # suppress <unk> tag
        # word_probs[:,:,1] = 0
        best_words = word_probs.argmax(axis=2).reshape(proposal_n)
        finish_n = 0
        for i, w in zip(range(proposal_n), best_words):
            if not pred_captions[i]:
                pred_captions[i] = [w]
                pred_logprobs[i] = math.log(word_probs[0, i, w])
                if pred_bbox and use_box_at == 0:
                    pred_bbox_offsets[i, :] = bbox_pred[0, i, :].copy()
            elif pred_captions[i][-1] != 0:
                pred_captions[i].append(w)
                pred_logprobs[i] += math.log(word_probs[0, i, w])
                if pred_bbox and (use_box_at < 0 or use_box_at >= step):
                    pred_bbox_offsets[i, :] = bbox_pred[0, i, :].copy()

            else:
                finish_n += 1

        input_sentence[:] = best_words
        forward_args['cont_sentence'][:] = 1
        if finish_n == proposal_n:
            break
    return pred_captions, pred_bbox_offsets, pred_logprobs


def im_detect(sess, net, im, boxes=None, use_box_at=-1):
    """Detect object classes in an image given object proposals.

    Arguments:
        im (ndarray): color image to test (in BGR order)
        boxes (ndarray): R x 4 array of object proposals or None (for RPN)
        use_box_at (int32): Use predicted box at a given timestep, default to the last one (use_box_at=-1)
    Returns:
        scores (ndarray): R x 1 array of object class scores
        pred_boxes (ndarray)): R x 4 array of predicted bounding boxes
        captions (list): length R list of list of word tokens (captions)
    """

    # for bbox unnormalization
    bbox_mean = np.array(cfg.TRAIN.BBOX_NORMALIZE_MEANS).reshape((1, 4))
    bbox_stds = np.array(cfg.TRAIN.BBOX_NORMALIZE_STDS).reshape((1, 4))

    blobs, im_scales = _get_blobs(im, boxes)
    assert len(im_scales) == 1, "Only single-image batch implemented"
    im_blob = blobs['data']
    blobs['im_info'] = np.array(
        [[im_blob.shape[1], im_blob.shape[2], im_scales[0]]],
        dtype=np.float32)

    if cfg.TEST.USE_BEAM_SEARCH:
        scores, box_offsets, captions, boxes = beam_search(sess, net, blobs, im_scales)
    else:
        scores, box_offsets, captions, boxes = greedy_search(sess, net, blobs, im_scales)

    # bbox target unnormalization
    box_deltas = box_offsets * bbox_stds + bbox_mean

    # do the transformation
    pred_boxes = bbox_transform_inv(boxes, box_deltas)
    pred_boxes = clip_boxes(pred_boxes, im.shape)

    return scores[:, 1], pred_boxes, captions


def greedy_search(sess, net, blobs, im_scales):
    # (TODO wu) for now it only works with "concat" mode

    # get initial states and rois
    if cfg.CONTEXT_FUSION:
        cap_state, loc_state, scores, \
            rois, gfeat_state = net.feed_image(sess,
                                               blobs['data'],
                                               blobs['im_info'][0])
    else:
        cap_state, loc_state, scores, rois = net.feed_image(sess, blobs['data'],
                                                            blobs['im_info'][0])

    # proposal boxes
    boxes = rois[:, 1:5] / im_scales[0]
    proposal_n = rois.shape[0]

    cap_probs = np.ones((proposal_n, 1), dtype=np.int32)
    # index of <EOS> in vocab
    end_idx = cfg.VOCAB_END_ID
    # captions = np.empty([proposal_n, 1], dtype=np.int32)
    bbox_offsets_list = []
    box_offsets = np.zeros((proposal_n, 4), dtype=np.float32)
    bbox_pred = np.zeros((proposal_n, 4), dtype=np.float32)
    for i in xrange(cfg.TIME_STEPS - 1):
        # dim: [proposal_n, ]
        input_feed = np.argmax(cap_probs, axis=1)
        if i == 0:
            captions = input_feed[:, None]
        else:
            captions = np.concatenate((captions, input_feed[:, None]), axis=1)
        # dim: [proposal_n, i+1]
        end_ids = np.where(input_feed == end_idx)[0]
        # prepare for seq length in dynamic rnn
        input_feed[end_ids] = 0
        box_offsets[end_ids] = bbox_pred[end_ids]
        if cfg.CONTEXT_FUSION:
            cap_probs, bbox_pred, cap_state, loc_state, \
                gfeat_state = net.inference_step(sess, input_feed,
                                                 cap_state, loc_state, gfeat_state)
        else:
            cap_probs, bbox_pred, cap_state, \
                loc_state = net.inference_step(sess, input_feed,
                                               cap_state, loc_state)
        bbox_offsets_list.append(bbox_pred)

    return scores, box_offsets, captions, boxes


def vis_detections(im_path, im, captions, dets, thresh=0.5, save_path='vis'):
    """Visual debugging of detections by saving images with detected bboxes."""
    # add html generation for better visualization
    print('visualizing ...')
    if not os.path.exists(save_path + '/images'):
        os.makedirs(save_path + '/images')
    im_name = im_path.split('/')[-1][:-4]
    page = open(os.path.join(save_path, im_name + '.html'), 'w')
    page.write('<hr><h2>Dense caption results for image %s</h2>' % im_name)
    for i in xrange(dets.shape[0]):
        bbox = dets[i, :4]
        score = dets[i, -1]
        caption = captions[i]
        if score > thresh:
            im_new = np.copy(im)

            cv2.rectangle(im_new, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 0, 255), 2)
            im_rel_path = 'images/%s_%d.jpg' % (im_name, i)
            cv2.imwrite('%s/%s' % (save_path, im_rel_path), im_new)
            page.write('<div style=\'border: 2px solid; width:166px; height:360px; display:inline-table\'>')
            page.write('<image width="260" height = "260" src=\'%s\'></image><br> <hr><label> %s </label></div>' % (
                im_rel_path, caption))
    page.write('<hr>')
    page.close()


def sentence(vocab, vocab_indices):
    # consider <eos> tag with id 0 in vocabulary
    for ei, idx in enumerate(vocab_indices):
        # End of sentence
        if idx == 2:
            break
    sentence = ' '.join([vocab[i] for i in vocab_indices[1:ei]])
    # suffix = ' ' + vocab[2]
    # if sentence.endswith(suffix):
    #     sentence = sentence[1:-len(suffix)]
    return sentence


def test_im(sess, net, im_path, vocab, pre_results, vis=True):
    im = cv2.imread(im_path)
    scores, boxes, captions = im_detect(sess, net, im, None, use_box_at=-1)
    pos_dets = np.hstack((boxes, scores[:, np.newaxis])) \
        .astype(np.float32, copy=False)
    keep = nms(pos_dets, cfg.TEST.NMS)
    pos_dets = pos_dets[keep, :]
    pos_scores = scores[keep]
    pos_captions = [sentence(vocab, captions[idx]) for idx in keep]
    pos_boxes = boxes[keep, :]
    if vis:
        vis_detections(im_path, im, pos_captions, pos_dets, save_path='./demo')
        results = vis_whtml(im_path, im, pos_captions, pos_dets, pre_results)

    return results


def test_net(sess, net, imdb, vis=True, use_box_at=-1):
    """Test a Fast R-CNN network on an image database."""
    num_images = len(imdb.image_index)
    if DEBUG:
        print 'number of images: %d' % num_images
    # all detections are collected into:
    #    all_regions[image] = list of {'image_id', caption', 'location', 'location_seq'}
    all_regions = [None] * num_images
    results = {}
    output_dir = get_output_dir(imdb, None)

    # timers
    _t = {'im_detect': Timer(), 'misc': Timer()}

    # if not cfg.TEST.HAS_RPN:
    # roidb = imdb.roidb

    # read vocabulary  & add <eos> tag
    vocab = list(imdb.get_vocabulary())
    vocab_extra = ['<EOS>', '<SOS>', '<PAD>']
    for ex in vocab_extra:
        vocab.insert(0, ex)

    for i in xrange(num_images):
        # filter out any ground truth boxes
        if cfg.TEST.HAS_RPN:
            box_proposals = None
        # else:
            # The roidb may contain ground-truth rois (for example, if the roidb
            # comes from the training or val split). We only want to evaluate
            # detection on the *non*-ground-truth rois. We select those the rois
            # that have the gt_classes field set to 0, which means there's no
            # ground truth.
            # box_proposals = roidb[i]['boxes'][roidb[i]['gt_classes'] == 0]

        im = cv2.imread(imdb.image_path_at(i))
        _t['im_detect'].tic()
        scores, boxes, captions = im_detect(sess, net,
                                            im, box_proposals,
                                            use_box_at=use_box_at)
        _t['im_detect'].toc()

        _t['misc'].tic()
        # only one positive class
        if DEBUG:
            print('shape of scores')
            print(scores.shape)

        pos_dets = np.hstack((boxes, scores[:, np.newaxis])) \
            .astype(np.float32, copy=False)
        keep = nms(pos_dets, cfg.TEST.NMS)
        pos_dets = pos_dets[keep, :]
        pos_scores = scores[keep]
        pos_captions = [sentence(vocab, captions[idx]) for idx in keep]
        pos_boxes = boxes[keep, :]
        if vis:
            vis_detections(imdb.image_path_at(i), im, pos_captions, pos_dets, save_path=os.path.join(output_dir, 'vis'))
        all_regions[i] = []
        # follow the format of baseline models routine
        for cap, box, prob in zip(pos_captions, pos_boxes, pos_scores):
            anno = {'image_id': i, 'prob': format(prob, '.3f'), 'caption': cap,
                    'location': box.tolist()}
            all_regions[i].append(anno)
        key = imdb.image_path_at(i).split('/')[-1]
        results[key] = {}
        results[key]['boxes'] = pos_boxes.tolist()
        results[key]['logprobs'] = np.log(pos_scores + eps).tolist()
        results[key]['captions'] = pos_captions

        _t['misc'].toc()

        print('im_detect: {:d}/{:d} {:.3f}s {:.3f}s'
              .format(i + 1, num_images, _t['im_detect'].average_time,
                      _t['misc'].average_time))
    # write file for evaluation with Torch code from Justin
    print('write to result.json')
    det_file = os.path.join(output_dir, 'results.json')
    with open(det_file, 'w') as f:
        json.dump(results, f)

    print('Evaluating detections')

    # gt_regions = imdb.get_gt_regions() # is a list
    gt_regions_merged = [None] * num_images
    # transform gt_regions into the baseline model routine
    for i, image_index in enumerate(imdb.image_index):
        new_gt_regions = []
        regions = imdb.get_gt_regions_index(image_index)
        for reg in regions['regions']:
            loc = np.array([reg['x'], reg['y'], reg['x'] + reg['width'], reg['y'] + reg['height']])
            anno = {'image_id': i, 'caption': reg['phrase'].encode('ascii', 'ignore'), 'location': loc}
            new_gt_regions.append(anno)
        # merge regions with large overlapped areas
        assert (len(new_gt_regions) > 0)
        gt_regions_merged[i] = region_merge(new_gt_regions)
    image_ids = range(num_images)
    vg_evaluator = VgEvalCap(gt_regions_merged, all_regions)
    vg_evaluator.params['image_id'] = image_ids
    vg_evaluator.evaluate()
