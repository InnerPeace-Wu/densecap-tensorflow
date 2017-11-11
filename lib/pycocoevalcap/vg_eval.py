# from __future__ import absolute_import
# from __future__ import division
# from __future__ import print_function

from tokenizer.ptbtokenizer import PTBTokenizer
from six.moves import xrange
import itertools
import numpy as np
import pprint
import time
#from bleu.bleu import Bleu
from meteor.meteor import Meteor
#from rouge.rouge import Rouge
#from cider.cider import Cider
#import nltk
import pdb


class VgEvalCap:
  def __init__(self, ref_caps, model_caps):
    self.evalImgs = []
    self.eval = {}
    self.imgToEval = {}
    self.ref = ref_caps
    self.pred = model_caps
    self.params = {'image_id': []}

  def evaluate(self):
    imgIds = self.params['image_id']
    # imgIds = self.coco.getImgIds()
    gts = {}
    res = {}
    gts_all = {}
    gts_region_idx = {}
    for imgId in imgIds:

      gts[imgId] = self.ref[imgId]
      res[imgId] = self.pred[imgId]
      gts_all[imgId] = []

      for i, anno in enumerate(gts[imgId]):
        for cap in anno['captions']:
          gts_all[imgId].append({'image_id': anno['image_id'], 'caption': cap, 'region_id': i})

    # =================================================
    # Set up scorers
    # =================================================
    print('tokenization...')
    tokenizer = PTBTokenizer()
    gts_tokens = tokenizer.tokenize(gts_all)
    res_tokens = tokenizer.tokenize(res)
    # insert caption tokens to gts
    for imgId in imgIds:
      for tokens, cap_info in zip(gts_tokens[imgId], gts_all[imgId]):
        region_id = cap_info['region_id']
        if 'caption_tokens' not in gts[imgId][region_id]:
          gts[imgId][region_id]['caption_tokens'] = []
        gts[imgId][region_id]['caption_tokens'].append(tokens)

    # =================================================
    # Compute scores
    # =================================================
    # Holistic score, as in DenseCap paper: multi-to-multi matching
    eval = {}

    print('computing Meteor score...')
    #score, scores = Meteor().compute_score_m2m(gts_tokens, res_tokens)
    #self.setEval(score, method)
    #self.setImgToEvalImgs(scores, imgIds, method)
    # print "Meteor (original): %0.3f"%(score)
    #scores_mean_im = np.zeros(len(imgIds))
    #tot_score = 0.0
    #tot_regions = 0
    # for i, scores_im in enumerate(scores):
    #  scores_mean_im[i] = np.array(scores_im).mean()
    #  tot_regions += len(scores_im)
    #  tot_score += scores_mean_im[i] * len(scores_im)
    # print "Meteor (calculated by mean over scores on all regions): %0.3f"%(tot_score / tot_regions)
    # print "Meteor (calculated by mean of scores of per image): %0.3f"%(scores_mean_im.mean())
    # self.setEvalImgs()
    # mean ap settings, as in DenseCap paper
    overlap_ratios = [0.3, 0.4, 0.5, 0.6, 0.7]
    metoer_score_th = [0, 0.05, 0.1, 0.15, 0.2, 0.25]
    ap_matrix = np.zeros((len(overlap_ratios), len(metoer_score_th)))
    gt_region_n = sum([len(gts[imgId]) for imgId in imgIds])
    # calculate the nxm bbox overlap in one pass
    #overlap_matrices = {}
    eval_stats = {}
    gts_tokens_match = {}
    res_tokens_match = {}
    all_keys = []
    t1 = time.time()
    for imgId in imgIds:
      model_caption_locations = res[imgId]
      gt_caption_locations = gts[imgId]
      # should be sorted using predicted prob in advance
      #model_caption_locations.sort(key=lambda x:-x['log_prob'])
      if len(model_caption_locations) == 0:
        continue
      ov_matrix = self.calculate_overlap_matrix(model_caption_locations, gt_caption_locations)
      match_gt_ids, match_ratios = self.bbox_match(ov_matrix)
      probs = np.array([x['prob'] for x in model_caption_locations])
      scores = np.zeros((len(res[imgId])))
      match_model_ids = np.where(match_gt_ids > -1)[0]
      match_pairs = zip(match_model_ids, match_gt_ids[match_model_ids])

      for model_id, gt_id in match_pairs:
        key = (imgId, model_id)
        all_keys.append(key)
        gts_tokens_match[key] = gts[imgId][gt_id]['caption_tokens']
        res_tokens_match[key] = [res_tokens[imgId][model_id]]
      #assert(gts_tokens_match.keys() == match_model_ids.tolist())
      #score_match, scores_match = Meteor().compute_score(gts_tokens_match, res_tokens_match)
      #scores[match_model_ids] = scores_match

      eval_stats[imgId] = {'match_ids': match_gt_ids, 'match_ratios': match_ratios, 'probs': probs, 'meteor_scores': scores}
    # compute meteor scores of matched regions in one pass
    score_match, scores_match = Meteor().compute_score(gts_tokens_match, res_tokens_match, imgIds=all_keys)
    for key, score in zip(all_keys, scores_match):
      eval_stats[key[0]]['meteor_scores'][key[1]] = score
    t2 = time.time()
    print('caption scoring finished, takes %f seconds' % (t2 - t1))

    all_match_ratios = np.concatenate([v['match_ratios'] for k, v in eval_stats.iteritems()])
    all_probs = np.concatenate([v['probs'] for k, v in eval_stats.iteritems()])
    all_scores = np.concatenate([v['meteor_scores'] for k, v in eval_stats.iteritems()])
    prob_order = np.argsort(all_probs)[::-1]
    all_match_ratios = all_match_ratios[prob_order]
    all_scores = all_scores[prob_order]

    for rid, overlap_r in enumerate(overlap_ratios):
      for th_id, score_th in enumerate(metoer_score_th):
        # compute AP for each setting
        tp = (all_match_ratios > overlap_r) & (all_scores > score_th)
        fp = 1 - tp
        tp = tp.cumsum().astype(np.float32)
        fp = fp.cumsum().astype(np.float32)
        rec = tp / gt_region_n
        prec = tp / (fp + tp)
        ap = 0
        all_t = np.linspace(0, 1, 100)
        apn = len(all_t)
        for t in all_t:
          mask = rec > t
          p = np.max(prec * mask)
          ap += p
        ap_matrix[rid, th_id] = ap / apn

    t3 = time.time()
    print('mean ap computing finished, takes %f seconds' % (t3 - t2))
    mean_ap = np.mean(ap_matrix) * 100  # percent
    print('mean match ratio is %0.3f' % all_match_ratios.mean())

    print('ap matrix')
    print(ap_matrix)
    print("mean average precision is %0.3f" % mean_ap)
    return(mean_ap)

  def calculate_overlap_matrix(self, model_caption_locations, gt_caption_locations):
    model_region_n = len(model_caption_locations)
    gt_region_n = len(gt_caption_locations)
    #overlap_matrix = np.zeros((model_region_n, gt_region_n))
    model_bboxes = np.array([x['location'] for x in model_caption_locations])  # nx4 matrix
    gt_bboxes = np.array([x['location'] for x in gt_caption_locations])
    # area, intersection area, union area
    model_bbox_areas = (model_bboxes[:, 2] - model_bboxes[:, 0]) * \
        (model_bboxes[:, 3] - model_bboxes[:, 1])
    gt_bbox_areas = (gt_bboxes[:, 2] - gt_bboxes[:, 0]) * \
        (gt_bboxes[:, 3] - gt_bboxes[:, 1])
    x_a1 = model_bboxes[:, 0].reshape(model_region_n, 1)
    x_a2 = model_bboxes[:, 2].reshape(model_region_n, 1)
    x_b1 = gt_bboxes[:, 0].reshape(1, gt_region_n)
    x_b2 = gt_bboxes[:, 2].reshape(1, gt_region_n)
    y_a1 = model_bboxes[:, 1].reshape(model_region_n, 1)
    y_a2 = model_bboxes[:, 3].reshape(model_region_n, 1)
    y_b1 = gt_bboxes[:, 1].reshape(1, gt_region_n)
    y_b2 = gt_bboxes[:, 3].reshape(1, gt_region_n)
    bbox_pair_x_diff = np.maximum(0, np.minimum(x_a2, x_b2) - np.maximum(x_a1, x_b1))
    bbox_pair_y_diff = np.maximum(0, np.minimum(y_a2, y_b2) - np.maximum(y_a1, y_b1))
    inter_areas = bbox_pair_x_diff * bbox_pair_y_diff
    # IoU
    union_areas = model_bbox_areas.reshape(model_region_n, 1) + gt_bbox_areas.reshape(1, gt_region_n)
    overlap_matrix = inter_areas / (union_areas - inter_areas)
    return overlap_matrix

  def bbox_match(self, overlap_matrix):
    # greedy matching of candiate bboxes to gt bboxes
    #assert(1 > overlap >= 0)

    model_n = overlap_matrix.shape[0]
    gt_n = overlap_matrix.shape[1]

    gt_flag = np.ones((gt_n), dtype=np.int32)
    match_ids = -1 * np.ones((model_n), dtype=np.int32)
    match_ratios = np.zeros((model_n))
    # modified bbox matching scheme, the same with Justin's code
    for i in xrange(model_n):
      overlap_step = overlap_matrix[i, :]
      max_overlap_id = np.argmax(overlap_step)
      if gt_flag[max_overlap_id] == 1 and overlap_step[max_overlap_id] > 0:
        gt_flag[max_overlap_id] = 0
        match_ratios[i] = overlap_step[max_overlap_id]
        match_ids[i] = max_overlap_id
      else:
        pass
    # for i in xrange(model_n):
    #   overlap_step = overlap_matrix[i,:] * gt_flag
    #   max_overlap_id = np.argmax(overlap_step)
    #   if overlap_step[max_overlap_id] > 0:
    #     gt_flag[max_overlap_id] = 0
    #     match_ratios[i] = overlap_step[max_overlap_id]
    #     match_ids[i] = max_overlap_id
    #   else:
    #     pass

    return match_ids, match_ratios

  def setEval(self, score, method):
    self.eval[method] = score

  def setImgToEvalImgs(self, scores, imgIds, method):
    for imgId, score in zip(imgIds, scores):
      if not imgId in self.imgToEval:
        self.imgToEval[imgId] = {}
        self.imgToEval[imgId]["image_id"] = imgId
      self.imgToEval[imgId][method] = score

  def setEvalImgs(self):
    self.evalImgs = [eval for imgId, eval in self.imgToEval.items()]
