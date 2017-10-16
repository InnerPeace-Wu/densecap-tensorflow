#!/usr/bin/env python

from collections import OrderedDict
import json
import numpy as np
import pprint
import cPickle as pickle
import string

def get_bbox_coord(norm_coord, do_clip=True):
  #input is a nx4 numpy array in normalized bbox coordinates
  #print norm_coord.shape
  #print norm_coord
  bboxes_coord = np.zeros(norm_coord.shape)
  #x,y,w,h
  bboxes_coord[:, :2] = norm_coord[:, :2]+0.5
  bboxes_coord[:, 2:] = np.exp(norm_coord[:, 2:])
  
  #x1,y1,x2,y2
  bboxes_coord2 = np.zeros(norm_coord.shape)
  bboxes_coord2[:, :2] = bboxes_coord[:, :2] - bboxes_coord[:, 2:] * 0.5
  bboxes_coord2[:, 2:] = bboxes_coord[:, :2] + bboxes_coord[:, 2:] * 0.5
  #clipping all coordinates to [0,1]
  if do_clip:
    bboxes_coord2 = np.minimum(np.maximum(bboxes_coord2, 0), 1)
  return bboxes_coord2


def get_bbox_iou_matrix(bboxes):
  region_n = bboxes.shape[0]
  #area, intersection area, union area
  bbox_areas = (bboxes[:,2] - bboxes[:,0]) * \
    (bboxes[:, 3] - bboxes[:, 1])
  
  x_a1 = bboxes[:,0].reshape(region_n,1)
  x_a2 = bboxes[:,2].reshape(region_n,1)
  x_b1 = bboxes[:,0].reshape(1,region_n)
  x_b2 = bboxes[:,2].reshape(1,region_n)
  y_a1 = bboxes[:,1].reshape(region_n,1)
  y_a2 = bboxes[:,3].reshape(region_n,1)
  y_b1 = bboxes[:,1].reshape(1,region_n)
  y_b2 = bboxes[:,3].reshape(1,region_n)
  bbox_pair_x_diff = np.maximum(0, np.minimum(x_a2, x_b2) - np.maximum(x_a1, x_b1))
  bbox_pair_y_diff = np.maximum(0, np.minimum(y_a2, y_b2) - np.maximum(y_a1, y_b1))
  inter_areas = bbox_pair_x_diff * bbox_pair_y_diff
  
  #IoU
  union_areas = bbox_areas.reshape(region_n,1) + bbox_areas.reshape(1,region_n)
 
  bbox_iou = inter_areas / (union_areas - inter_areas)
  return bbox_iou
  
def nms(region_info, bbox_th=0.3):
  #non-maximum surpression
  region_info.sort(key = lambda x: -x['log_prob'])
  #keep_index = []
  region_n = len(region_info)
  #fast computation of pairwise IoU
  #pick the bbox of last timestep of each sample
  #print 'region_info length %d' % len(region_info)
  all_bboxes = np.array([x['location'][-1,:] for x in region_info])# nx4 matrix
  bbox_iou = get_bbox_iou_matrix(all_bboxes)
  bbox_iou_th = bbox_iou < bbox_th
  keep_flag = np.ones((region_n),dtype=np.uint8)

  for i in xrange(region_n-1):
    if keep_flag[i]:
      keep_flag[i+1:] = np.logical_and(keep_flag[i+1:], bbox_iou_th[i,i+1:])  
  print 'sum of keep flag'
  print keep_flag.sum()
  return [region_info[i] for i in xrange(region_n) if keep_flag[i]] 

def region_merge(region_info, bbox_th=0.7):
  #merging ground truth bboxes

  #keep_index = []
  region_n = len(region_info)
  region_merged = []
  #fast computation of pairwise IoU
  #pick the bbox of last timestep of each sample
  all_bboxes = np.array([x['location'] for x in region_info], dtype = np.float32)# nx4 matrix
  bbox_iou = get_bbox_iou_matrix(all_bboxes)
  bbox_iou_th = bbox_iou > bbox_th
  bbox_iou_overlap_n = bbox_iou_th.sum(axis = 0)

  merge_flag = np.ones((region_n),dtype=np.uint8)
  unmerged_region = region_n
  while unmerged_region > 0:
    max_overlap_id = np.argmax(bbox_iou_overlap_n)
    assert bbox_iou_overlap_n[max_overlap_id] > 0
    merge_group = np.nonzero(bbox_iou_th[max_overlap_id,:] & merge_flag)[0]
    unmerged_region -= len(merge_group)
    merge_flag[merge_group] = 0
    bbox_iou_overlap_n[merge_group] = 0
    bbox_group = all_bboxes[merge_group,:].reshape(len(merge_group),4)
    caption_group = [region_info[i]['caption'] for i in merge_group]
    bbox_mean = np.mean(bbox_group, axis = 0).tolist()
    region_merged.append({'image_id':region_info[max_overlap_id]['image_id'], \
      'captions': caption_group, 'location': bbox_mean})
  return region_merged    

