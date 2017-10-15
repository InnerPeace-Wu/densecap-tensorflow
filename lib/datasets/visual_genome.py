# --------------------------------------------------------
# DenseCap-Tensorflow
# Written by InnerPeace
# This file is adapted from Linjie's work
# --------------------------------------------------------

import os
import numpy as np
import scipy.sparse
import cPickle
import uuid
import json
import logging
from os.path import join as pjoin

from datasets.imdb import imdb
from lib.config import cfg

# import xml.etree.ElementTree as ET
# import datasets.ds_utils as ds_utils
# import scipy.io as sio
# import utils.cython_bbox
# import subprocess

DEBUG = False
USE_CACHE = True
UNK_IDENTIFIER = '<unk>'
DEFAULT_PATH = 'data/visual_genome'


class visual_genome(imdb):
    def __init__(self, image_set, version):
        imdb.__init__(self, 'vg_' + version + '_' + image_set)
        # image_set from ['train', 'val', 'test']
        self._image_set = image_set

        self._data_path = '%s/%s' % (DEFAULT_PATH, version)
        cfg.CACHE_DIR = self._data_path
        self._image_ext = '.jpg'
        logging.info('data_path: %s' % self._data_path)
        self.region_imset_path = os.path.join(self._data_path,
                                         '%s_gt_regions' % image_set)
        self._classes = ('__background__', '__foreground__')

        # TODO: change regoin loads procedure
        self._gt_regions = json.load(open(self.region_imset_path))

        self._image_index = self._load_image_set_index()
        # Default to roidb handler
        self._roidb_handler = self.rpn_roidb
        self._salt = str(uuid.uuid4())
        vocab_path = os.path.join(self._data_path, 'vocabulary.txt')
        with open(vocab_path, 'r') as f:
            self._vocabulary_inverted = [line.strip() for line in f]

        self._vocabulary = dict([(w, i) for i, w in enumerate(self._vocabulary_inverted)])

        assert os.path.exists(self._data_path), \
            'Path does not exist: {}'.format(self._data_path)

    def image_path_at(self, i):
        """
        Return the absolute path to image i in the image sequence.
        """
        return self.image_path_from_index(self._image_index[i])

    def image_path_from_index(self, index):
        """
        Construct an image path from the image's "index" identifier.
        """
        # image_path = self._gt_regions[index]['path']
        # load region from a json file
        with open(pjoin(self.region_imset_path, '%s.json' % index), 'r') as f:
            image_path = json.load(f)['path']
        assert os.path.exists(image_path), \
            'Path does not exist: {}'.format(image_path)
        return image_path

    def _load_image_set_index(self, ext='json'):
        """
        Load the indexes listed in this dataset's image set file.
        """
        if ext == 'json':
            path = pjoin(cfg.SPLIT_DIR, 'densecap_splits.json')
            with open(path, 'r') as f:
                image_index = json.load(f)[self._image_set]
        elif ext == 'txt':
            path = pjoin(cfg.SPLIT_DIR, '%s.txt' % self._image_set)
            with open(path, 'r') as f:
                image_index = [line.strip() for line in f.readlines()]

        # image_index = [key for key in self._gt_regions]

        return image_index

    def get_gt_regions(self):
        return [v for k, v in self._gt_regions.iteritems()]

    def get_gt_regions_index(self, index):
        return self._gt_regions[index]

    def get_vocabulary(self):
        return self._vocabulary_inverted

    def gt_roidb(self):
        """
        Return the database of ground-truth regions of interest.

        This function loads/saves from/to a cache file to speed up future calls.
        """
        cache_file = pjoin(self._data_path, self._image_set + '_gt_roidb.pkl')
        cache_file_phrases = pjoin(self._data_path, self._image_set + '_gt_phrases.pkl')
        if os.path.exists(cache_file) and USE_CACHE:
            with open(cache_file, 'rb') as fid:
                roidb = cPickle.load(fid)
            logging.info('{} gt roidb loaded from {}'.format(self._image_set, cache_file))
            return roidb

        gt_roidb = [self._load_vg_annotation(index) for index in self._image_index]
        gt_phrases = {}
        for k, v in self._gt_regions.iteritems():
            for reg in v['regions']:
                gt_phrases[reg['id']] = self._line_to_stream(reg['phrase_tokens'])

                if DEBUG:
                    # CHECK consistency
                    for wi, w in zip(gt_phrases[reg['id']], reg['phrase_tokens']):
                        vocab_w = self._vocabulary_inverted[wi - 1]
                        print vocab_w, w
                        assert (vocab_w == UNK_IDENTIFIER or vocab_w == w)
        with open(cache_file, 'wb') as fid:
            cPickle.dump(gt_roidb, fid, cPickle.HIGHEST_PROTOCOL)
        with open(cache_file_phrases, 'wb') as fid:
            cPickle.dump(gt_phrases, fid, cPickle.HIGHEST_PROTOCOL)
        print 'wrote gt roidb to {}'.format(cache_file)
        print 'wrote gt phrases to {}'.format(cache_file_phrases)
        return gt_roidb

    def rpn_roidb(self):
        if self._image_set != 'test':
            gt_roidb = self.gt_roidb()
            rpn_roidb = self._load_rpn_roidb(gt_roidb)
            roidb = imdb.merge_roidbs(gt_roidb, rpn_roidb)
        else:
            roidb = self._load_rpn_roidb(None)

        return roidb

    def _load_rpn_roidb(self, gt_roidb):
        filename = self.config['rpn_file']
        print 'loading {}'.format(filename)
        assert os.path.exists(filename), \
            'rpn data not found at: {}'.format(filename)
        with open(filename, 'rb') as f:
            box_list = cPickle.load(f)
        return self.create_roidb_from_box_list(box_list, gt_roidb)

    def _line_to_stream(self, sentence):
        stream = []
        for word in sentence:
            word = word.strip()
            if word in self._vocabulary:
                stream.append(self._vocabulary[word])
            else:  # unknown word; append UNK
                stream.append(self._vocabulary[UNK_IDENTIFIER])
        # increment the stream -- 0 will be the EOS character
        stream = [s + 1 for s in stream]
        return stream

    def _load_vg_annotation(self, index):
        """
        Load image and bounding boxes info from XML file in the PASCAL VOC
        format.
        """

        regions = self._gt_regions[index]['regions']
        num_regs = len(regions)
        boxes = np.zeros((num_regs, 4), dtype=np.uint16)
        gt_classes = np.zeros((num_regs), dtype=np.int32)
        overlaps = np.zeros((num_regs, self.num_classes), dtype=np.float32)
        # "Seg" area for pascal is just the box area
        seg_areas = np.zeros((num_regs), dtype=np.float32)

        # Load object bounding boxes into a data frame.
        for ix, reg in enumerate(regions):
            # Make pixel indexes 0-based
            x1 = reg['x']
            y1 = reg['y']
            x2 = reg['x'] + reg['width']
            y2 = reg['y'] + reg['height']

            boxes[ix, :] = [x1, y1, x2, y2]
            gt_classes[ix] = reg['id']  # replace the class id with region id so that can retrieve the caption later
            overlaps[ix, 1] = 1.0
            seg_areas[ix] = (x2 - x1 + 1) * (y2 - y1 + 1)

        overlaps = scipy.sparse.csr_matrix(overlaps)
        return {'boxes': boxes,
                'gt_classes': gt_classes,
                'gt_overlaps': overlaps,
                'flipped': False,
                'seg_areas': seg_areas}


if __name__ == '__main__':
    d = visual_genome('train', '1.0')
    res = d.roidb
    from IPython import embed;

    embed()
