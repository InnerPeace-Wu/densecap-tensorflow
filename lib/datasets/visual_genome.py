# --------------------------------------------------------
# DenseCap-Tensorflow
# Written by InnerPeace
# This file is adapted from Linjie's work
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np
import scipy.sparse
import uuid
import json
import six
from tqdm import tqdm
from six.moves import xrange, cPickle
from os.path import join as pjoin

from lib.datasets.imdb import imdb
from lib.config import cfg
from lib.limit_ram.utils import pre_roidb, flip_image
from lib.limit_ram.utils import is_valid_limitRam


DEBUG = False
# TODO: It's time comsuming in limit-ram mode. Make sure to set False once you
# finished preparing gt roidbs.
USE_CACHE = True
UNK_IDENTIFIER = '<unk>'


class visual_genome(imdb):
    def __init__(self, image_set, version):
        imdb.__init__(self, 'vg_' + version + '_' + image_set)
        # image_set from ['train', 'val', 'test']
        self._image_set = image_set

        self._data_path = '%s/%s' % (cfg.DATA_DIR, version)
        cfg.CACHE_DIR = self._data_path

        if cfg.LIMIT_RAM:
            # used for limit memory reading mode
            self._cache_path = '%s/%s_cache' % (cfg.DATA_DIR, version)
            # return path of directory
            self.region_imset_path = os.path.join(self._data_path,
                                                  '%s_gt_regions' % image_set)
        else:
            # return path of the json file
            self.region_imset_path = os.path.join(self._data_path,
                                                  '%s_gt_regions.json' % image_set)
            self._gt_regions = json.load(open(self.region_imset_path))

        self._image_ext = '.jpg'
        print('data_path: %s' % self._data_path)

        self._classes = ('__background__', '__foreground__')

        self._image_index = self._load_image_set_index()

        # Default to roidb handler
        self._roidb_handler = self.gt_roidb
        self._salt = str(uuid.uuid4())
        vocab_path = os.path.join(self._data_path, 'vocabulary.txt')
        with open(vocab_path, 'r') as f:
            self._vocabulary_inverted = [line.strip() for line in f]

        self._vocabulary = dict([(w, i) for i, w in enumerate(self._vocabulary_inverted)])

        # test for overfitting a minibatch
        if cfg.ALL_TEST:
            if image_set == 'train':
                self._image_index = self._image_index[:cfg.ALL_TEST_NUM_TRAIN]
            elif image_set == 'val':
                self._image_index = self._image_index[:cfg.ALL_TEST_NUM_VAL]
            elif image_set == 'test':
                self._image_index = self._image_index[:cfg.ALL_TEST_NUM_TEST]
            else:
                raise ValueError('Please check the name of the image set.')

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
        if cfg.LIMIT_RAM:
            # load region from a json file
            with open(pjoin(self.region_imset_path, '%s.json' % index), 'r') as f:
                image_path = json.load(f)['path']
        else:
            image_path = self._gt_regions[str(index)]['path']

        assert os.path.exists(image_path), \
            'Path does not exist: {}'.format(image_path)
        return image_path

    def _load_image_set_index(self, ext='json'):
        """
        Load the indexes listed in this dataset's image set file.
        """
        if cfg.LIMIT_RAM:
            if ext == 'json':
                path = pjoin(cfg.SPLIT_DIR, 'densecap_splits.json')
                with open(path, 'r') as f:
                    # NOTE: the return index has entries with INT type
                    image_index = json.load(f)[self._image_set]
                    print ("loading splits from {}".format(path))
            elif ext == 'txt':
                path = pjoin(cfg.SPLIT_DIR, '%s.txt' % self._image_set)
                with open(path, 'r') as f:
                    image_index = [line.strip() for line in f.readlines()]
                print ("loading splits from {}".format(path))
        else:
            image_index = [key for key in self._gt_regions]

        print("Number of examples: {}".format(len(image_index)))
        return image_index

    def get_gt_regions(self):
        return [v for k, v in six.iteritems(self._gt_regions)]

    def get_gt_regions_index(self, index):
        if cfg.LIMIT_RAM:
            with open(pjoin(self.region_imset_path, '%s.json' % index), 'r') as f:
                regions = json.load(f)
        else:
            regions = self._gt_regions[index]

        return regions

    def get_vocabulary(self):
        return self._vocabulary_inverted

    def gt_roidb(self):
        if cfg.LIMIT_RAM:
            gt_roidb = self.gt_roidb_limit_ram()
        else:
            gt_roidb = self.gt_roidb_unlim_ram()

        return gt_roidb

    def gt_roidb_unlim_ram(self):
        """
        Return the database of ground-truth regions of interest.

        This function loads/saves from/to a cache file to speed up future calls.
        """
        cache_file = pjoin(self._data_path, self._image_set + '_gt_roidb.pkl')
        cache_file_phrases = pjoin(self._data_path, self._image_set + '_gt_phrases.pkl')
        if os.path.exists(cache_file) and USE_CACHE:
            with open(cache_file, 'rb') as fid:
                roidb = cPickle.load(fid)
            print('{} gt roidb loaded from {}'.format(self._image_set, cache_file))
            return roidb

        gt_roidb = [self._load_vg_annotation(index) for index in self._image_index]
        gt_phrases = {}
        for k, v in six.iteritems(self._gt_regions):
            for reg in v['regions']:
                gt_phrases[reg['region_id']] = self._line_to_stream(reg['phrase_tokens'])

        with open(cache_file, 'wb') as fid:
            cPickle.dump(gt_roidb, fid, cPickle.HIGHEST_PROTOCOL)
        with open(cache_file_phrases, 'wb') as fid:
            cPickle.dump(gt_phrases, fid, cPickle.HIGHEST_PROTOCOL)
        print('wrote gt roidb to {}'.format(cache_file))
        # print('wrote gt phrases to {}'.format(cache_file_phrases))
        return gt_roidb

    def gt_roidb_limit_ram(self):
        # cache_file_phrases = pjoin(self._cache_path, self._image_set + '_gt_phrases.pkl')
        roidb_cache_path = pjoin(self._cache_path, self._image_set + '_gt_roidb')
        if os.path.exists(roidb_cache_path) and USE_CACHE:
            print("{} gt roidb could be loaded from {}".format(self._image_set,
                                                               roidb_cache_path))
            with open(roidb_cache_path + '/image_index.json', 'r') as fi:
                self._image_index = json.load(fi)
            print("Getting gt roidb and number of examples is:{}".format(len(self._image_index)))
            return roidb_cache_path

        elif not os.path.exists(roidb_cache_path):
            os.makedirs(roidb_cache_path)

        image_index = []
        exclude_index = []
        for i in tqdm(xrange(len(self._image_index)), desc="%s" % self._image_set):
            idx = self._image_index[i]
            dictionary = self._load_vg_annotation(idx)
            path_i = roidb_cache_path + '/%s.pkl' % idx
            if not os.path.exists(path_i):
                if is_valid_limitRam(pre_roidb(dictionary)):
                    if not isinstance(idx, six.string_types):
                        idx = str(idx)
                    image_index.append(idx)
                    with open(roidb_cache_path + '/%s.pkl' % idx, 'wb') as f:
                        cPickle.dump(dictionary, f, cPickle.HIGHEST_PROTOCOL)
                else:
                    exclude_index.append(idx)

            # check for flipping only during training
            if cfg.TRAIN.USE_FLIPPED and self._image_set == 'train':
                flip_dict = flip_image(dictionary)
                flip_id = flip_dict['image_id']
                if is_valid_limitRam(pre_roidb(flip_dict)):
                    image_index.append(flip_id)
                    with open(roidb_cache_path + '/%s.pkl' % flip_id, 'wb') as f:
                        cPickle.dump(flip_dict, f, cPickle.HIGHEST_PROTOCOL)
                else:
                    exclude_index.append(flip_id)

        print("filter out {} images.".format(len(exclude_index)))
        print("remaining {} iamges for {} set".format(len(image_index),
                                                      self._image_set))
        self._image_index = image_index
        with open(roidb_cache_path + '/image_index.json', 'w') as fi:
            json.dump(image_index, fi)

        return roidb_cache_path

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
        print('loading {}'.format(filename))
        assert os.path.exists(filename), \
            'layers data not found at: {}'.format(filename)
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
        # increment the stream --
        # 0 will be the <pad> character
        # 1 will be the <SOS> character
        # 2 will be the <EOS> character

        stream = [s + 3 for s in stream]
        return stream

    def _load_vg_annotation(self, index):
        """
        Load image and bounding boxes info from XML file in the PASCAL VOC
        format.
        """
        if not isinstance(index, six.string_types):
            index = str(index)
        if not cfg.LIMIT_RAM:
            regions = self._gt_regions[index]['regions']
        else:
            with open(self.region_imset_path + '/%s.json' % index, 'r') as f:
                data_json = json.load(f)
                regions = data_json['regions']

        gt_phrases = []
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
            # replace the class id with region id so that can retrieve the caption later
            gt_classes[ix] = reg['region_id']
            overlaps[ix, 1] = 1.0
            seg_areas[ix] = (x2 - x1 + 1) * (y2 - y1 + 1)

            # if cfg.LIMIT_RAM:
            # gt_phrases[reg['region_id']] = self._line_to_stream(reg['phrase_tokens'])
            gt_phrases.append(self._line_to_stream(reg['phrase_tokens']))
            if DEBUG:
                # CHECK consistency
                for wi, w in zip(gt_phrases[ix], reg['phrase_tokens']):
                    vocab_w = self._vocabulary_inverted[wi - 1]
                    print(vocab_w, w)
                    assert (vocab_w == UNK_IDENTIFIER or vocab_w == w)

        sparse_overlaps = scipy.sparse.csr_matrix(overlaps)
        dictionary = {'boxes': boxes,
                      'gt_classes': gt_classes,
                      'gt_overlaps': sparse_overlaps,
                      'flipped': False,
                      'gt_phrases': gt_phrases,
                      'seg_areas': seg_areas}
        if cfg.LIMIT_RAM:
            dictionary.update({
                'image': data_json['path'],
                'width': data_json['width'],
                'height': data_json['height'],
                'image_id': data_json['id']
            })

        return dictionary


if __name__ == '__main__':

    from IPython import embed
    embed()
