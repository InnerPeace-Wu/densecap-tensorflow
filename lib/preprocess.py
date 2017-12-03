# ----------------------------------------------
# DenseCap
# Written by InnerPeace
# This file is adapted from Linjie's work
# ----------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

"""
Preprocessing of visual genome dataset, including vocabularity generation,
removing invalid bboxes and phrases, tokenization, and result saving
"""

import sys
sys.path.append("..")

import itertools
import os
import string
import json
import time
import numpy as np
from six.moves import xrange
from collections import Counter
from config import cfg
from info.read_splits import read_splits
import os.path as osp
from tqdm import tqdm
import argparse

parser = argparse.ArgumentParser(description='Preprocessing visual genome')
parser.add_argument('--version', dest='version', type=float, default=1.2, help='the version of visual genome dataset.')
# NOTE: one need to change the path accordingly
parser.add_argument('--path', dest='path', type=str, default='/home/joe/git/VG_raw_data', help='directory saving the raw dataset')
parser.add_argument('--output_dir', dest='output_dir', type=str, default='/home/joe/git/visual_genome', help='output directory of data files')
parser.add_argument('--limit_ram', dest='limit_ram', action='store_true')
parser.add_argument('--max_words', dest='max_words', type=int, default=10, help='maximum length of words for training.')
args = parser.parse_args()

MAX_WORDS = args.max_words
VG_VERSION = args.version
VG_PATH = args.path
VG_IMAGE_ROOT = '%s/images' % VG_PATH
cfg.LIMIT_RAM = args.limit_ram
if cfg.LIMIT_RAM:
    # regions directory path
    VG_REGION_PATH = '%s/%s/regions' % (VG_PATH, VG_VERSION)
else:
    # read whole regions with a json file
    VG_REGION_PATH = '%s/%s/region_descriptions.json' % (VG_PATH, VG_VERSION)

VG_METADATA_PATH = '%s/%s/image_data.json' % (VG_PATH, VG_VERSION)
vocabulary_size = cfg.VOCAB_SIZE  # 10497#from dense caption paper
HAS_VOCAB = False
OUTPUT_DIR = args.output_dir + '/%s' % VG_VERSION
# In default, we read from json file
READ_SPLITS_FROM_TXT = False
SPLITS_JSON = osp.join(cfg.ROOT_DIR, 'info/densecap_splits.json')

# UNK_IDENTIFIER is the word used to identify unknown words
UNK_IDENTIFIER = '<unk>'


class VGDataProcessor:
    def __init__(self, split_name, image_data, regions_all=None, vocab=None,
                 split_ids=[], max_words=MAX_WORDS):
        self.max_words = max_words
        self.images = {}
        phrases_all = []
        num_invalid_bbox = 0
        num_bbox = 0
        num_empty_phrase = 0
        if cfg.LIMIT_RAM:
            self.save_path = OUTPUT_DIR + '/%s_gt_regions' % split_name
            if not os.path.exists(self.save_path):
                os.makedirs(self.save_path)

        tic = time.time()
        for i in tqdm(xrange(len(image_data)), desc='%s' % split_name):
            image_info = image_data[i]
            # NOTE: for VG_1.2 and VG_1.0 the key in image_info about id is different.
            im_id = image_info['image_id']

            if not im_id in split_ids:
                continue

            # open the region description file
            if cfg.LIMIT_RAM:
                with open(VG_REGION_PATH + '/%s.json' % im_id, 'r') as f:
                    item = json.load(f)
            else:
                item = regions_all[i]
            if item['id'] != im_id:
                print('region and image metadata inconsistent with regions id: %s, image id: %s' %
                      (item['id'], image_info['image_id']))
                exit()
            # tokenize phrase
            num_bbox += len(item['regions'])
            regions_filt = []
            for obj in item['regions']:
                # remove invalid regions
                if obj['x'] < 0 or obj['y'] < 0 or \
                        obj['width'] <= 0 or obj['height'] <= 0 or \
                        obj['x'] + obj['width'] >= image_info['width'] or \
                        obj['y'] + obj['height'] >= image_info['height']:
                    num_invalid_bbox += 1
                    continue
                phrase = obj['phrase'].strip().encode('ascii', 'ignore').lower()

                # remove empty sentence
                if len(phrase) == 0:
                    num_empty_phrase += 1
                    continue

                obj['phrase_tokens'] = phrase.translate(None, string.punctuation).split()
                # remove regions with caption longer than max_words
                if len(obj['phrase_tokens']) > max_words:
                    continue
                regions_filt.append(obj)
                phrases_all.append(obj['phrase_tokens'])
            im_path = '%s/%d.jpg' % (VG_IMAGE_ROOT, im_id)
            Dict = {'path': im_path, 'regions': regions_filt, 'id': im_id,
                    'height': image_info['height'], 'width': image_info['width']}
            if cfg.LIMIT_RAM:
                with open(self.save_path + '/%s.json' % im_id, 'wb') as f:
                    json.dump(Dict, f)
            else:
                self.images[item['id']] = Dict
        toc = time.time()
        print('processing %s set with time: %.2f seconds' % (split_name, toc - tic))
        print("there are %d invalid bboxes out of %d" % (num_invalid_bbox, num_bbox))
        print("there are %d empty phrases after triming" % num_empty_phrase)
        if vocab is None:
            self.init_vocabulary(phrases_all)
        else:
            self.vocabulary_inverted = vocab
        self.vocabulary = {}
        for index, word in enumerate(self.vocabulary_inverted):
            self.vocabulary[word] = index

    def init_vocabulary(self, phrases_all):
        word_freq = Counter(itertools.chain(*phrases_all))
        print("Found %d unique word tokens." % len(word_freq.items()))
        vocab_freq = word_freq.most_common(vocabulary_size - 1)
        self.vocabulary_inverted = [x[0] for x in vocab_freq]
        self.vocabulary_inverted.insert(0, UNK_IDENTIFIER)
        print("Using vocabulary size %d." % vocabulary_size)
        print("The least frequent word in our vocabulary is '%s' and appeared %d times." %
              (vocab_freq[-1][0], vocab_freq[-1][1]))

    def dump_vocabulary(self, vocab_filename):
        print('Dumping vocabulary to file: %s' % vocab_filename)
        with open(vocab_filename, 'wb') as vocab_file:
            for word in self.vocabulary_inverted:
                vocab_file.write('%s\n' % word)
        print('Done.')


VG_IMAGE_PATTERN = '%s/%%d.jpg' % VG_IMAGE_ROOT

# NOTE: one need to run read splits to generate separate splits
SPLITS_PATTERN = cfg.ROOT_DIR + '/info/%s.txt'


def process_dataset(split_name, vocab=None):
    # 1. read split ids from separate txt files
    if READ_SPLITS_FROM_TXT:
        read_splits()
        split_image_ids = []
        with open(SPLITS_PATTERN % split_name, 'r') as split_file:
            for line in split_file.readlines():
                line_id = int(line.strip())
                split_image_ids.append(line_id)

    # 2. read split ids from json file
    else:
        with open(SPLITS_JSON, 'r') as f:
            split_image_ids = json.load(f)[split_name]
    print('split image number: %d for split name: %s' % (len(split_image_ids), split_name))

    print('start loading json files...')
    t1 = time.time()
    if not cfg.LIMIT_RAM:
        regions_all = json.load(open(VG_REGION_PATH))
    else:
        regions_all = None
    image_data = json.load(open(VG_METADATA_PATH))
    t2 = time.time()
    print('%f seconds for loading' % (t2 - t1))
    processor = VGDataProcessor(split_name, image_data, regions_all,
                                split_ids=split_image_ids, vocab=vocab)

    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    if vocab is None:
        vocab_out_path = '%s/vocabulary.txt' % OUTPUT_DIR
        processor.dump_vocabulary(vocab_out_path)

    # dump image region dict
    if not cfg.LIMIT_RAM:
        with open(OUTPUT_DIR + '/%s_gt_regions.json' % split_name, 'w') as f:
            json.dump(processor.images, f)

    return processor.vocabulary_inverted


def process_vg():
    vocab = None
    # use existing vocabulary
    if HAS_VOCAB:
        vocab_path = '%s/vocabulary.txt' % OUTPUT_DIR
        with open(vocab_path, 'r') as f:
            vocab = [line.strip() for line in f]

    datasets = ['train', 'val', 'test']
    for split_name in datasets:
        vocab = process_dataset(split_name, vocab=vocab)


if __name__ == "__main__":
    process_vg()
