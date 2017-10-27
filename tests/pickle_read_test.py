# --------------------------------------------------------
# DenseCap-Tensorflow
# Written by InnerPeace
# This file is adapted from Ross Girshick's work
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from os.path import join as pjoin
from six.moves import cPickle

def pickle_test():
    DEFAULT_PATH = '/home/joe/git/visual_genome_test'
    cache = pjoin(DEFAULT_PATH, '1.2_cache/pre_gt_roidb', '1.pkl')
    cache_flip = pjoin(DEFAULT_PATH, '1.2_cache/pre_gt_roidb', '1_flip.pkl')
    ori = pjoin(DEFAULT_PATH, '1.2', 'pre_gt_roidb.pkl')
    phra = pjoin(DEFAULT_PATH, '1.2', 'pre_gt_phrases.pkl')
    with open(cache, 'rb') as fc:
        data_cache = cPickle.load(fc)
    with open(cache_flip, 'rb') as f:
        data_flip = cPickle.load(f)
    with open(ori, 'rb') as fo:
        data_ori = cPickle.load(fo)
    with open(phra, 'rb') as fp:
        data_phra = cPickle.load(fp)
    # from IPython import embed;
    # embed()

    print(data_cache)
    print ('flip------------------')
    print(data_flip)
    print ('ori------------------')
    print(data_ori)
    print("data ori length:", len(data_ori))
    print ('phrase------------------')
    print (data_phra)
    # print (data_phra[2239])


if __name__ == '__main__':
    pickle_test()
