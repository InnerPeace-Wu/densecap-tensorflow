# --------------------------------------------------------
# DenseCap-Tensorflow
# Written by InnerPeace
# This file is adapted from Ross Girshick's work
# --------------------------------------------------------

import cPickle
from os.path import join as pjoin

def pickle_test():
    DEFAULT_PATH = '/home/joe/git/visual_genome_test'
    cache = pjoin(DEFAULT_PATH, '1.2_cache/pre_gt_roidb', '1.pkl')
    ori = pjoin(DEFAULT_PATH, '1.2', 'pre_gt_roidb.pkl')
    phra = pjoin(DEFAULT_PATH, '1.2', 'pre_gt_phrases.pkl')
    with open(cache, 'rb') as fc:
        data_cache = cPickle.load(fc)
    with open(ori, 'rb') as fo:
        data_ori = cPickle.load(fo)
    with open(phra, 'rb') as fp:
        data_phra = cPickle.load(fp)

    # print(data_cache)
    # print ('------------------')
    # print(data_ori)
    print data_phra


if __name__ == '__main__':
    pickle_test()
