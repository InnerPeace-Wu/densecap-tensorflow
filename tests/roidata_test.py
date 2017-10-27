# --------------------------------------------------------
# DenseCap-Tensorflow
# Written by InnerPeace
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from lib.fast_rcnn.layer import RoIDataLayer
from lib.config import cfg
from lib.datasets.visual_genome import visual_genome
import lib.fast_rcnn.roidb as rdl_roidb
import cv2
import numpy as np
from six.moves import xrange

# cfg.LIMIT_RAM = False
DEFAULT_PATH = '/home/joe/git/visual_genome_test/1.2'


# def roidata_test(roidb, num_classes=2):
#     data = RoIDataLayer(roidb, num_classes=num_classes)

def get_training_roidb(imdb):
    """Returns a roidb (Region of Interest database) for use in training."""
    if cfg.TRAIN.USE_FLIPPED and not cfg.LIMIT_RAM:
        print('Appending horizontally-flipped training examples...')
        imdb.append_flipped_images()
        print('done')

    print('Preparing training data...')
    rdl_roidb.prepare_roidb(imdb)
    print('done')

    return imdb.roidb


def filter_roidb(roidb):
    """Remove roidb entries that have no usable RoIs."""

    def is_valid(entry):
        # Valid images have:
        #   (1) At least one foreground RoI OR
        #   (2) At least one background RoI
        overlaps = entry['max_overlaps']
        # find boxes with sufficient overlap
        fg_inds = np.where(overlaps >= cfg.TRAIN.FG_THRESH)[0]
        # Select background RoIs as those within [BG_THRESH_LO, BG_THRESH_HI)
        bg_inds = np.where((overlaps < cfg.TRAIN.BG_THRESH_HI) &
                           (overlaps >= cfg.TRAIN.BG_THRESH_LO))[0]
        # image is only valid if such boxes exist
        valid = len(fg_inds) > 0 or len(bg_inds) > 0
        return valid

    num = len(roidb)
    filtered_roidb = [entry for entry in roidb if is_valid(entry)]
    num_after = len(filtered_roidb)
    print('Filtered {} roidb entries: {} -> {}'.format(num - num_after,
                                                       num, num_after))
    return filtered_roidb


def vis_regions(im, regions, phrases=None, path='/home/joe/git/VG_raw_data/images_test'):
    vocab_path = '%s/vocabulary.txt' % DEFAULT_PATH
    with open(vocab_path, 'r') as f:
        vocab = [line.strip() for line in f]

    mean_values = np.array([[[102.9801, 115.9465, 122.7717]]])
    im = im + mean_values  # offset to original values

    for i in xrange(len(regions)):
        if i > 9:
            print ('save 10 examples and break out.')
            break
        bbox = regions[i, :4]
        region_id = regions[i, 4]
        # position 0,1,2 have been taken
        caption = ' '.join([vocab[j - 3] if j-3>=0 else "" for j in phrases[i]])
        im_new = np.copy(im)
        cv2.rectangle(im_new, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 0, 255), 2)
        cv2.imwrite('%s/%s.jpg' % (path, caption), im_new)

def get_data_test():
    imdb = visual_genome('pre', '1.2')
    if cfg.LIMIT_RAM:
        roidb = imdb.roidb
    else:
        roidb = get_training_roidb(imdb)
        roidb = filter_roidb(roidb)
    rdata = RoIDataLayer(roidb)
    data = rdata.forward()

    return data


if __name__ == '__main__':
    imdb = visual_genome('pre', '1.2')
    if cfg.LIMIT_RAM:
        roidb = imdb.roidb
    else:
        roidb = get_training_roidb(imdb)
        roidb = filter_roidb(roidb)
    rdata = RoIDataLayer(roidb)
    data = rdata.forward()
    # data = rdata.forward()
    print(data)
    regions = data['gt_boxes']
    im = data['data'][0]
    phrases = data['gt_phrases']
    vis_regions(im, regions, phrases=phrases)

    # from IPython import embed;
    #
    # embed()
