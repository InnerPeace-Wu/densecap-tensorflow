# --------------------------------------------------------
# DenseCap-Tensorflow
# Written by InnerPeace
# This file is adapted from Linjie's work and Xinlei's work
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from os.path import join as pjoin
from lib.config import cfg
from lib.fast_rcnn.minibatch import get_minibatch
import numpy as np
import time
import json


class RoIDataLayer(object):
    """densecap data layer used for training."""

    def __init__(self, roidb, random=False):
        """set the roidb to be used by this layer during training."""
        self._roidb = roidb
        # set a random flag
        self._random = random
        self._shuffle_roidb_inds()

    def _shuffle_roidb_inds(self):
        """Randomly permute the training roidb."""

        # if the random flag is set,
        # then the database is shuffled according to system time
        # useful for the validation set.
        if self._random:
            st0 = np.random.get_state()
            millis = int(round(time.time() * 1000)) % 4294967259
            np.random.seed(millis)

        if not cfg.LIMIT_RAM:
            # with sending in the giant roidb list
            if cfg.TRAIN.ASPECT_GROUPING:
                widths = np.array([r['width'] for r in self._roidb])
                heights = np.array([r['height'] for r in self._roidb])
                horz = (widths >= heights)
                vert = np.logical_not(horz)
                horz_inds = np.where(horz)[0]
                vert_inds = np.where(vert)[0]
                inds = np.hstack((
                    np.random.permutation(horz_inds),
                    np.random.permutation(vert_inds)))
                inds = np.reshape(inds, (-1, 2))
                row_perm = np.random.permutation(np.arange(inds.shape[0]))
                inds = np.reshape(inds[row_perm, :], (-1,))
                self._perm = inds
            else:
                self._perm = np.random.permutation(np.arange(len(self._roidb)))
        else:
            # LIMIT_RAM and 'roidb' is the path to saved gt_roidbs.
            index_path = self._roidb + '/image_index.json'
            with open(index_path, 'r') as f:
                self._image_index = json.load(f)
                print("LIMIT_RAM version and load index from {}".format(index_path))
            self._perm = np.random.permutation(np.arange(len(self._image_index)))

        # restore the random state
        if self._random:
            np.random.set_state(st0)

        self._cur = 0

    def _get_next_minibatch_inds(self):
        """Return the roidb indices for the next minibatch."""
        if self._cur + cfg.TRAIN.IMS_PER_BATCH >= len(self._perm):
            self._shuffle_roidb_inds()

        db_inds = self._perm[self._cur:self._cur + cfg.TRAIN.IMS_PER_BATCH]
        self._cur += cfg.TRAIN.IMS_PER_BATCH
        return db_inds

    def _get_next_minibatch(self):
        """Return the blobs to be used for the next minibatch.

        If cfg.TRAIN.USE_PREFETCH is True, then blobs will be computed in a
        separate process and made available through self._blob_queue.
        """
        db_inds = self._get_next_minibatch_inds()
        if cfg.LIMIT_RAM:
            assert len(db_inds) == 1, "LIMIT_RAM version only support one " \
                                      "image per minibatch."
            # it is the exact file path in the 'roidb' directory.
            minibatch_db = self._image_index[db_inds[0]]
            minibatch_db = pjoin(self._roidb, "%s.pkl" % minibatch_db)
        else:
            minibatch_db = [self._roidb[i] for i in db_inds]
        return get_minibatch(minibatch_db)

    def forward(self):
        """Get blobs"""
        blobs = self._get_next_minibatch()
        return blobs
