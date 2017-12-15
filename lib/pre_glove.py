# --------------------------------------------------------
# DenseCap-Tensorflow
# Written by InnerPeace
# This file is adapted from cs224-2017 stanford
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


from tensorflow.python.platform import gfile
from os.path import join as pjoin
from tqdm import *
import numpy as np
import os

from config import cfg


_PAD = b"<pad>"
_SOS = b"<sos>"
_EOS = b"<eos>"


def initialize_vocabulary(vocabulary_path):
    # map vocab to word embeddings
    if gfile.Exists(vocabulary_path):
        rev_vocab = [_PAD, _SOS, _EOS]
        with gfile.GFile(vocabulary_path, mode="r") as f:
            rev_vocab.extend(f.readlines())
        rev_vocab = [line.strip('\n') for line in rev_vocab]
        vocab = dict([(x, y) for (y, x) in enumerate(rev_vocab)])
        return vocab, rev_vocab
    else:
        raise ValueError("Vocabulary file %s not found.", vocabulary_path)


def process_glove(vocab_list, save_path, size=4e5, random_init=True):
    """
    :param vocab_list: [vocab]
    :return:
    """
    if not gfile.Exists(save_path + ".npz"):
        glove_path = os.path.join(cfg.DATA_DIR, "glove.6B.{}d.txt".format(cfg.GLOVE_DIM))
        if random_init:
            glove = np.random.randn(len(vocab_list), cfg.GLOVE_DIM)
        else:
            glove = np.zeros((len(vocab_list), cfg.GLOVE_DIM))
        found = 0
        with open(glove_path, 'r') as fh:
            for line in tqdm(fh, total=size):
                array = line.lstrip().rstrip().split(" ")
                word = array[0]
                vector = list(map(float, array[1:]))
                if word in vocab_list:
                    idx = vocab_list.index(word)
                    glove[idx, :] = vector
                    found += 1
                if word.capitalize() in vocab_list:
                    idx = vocab_list.index(word.capitalize())
                    glove[idx, :] = vector
                    found += 1
                if word.upper() in vocab_list:
                    idx = vocab_list.index(word.upper())
                    glove[idx, :] = vector
                    found += 1

        print("{}/{} of word vocab have corresponding vectors in {}".format(found, len(vocab_list), glove_path))
        np.savez_compressed(save_path, glove=glove)
        print("saved trimmed glove matrix at: {}".format(save_path))


if __name__ == "__main__":
    vocab_path = pjoin(cfg.CACHE_DIR, 'vocabulary.txt')
    vocab, rev_vocab = initialize_vocabulary(vocab_path)
    process_glove(rev_vocab, cfg.DATA_DIR + "/glove.trimmed.{}".format(cfg.GLOVE_DIM),
                  random_init=True)
