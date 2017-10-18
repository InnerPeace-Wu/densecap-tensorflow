# ----------------------------------------------
# DenseCap
# Written by InnerPeace
# ----------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

'''Read splits'''

import json

def read_splits():
    file = 'densecap_splits.json'
    with open(file, 'r') as f:
        data = json.load(f)
    splits = ['train', 'val', 'test']
    for split in splits:
        print("%s set has %s examples." % (split, len(data[split])))
        with open(split + '.txt', 'w') as f:
            for id in data[split]:
                f.write("%s\n" % id)


if __name__ == '__main__':
    read_splits()
