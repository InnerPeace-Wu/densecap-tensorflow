# ----------------------------------------------
# DenseCap
# Written by InnerPeace
# ----------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

"""read large region description json files"""

import ijson
import json
import sys
import os

VG_VERSION = '1.2'
VG_PATH = '/home/joe/git/VG_raw_data'
# TODO: delete testing option
# VG_REGION_PATH = '%s/%s/region_descriptions.json' % (VG_PATH, VG_VERSION)
VG_REGION_PATH = '%s/%s/region_descriptions_test.json' % (VG_PATH, VG_VERSION)
# REGION_JSON = '%s/%s/regions' % (VG_PATH, VG_VERSION)
REGION_JSON = '%s/%s/regions_test' % (VG_PATH, VG_VERSION)


def read_regions():
    if not os.path.exists(REGION_JSON):
        os.makedirs(REGION_JSON)
    parser = ijson.parse(open(VG_REGION_PATH))
    last_value = None
    Dic = {}
    regions = []
    dic = {}
    count = 0
    for prefix, event, value in parser:
        sys.stdout.write('>>> %d \r' % count)
        sys.stdout.flush()
        if value == 'regions':
            Dic = {}
            regions = []
            last_value = None
        elif last_value == 'id' and value:
            count += 1
            Dic['regions'] = regions
            Dic['id'] = value
            with open(REGION_JSON + '/%s.json' % value, 'w') as f:
                json.dump(Dic, f)
        elif event == 'map_key':
            last_value = value
        elif event == 'end_map':
            regions.append(dic)
            dic = {}
            last_value = None
        elif last_value:
            dic[last_value] = value


if __name__ == '__main__':
    read_regions()
