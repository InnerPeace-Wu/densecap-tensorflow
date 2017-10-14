# ----------------------------------------------
# DenseCap
# Written by InnerPeace
# ----------------------------------------------

"""read large region description json files"""

import ijson
import json
import tqdm

def read_regions( ):
    VG_VERSION = '1.2'
    VG_PATH = '/home/joe/git/VG_raw_data'
    VG_REGION_PATH = '%s/%s/region_descriptions.json' % (VG_PATH, VG_VERSION)
    # parser = ijson.parse(open('test_region.json'))
    parser = ijson.parse(open(VG_REGION_PATH))

    last_value = None
    Dic = {}
    regions = []
    dic = {}
    for prefix, event, value in parser:
        if value == 'regions':
            Dic = {}
            regions = []
            last_value = None
        elif last_value == 'id':
            Dic['regions'] = regions
            Dic['id'] = value
            with open('test_id_%s.json' % value, 'w') as f:
                json.dump(Dic, f)
                break
        elif event == 'map_key':
            last_value = value
        elif event == 'end_map':
            regions.append(dic)
            dic = {}
            last_value = None
        elif last_value:
            dic[last_value] = value

def equal_test( ):
    new = json.load(open('test_id_1.json'))
    old = json.load(open('true_id_1.json'))
    if old == new:
        print('success!')
    else:
        print('ERROR!')

    '''OUT: yes'''

def json_line_read( ):
    with open('true_id_1.json', 'r') as f:
        for line in f:
            print(line)

if __name__ == '__main__':
    # read_regions()
    # equal_test()
    json_line_read()