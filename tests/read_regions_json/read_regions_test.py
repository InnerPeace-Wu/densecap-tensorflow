# ----------------------------------------------
# DenseCap
# Written by InnerPeace
# ----------------------------------------------

"""read large region description json files"""

import ijson
import json
# import tqdm

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
    new = json.load(open('true_id_1_out.json'))
    old = json.load(open('true_id_1.json'))
    if old == new:
        print('success!')
    else:
        print('ERROR!')

    '''OUT: success!'''


def json_line_read( ):
    '''This is not working'''

    with open('true_id_1.json', 'r') as f:
        for line in f:
            print(line)


def read_time_test( ):
    path = '/home/joe/git/visual_genome_test/1.2/pre_gt_regions/1.json'
    import time
    tic = time.time()
    with open(path, 'r') as f:
        data = json.load(f)
    toc = time.time()
    print ('read time: %s seconds' % (toc - tic))

def read_all_regions_test():
    '''it gonna kill my computer'''
    from tqdm import tqdm
    path = '/home/joe/git/visual_genome/1.2/train_gt_regions/'
    split_path = '/home/joe/git/densecap/info/densecap_splits.json'
    with open(split_path, 'r') as fid:
        img_index = json.load(fid)['train']
    all_regions = {}
    for i in tqdm(xrange(len(img_index)), desc='train set'):
        idx = img_index[i]
        with open(path+'%s.json'%idx, 'r') as f:
            all_regions["%s"%idx] = json.load(f)

if __name__ == '__main__':
    # read_regions()
    # equal_test()
    # json_line_read()
    # read_time_test()
    read_all_regions_test()
