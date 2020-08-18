'''image paragraph pair generator'''

import os
import json
import pandas as pd
import configparser

def gen_ip_pair(n_hits, par_type):
    '''generate n_hits image-paragraph pairs for evaluation'''
    CONFIG = configparser.ConfigParser()
    CONFIG.read('config.ini')
    gen_par_path = CONFIG['basepath']['gen_par_path']
    gt_par_path = CONFIG['basepath']['gt_par_path']
    
    if os.path.isfile('./images_published_' + par_type + '.json'):
        pass
    else:
        with open('./images_published_' + par_type + '.json', 'w') as f9:
            json.dump({}, f9)

    with open('./images_to_publish.json', 'r') as f4:
        images_to_publish = json.load(f4)
    with open('./images_published_' + par_type + '.json', 'r') as f5:
        images_published = json.load(f5)
    with open(gen_par_path + par_type + '-beam_2_cidertest.json', 'r') as f0:
        these_pars = json.load(f0)
        
    LINKS_LIST = []
    add_published = {}
    h_num = 0
    for img_id, img_link in images_to_publish.items():
        if img_id in images_published.keys():
            continue
        elif img_id not in images_published.keys():
            if h_num < n_hits:
                for each_result in these_pars:
                    if str(each_result['image_id']) == img_id:
                        hyps = each_result['hypotheses']
                        LINKS_LIST.append((img_link, hyps))
                        h_num += 1
                        add_published[img_id] = hyps
    images_published.update(add_published)
    with open('./images_published_' + par_type + '.json', 'w') as f6:
        json.dump(images_published, f6)
    return LINKS_LIST

if __name__ == '__main__':
    GENER_PAIRS = gen_ip_pair(2, 'LANGUAGE')
