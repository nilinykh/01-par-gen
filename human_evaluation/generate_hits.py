'''image paragraph pair generator'''

import os
import json
import pandas as pd
import argparse
import time
import utils

def gen_ip_pair(args):
    
    '''generate n_hits image-paragraph pairs for evaluation'''
    
    gen_par_path = args.config.get('gen_par_path')
    gt_par_path = args.config.get('gt_par_path')
    
    if os.path.isfile('./hit_inputs/hit_input_' + args.model_type + '.json'):
        pass
    else:
        with open('./hit_inputs/hit_input_' + args.model_type + '.json', 'w') as f9:
            json.dump({}, f9)

    with open('./images_to_publish.json', 'r') as f4:
        images_to_publish = json.load(f4)
    with open('./hit_inputs/hit_input_' + args.model_type + '.json', 'r') as f5:
        images_published = json.load(f5)
    with open(gen_par_path + args.model_type + '-beam_2_cidertest.json', 'r') as f0:
        these_pars = json.load(f0)
        
    LINKS_LIST = []
    for img_id, img_link in images_to_publish.items():
        if img_id in images_published.keys():
            continue
        elif img_id not in images_published.keys():
            for each_result in these_pars:
                if str(each_result['image_id']) == img_id:
                    hyps = each_result['hypotheses']
                    LINKS_LIST.append([img_link, hyps, args.model_type])

    strs = [json.dumps(innerlist) for innerlist in LINKS_LIST]
    strs = "%s" % "\n".join(strs)
    with open('./hit_inputs/hit_input_' + args.model_type + '.json', 'w') as f6:
        f6.write(strs)

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='./config.json',
                      type=utils.json_file)
    parser.add_argument('--model_type', type=str, default='VISUAL',
                       help='which type of paragraphs to publish')
    args = parser.parse_args()
    
    gen_ip_pair(args)
