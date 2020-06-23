'''

original source of data and splits: https://cs.stanford.edu/people/ranjaykrishna/im2p/index.html

    input1: paragraph dataset, .json file
    input2: merged file with splits (train, val, test), .json file
    
    output1: paragraphs in MSCOCO Karpathy style
    
'''

import json
import argparse
from tqdm import tqdm

from spacy.lang.en import English
from autocorrect import Speller
spell = Speller(lang='en')
nlp = English()
nlp.add_pipe(nlp.create_pipe('sentencizer'))

    
def main(params):
    
    with open(params['input_par_json'] + 'paragraphs_v1.json', 'r') as f:
        paragraphs = json.load(f)
    with open(params['input_splits_json'] + 'splits.json', 'r') as i:
        splits = json.load(i)
        
    curr_sentence_id = 0
    all_images = {}
    all_images['images'] = []
    
    for par_num, paragraph in tqdm(enumerate(paragraphs, 1)):
        
        filepath = paragraph['url'].split('/')[-2]
        filename = paragraph['url'].split('/')[-1]
        imgid = par_num
        split = splits[str(filename.split('.')[0])]

        paragraph = paragraph['paragraph']
        par_spacy = nlp(paragraph)
        sentences = [sent for sent in par_spacy.sents]

        sent_dict = {}
        sent_dict['filepath'] = filepath
        sent_dict['filename'] = filename
        sent_dict['imgid'] = imgid
        sent_dict['split'] = split
        sent_dict['sentences'] = []

        sent_ids = []
        tokens_dict = {}

        for raw_sentence in sentences:
            tokens_dict = {}
            tokens = [token.orth_.lower().rstrip() for token in raw_sentence]
            tokens = [t for t in tokens if t != '']
            # check mispelled
            tokens = [spell(t) for t in tokens]
            
            if len(tokens) != 0:
                tokens_dict['tokens'] = tokens
                tokens_dict['raw'] = str(raw_sentence)
                tokens_dict['imgid'] = imgid
                tokens_dict['sentid'] = curr_sentence_id
                sent_ids.append(curr_sentence_id)
                curr_sentence_id += 1
                sent_dict['sentences'].append(tokens_dict)

        sent_dict['stanford_par_id'] = int(filename.split('.jpg')[0])
        sent_dict['sentids'] = sent_ids
        all_images['images'].append(sent_dict)
        
    with open(params['output_par_json'] + 'dataset_paragraphs_v1.json', 'w') as w:
        json.dump(all_images, w)

    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_par_json', default='/home/xilini/par-gen/01-par-gen/data/', help='location of the file with original paragraphs')
    parser.add_argument('--input_splits_json', default='/home/xilini/par-gen/01-par-gen/data/', help='location of the merged paragraph splits')
    parser.add_argument('--output_par_json', default='/home/xilini/par-gen/01-par-gen/data/', help='location of the output file with preprocessed paragraphs')
    args = parser.parse_args()
    params = vars(args) # converty to ordinary dictionary
    main(params)