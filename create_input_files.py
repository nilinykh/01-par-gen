import os
import argparse
from random import seed
from tqdm import tqdm
from collections import Counter
import h5py
import json


def main(params):
    
    '''
    Creates input files for training, validation, and test dat
    '''

    # Read Karpathy-styled JSON
    with open(params['paragraph_json_path'] + 'dataset_paragraphs_v1.json', 'r') as j:
        data = json.load(j)

    # Read image paths and captions for each image
    train_image_paths = []
    train_image_captions = []
    val_image_paths = []
    val_image_captions = []
    test_image_paths = []
    test_image_captions = []
    word_freq = Counter()

    for img in data['images']:
        captions = []
        for c in img['sentences']:
            # Update word frequency
            word_freq.update(c['tokens'])
            if len(c['tokens']) <= params['max_words']:
                captions.append(c['tokens'])
            elif len(c['tokens']) > params['max_words']:
                captions.append(c['tokens'][:params['max_words']])
        if len(captions) == 0:
            continue

        path = os.path.join(params['image_folder'], img['filepath'], img['filename'])
        
        if img['split'] in {'train'}:
            train_image_paths.append(path)
            train_image_captions.append(captions)
        elif img['split'] in {'val'}:
            val_image_paths.append(path)
            val_image_captions.append(captions)
        elif img['split'] in {'test'}:
            test_image_paths.append(path)
            test_image_captions.append(captions)

    # Sanity check
    assert len(train_image_paths) == len(train_image_captions)
    assert len(val_image_paths) == len(val_image_captions)
    assert len(test_image_paths) == len(test_image_captions)
    
    # Save files with image paths

    with open(params['data_path'] + 'train_imgs_path.txt', 'w') as f:
        for path in train_image_paths:
            f.write(path)
            f.write('\n')
    with open(params['data_path'] + 'val_imgs_path.txt', 'w') as f:
        for path in val_image_paths:
            f.write(path)
            f.write('\n')
    with open(params['data_path'] + 'test_imgs_path.txt', 'w') as f:
        for path in test_image_paths:
            f.write(path)
            f.write('\n')

    # Create word map
    words = [w for w in word_freq.keys() if word_freq[w] > params['min_word_freq']]
    word_map = {k: v + 1 for v, k in enumerate(words)}
    word_map['<unk>'] = len(word_map) + 1
    word_map['<start>'] = len(word_map) + 1
    word_map['<end>'] = len(word_map) + 1
    word_map['<pad>'] = 0
        
    # get visual features, background representation, bounding boxes
    train_feats = h5py.File(params['data_path'] + 'train_multimodal_50.h5', 'r')['feats']
    val_feats = h5py.File(params['data_path'] + 'val_multimodal_50.h5', 'r')['feats']
    test_feats = h5py.File(params['data_path'] + 'test_multimodal_50.h5', 'r')['feats']
    
    train_phrases = h5py.File(params['data_path'] + 'train_multimodal_50.h5', 'r')['scores']
    val_phrases = h5py.File(params['data_path'] + 'val_multimodal_50.h5', 'r')['scores']
    test_phrases = h5py.File(params['data_path'] + 'test_multimodal_50.h5', 'r')['scores']
    
    train_boxes = h5py.File(params['data_path'] + 'train_multimodal_50.h5', 'r')['boxes']
    val_boxes = h5py.File(params['data_path'] + 'val_multimodal_50.h5', 'r')['boxes']
    test_boxes = h5py.File(params['data_path'] + 'test_multimodal_50.h5', 'r')['boxes']
        
    # Create a base/root name for all output files
    base_filename = str(params['max_sentences']) + '_sent_' + str(params['min_word_freq']) + '_min_word'

    # Save word map to a JSON
    with open(os.path.join(params['output_folder'], 'wordmap_' + base_filename + '.json'), 'w') as j:
        json.dump(word_map, j)

    # save images, captions, their lengths to separate files
    seed(123)
    for impaths, imcaps, split in [(train_image_paths, train_image_captions, 'TRAIN'),
                                   (val_image_paths, val_image_captions, 'VAL'),
                                   (test_image_paths, test_image_captions, 'TEST')]:

        with h5py.File(os.path.join(params['output_folder'], split + '_images_' + base_filename + '.hdf5'), 'a') as h:
            
            h.attrs['sentences_per_paragraph'] = params['max_sentences']
            images = h.create_dataset('images', (len(impaths), 50, 4096), dtype='uint8')
            dt = h5py.special_dtype(vlen=str)
            imageids = h.create_dataset('image_ids', (len(impaths),), dtype=dt)
            phrases = h.create_dataset('phrases', (len(impaths), 50, 15, 512))
            boxes = h.create_dataset('boxes', (len(impaths), 50, 4))

            print("\nReading %s images and captions, storing to file...\n" % split)

            enc_captions = []
            caplens = []

            for i, path in enumerate(tqdm(impaths)):

                imcaps[i] = [elem for elem in imcaps[i] if elem != []]

                # Sample captions
                if len(imcaps[i]) < params['max_sentences']:
                    captions = imcaps[i] + [['<pad>'] for _ in range(params['max_sentences'] - len(imcaps[i]))]
                elif len(imcaps[i]) > params['max_sentences']:
                    captions = imcaps[i][:params['max_sentences']]
                else:
                    captions = imcaps[i]

                # Sanity check
                assert len(captions) == params['max_sentences']

                if path in train_image_paths:
                    this_path_index = train_image_paths.index(path)
                    img_feat = train_feats[this_path_index]
                    img_phrases = train_phrases[this_path_index]
                    img_boxes = train_boxes[this_path_index]
                    assert img_feat.shape == (50, 4096)
                    assert img_phrases.shape == (50, 15, 512)
                    assert img_boxes.shape == (50, 4)
                elif path in val_image_paths:
                    this_path_index = val_image_paths.index(path)
                    img_feat = val_feats[this_path_index]
                    img_phrases = val_phrases[this_path_index]
                    img_boxes = val_boxes[this_path_index]
                    assert img_feat.shape == (50, 4096)
                    assert img_phrases.shape == (50, 15, 512)
                    assert img_boxes.shape == (50, 4)
                elif path in test_image_paths:
                    this_path_index = test_image_paths.index(path)
                    img_feat = test_feats[this_path_index]
                    img_phrases = test_phrases[this_path_index]
                    img_boxes = test_boxes[this_path_index]
                    assert img_feat.shape == (50, 4096)
                    assert img_phrases.shape == (50, 15, 512)
                    assert img_boxes.shape == (50, 4)
                else:
                    raise Exception('Path not found!')

                images[i] = img_feat
                imageids[i] = path
                phrases[i] = img_phrases
                boxes[i] = img_boxes

                for j, cap in enumerate(captions):
                    # Encode captions
                    if cap[0] != '<pad>':
                        # if it is not empty sentence
                        enc_c = [word_map['<start>']] + [word_map.get(word, word_map['<unk>']) for word in cap] + [word_map['<end>']] + [word_map['<pad>']] * (params['max_words'] - len(cap))
                        c_len = len(cap) + 2
                    elif cap[0] == '<pad>':
                        enc_c = [word_map['<pad>']] * (params['max_words'] + 2)
                        c_len = 0
                    enc_captions.append(enc_c)
                    caplens.append(c_len)

            # Sanity check
            assert images.shape[0] * params['max_sentences'] == len(enc_captions) == len(caplens)

            # Save encoded captions and their lengths to JSON files
            with open(os.path.join(params['output_folder'], split + '_captions_' + base_filename + '.json'), 'w') as j:
                json.dump(enc_captions, j)
            with open(os.path.join(params['output_folder'], split + '_caplens_' + base_filename + '.json'), 'w') as j:
                json.dump(caplens, j)

                
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--paragraph_json_path', default='/home/xilini/par-gen/01-par-gen/data/', help='location of the preprocessed paragraphs')
    parser.add_argument('--image_folder', default='/home/xilini/vis-data/', help='location of original VG images')
    parser.add_argument('--data_path', default='/home/xilini/par-data/', help='location with densecap output files')
    parser.add_argument('--max_sentences', default=6, help='how many sentence to keep in the paragraphs')
    parser.add_argument('--max_words', default=50, help='how many words to keep in the paragraph sentences')
    parser.add_argument('--min_word_freq', default=1, help='minimum word frequency to keep')
    parser.add_argument('--output_folder', default='/home/xilini/par-gen/01-par-gen/data/', help='output folder for h5 files')
    args = parser.parse_args()
    params = vars(args) # converty to ordinary dictionary
    main(params)