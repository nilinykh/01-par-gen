import os
from random import seed
import pathlib
from queue import PriorityQueue
from imageio import imread
from skimage.transform import resize
from tqdm import tqdm
from collections import Counter

import numpy as np
import h5py
import json
import torch
import torch.backends.cudnn as cudnn

device = torch.device("cuda")
cudnn.benchmark = True


def create_input_files(dataset,
                       paragraph_json_path,
                       image_folder,
                       densecap_feat_folder,
                       max_sentences,
                       min_word_freq,
                       output_folder,
                       encoder_type,
                       max_words=50):
    '''
    Creates input files for training, validation, and test data.

    :param dataset: name of dataset, should be 'vg' only
    :param paragraph_json_path: path to the paragraphs and splits
    :param image_folder: folder with downloaded images
    :param max_sentences: number of paragraph sentences to sample per image
    :param min_word_freq: words occuring less frequently than this threshold are binned as <unk>s
    :param output_folder: folder to save files
    :param max_words: max length of each sampled sentence, truncate otherwise
    '''

    assert dataset in {'coco', 'flickr8k', 'flickr30k', 'vg'}

    # Read Karpathy JSON
    with open(paragraph_json_path, 'r') as j:
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
            if len(c['tokens']) <= max_words:
                captions.append(c['tokens'])
            elif len(c['tokens']) > max_words:
                captions.append(c['tokens'][:max_words])

        if len(captions) == 0:
            continue

        path = os.path.join(image_folder, img['filepath'], img['filename']) if dataset == 'vg' else os.path.join(
            image_folder, img['filename'])

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

    # Create word map
    words = [w for w in word_freq.keys() if word_freq[w] > min_word_freq]
    word_map = {k: v + 1 for v, k in enumerate(words)}
    word_map['<unk>'] = len(word_map) + 1
    word_map['<start>'] = len(word_map) + 1
    word_map['<end>'] = len(word_map) + 1
    #word_map['<empty>'] = len(word_map) + 1
    word_map['<pad>'] = 0

    # for densecap feature organisation
    with open(densecap_feat_folder + 'train_imgs_path.txt', 'r') as f:
        train_paths = [line.rstrip() for line in f.readlines()]
    with open(densecap_feat_folder + 'val_imgs_path.txt', 'r') as f:
        val_paths = [line.rstrip() for line in f.readlines()]
    with open(densecap_feat_folder + 'test_imgs_path.txt', 'r') as f:
        test_paths = [line.rstrip() for line in f.readlines()]
    train_feats = h5py.File(densecap_feat_folder + 'train_paragraph_feat.h5', 'r')['feats']
    val_feats = h5py.File(densecap_feat_folder + 'val_paragraph_feat.h5', 'r')['feats']
    test_feats = h5py.File(densecap_feat_folder + 'test_paragraph_feat.h5', 'r')['feats']

    # Create a base/root name for all output files
    base_filename = dataset + '_' + str(max_sentences) + '_sent_per_img_' + str(min_word_freq) + '_min_word_freq'

    # Save word map to a JSON
    with open(os.path.join(output_folder, 'WORDMAP_' + base_filename + encoder_type + '.json'), 'w') as j:
        json.dump(word_map, j)

    #train_val_image_paths = train_image_paths + val_image_paths
    #train_val_image_captions = train_image_captions + val_image_captions

    # Sample captions for each image, save images to HDF5 file, and captions and their lengths to JSON files
    seed(123)
    for impaths, imcaps, split in [(train_image_paths, train_image_captions, 'TRAIN'),
                                   (val_image_paths, val_image_captions, 'VAL'),
                                   (test_image_paths, test_image_captions, 'TEST')]:

        with h5py.File(os.path.join(output_folder, split + '_IMAGES_' + base_filename + encoder_type + '.hdf5'), 'a') as h:
            # Make a note of the number of captions we are sampling per image
            h.attrs['sentences_per_paragraph'] = max_sentences

            # Create dataset inside HDF5 file to store images (depends on which features have been used)
            if encoder_type == 'resnet512':
                images = h.create_dataset('images', (len(impaths), 3, 256, 256), dtype='uint8')
            elif encoder_type == 'densecap':
                images = h.create_dataset('images', (len(impaths), 50, 4096), dtype='uint8')
            # Create dataset inside HDF5 file to store image ids
            dt = h5py.special_dtype(vlen=str)
            imageids = h.create_dataset('image_ids', (len(impaths),), dtype=dt)

            print("\nReading %s images and captions, storing to file...\n" % split)

            enc_captions = []
            caplens = []

            for i, path in enumerate(tqdm(impaths)):

                imcaps[i] = [elem for elem in imcaps[i] if elem != []]

                # Sample captions
                if len(imcaps[i]) < max_sentences:
                    captions = imcaps[i] + [['<pad>'] for _ in range(max_sentences - len(imcaps[i]))]

                elif len(imcaps[i]) > max_sentences:
                    captions = imcaps[i][:max_sentences]
                else:
                    captions = imcaps[i]

                # Sanity check
                assert len(captions) == max_sentences

                if encoder_type == 'densecap':

                    if path in train_paths:
                        this_path_index = train_paths.index(path)
                        img_feat = train_feats[this_path_index]
                        assert img_feat.shape == (50, 4096)
                    elif path in val_paths:
                        this_path_index = val_paths.index(path)
                        img_feat = val_feats[this_path_index]
                        assert img_feat.shape == (50, 4096)
                    elif path in test_paths:
                        this_path_index = test_paths.index(path)
                        img_feat = test_feats[this_path_index]
                        assert img_feat.shape == (50, 4096)
                    else:
                        raise Exception('Path not found!')

                    images[i] = img_feat
                    imageids[i] = path

                elif encoder_type == 'resnet512':

                    # these loops do not take much time, leave them as it is for now
                    if path in train_paths:
                        this_path_index = train_paths.index(path)
                        img = imread(impaths[i])
                        if len(img.shape) == 2:
                            img = img[:, :, np.newaxis]
                            img = np.concatenate([img, img, img], axis=2)
                        img = resize(img, (256, 256))
                        img = img.transpose(2, 0, 1)
                        assert img.shape == (3, 256, 256)
                        assert np.max(img) <= 255
                        images[i] = img
                        imageids[i] = path
                    if path in val_paths:
                        this_path_index = val_paths.index(path)
                        img = imread(impaths[i])
                        if len(img.shape) == 2:
                            img = img[:, :, np.newaxis]
                            img = np.concatenate([img, img, img], axis=2)
                        img = resize(img, (256, 256))
                        img = img.transpose(2, 0, 1)
                        assert img.shape == (3, 256, 256)
                        assert np.max(img) <= 255
                        images[i] = img
                        imageids[i] = path
                    if path in test_paths:
                        this_path_index = test_paths.index(path)
                        img = imread(impaths[i])
                        if len(img.shape) == 2:
                            img = img[:, :, np.newaxis]
                            img = np.concatenate([img, img, img], axis=2)
                        img = resize(img, (256, 256))
                        img = img.transpose(2, 0, 1)
                        assert img.shape == (3, 256, 256)
                        assert np.max(img) <= 255
                        images[i] = img
                        imageids[i] = path

                for j, cap in enumerate(captions):
                    # Encode captions
                    if cap[0] != '<pad>':

                        enc_c = [word_map['<start>']] + [word_map.get(word, word_map['<unk>']) for word in cap] + [
                            word_map['<end>']] + [word_map['<pad>']] * (max_words - len(cap))

                    elif cap[0] == '<pad>':

                        enc_c = [word_map['<pad>']] * (max_words + 2)

                    # Find caption lengths
                    c_len = len(cap) + 2

                    enc_captions.append(enc_c)
                    caplens.append(c_len)

            # Sanity check
            assert images.shape[0] * max_sentences == len(enc_captions) == len(caplens)

            # Save encoded captions and their lengths to JSON files
            with open(os.path.join(output_folder, split + '_CAPTIONS_' + base_filename + encoder_type + '.json'), 'w') as j:
                json.dump(enc_captions, j)

            with open(os.path.join(output_folder, split + '_CAPLENS_' + base_filename + encoder_type + '.json'), 'w') as j:
                json.dump(caplens, j)

def caplens_eos(caplens, max_sents):
    caplens_f = torch.flatten(caplens).type(torch.LongTensor)
    caplens_f[caplens_f != 3] = 0
    caplens_f[caplens_f == 3] = 1
    caplens_f = caplens_f.unsqueeze(1)
    init_inx = np.array([i for i in range(len(caplens_f)) if i % max_sents == 0])
    return caplens_f, init_inx


def save_checkpoint(data_name,
                    epoch,
                    epochs_since_improvement,
                    encoder,
                    sentence_decoder,
                    word_decoder,
                    encoder_optimizer,
                    sentence_optimizer,
                    word_optimizer,
                    val_loss,
                    is_best,
                    exp_num):
    '''
    save model's checkpoint (with controlling if current model is the best one or not based on validation loss)
    '''
    state = {'epoch': epoch,
             'epochs_since_improvement': epochs_since_improvement,
             'val_loss': val_loss,
             'encoder': encoder,
             'sentence_decoder': sentence_decoder,
             'word_decoder': word_decoder,
             'encoder_optimizer': encoder_optimizer,
             'sentence_optimizer': sentence_optimizer,
             'word_optimizer': word_optimizer,
             'exp_num': exp_num}

    pathlib.Path(f'./checkpoints/{exp_num}').mkdir(parents=True, exist_ok=True)
    pathname = f'./checkpoints/{exp_num}/'
    filename = f'checkpoint.pth.tar'
    best_filename = f'BEST_checkpoint.pth.tar'

    torch.save(state, pathname + filename)
    if is_best:
        torch.save(state, pathname + best_filename)


class AverageMeter(object):
    """
    Keeps track of most recent, average, sum, and count of a metric.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def adjust_learning_rate(optimizer, shrink_factor):
    """
    Shrinks learning rate by a specified factor.

    :param optimizer: optimizer whose learning rate must be shrunk.
    :param shrink_factor: factor in interval (0, 1) to multiply learning rate with.
    """

    print("\nDECAYING learning rate.")
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr'] * shrink_factor
    print("The new learning rate is %f\n" % (optimizer.param_groups[0]['lr'],))


def accuracy(scores, targets, k):
    """
    Computes top-k accuracy, from predicted and true labels.

    :param scores: scores from the model
    :param targets: true labels
    :param k: k in top-k accuracy
    :return: top-k accuracy
    """

    batch_size = targets.size(0)
    _, ind = scores.topk(k, 1, True, True)
    correct = ind.eq(targets.view(-1, 1).expand_as(ind))
    correct_total = correct.view(-1).float().sum()  # 0D tensor
    return correct_total.item() * (100.0 / batch_size)
