import os
from random import seed
import pathlib
from queue import PriorityQueue
from imageio import imread
from skimage.transform import resize
from tqdm import tqdm
from collections import Counter
import matplotlib.pyplot as plt
import operator

import numpy as np
import h5py
import json
import torch
from torch import nn
import torch.backends.cudnn as cudnn
from torch.nn.utils.rnn import pack_padded_sequence
import torch.nn.functional as F

device = torch.device("cuda")
cudnn.benchmark = True


def create_input_files(dataset,
                       paragraph_json_path,
                       image_folder,
                       image_paths,
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

    with open(image_paths + 'train_imgs_path.txt', 'w') as f:
        for path in train_image_paths:
            f.write(path)
            f.write('\n')
    with open(image_paths + 'val_imgs_path.txt', 'w') as f:
        for path in val_image_paths:
            f.write(path)
            f.write('\n')
    with open(image_paths + 'test_imgs_path.txt', 'w') as f:
        for path in test_image_paths:
            f.write(path)
            f.write('\n')

    # Create word map
    words = [w for w in word_freq.keys() if word_freq[w] > min_word_freq]
    word_map = {k: v + 1 for v, k in enumerate(words)}
    word_map['<unk>'] = len(word_map) + 1
    word_map['<start>'] = len(word_map) + 1
    word_map['<end>'] = len(word_map) + 1
    word_map['<pad>'] = 0

    # for densecap feature organisation
    with open(image_paths + 'train_imgs_path.txt', 'r') as f:
        train_paths = [line.rstrip() for line in f.readlines()]
    with open(image_paths + 'val_imgs_path.txt', 'r') as f:
        val_paths = [line.rstrip() for line in f.readlines()]
    with open(image_paths + 'test_imgs_path.txt', 'r') as f:
        test_paths = [line.rstrip() for line in f.readlines()]
        
    #print(image_paths)
    train_feats = h5py.File(image_paths + 'train_multimodal_50.h5', 'r')['feats']
    val_feats = h5py.File(image_paths + 'val_multimodal_50.h5', 'r')['feats']
    test_feats = h5py.File(image_paths + 'test_multimodal_50.h5', 'r')['feats']
    
    train_phrases = h5py.File(image_paths + 'train_multimodal_50.h5', 'r')['scores']
    val_phrases = h5py.File(image_paths + 'val_multimodal_50.h5', 'r')['scores']
    test_phrases = h5py.File(image_paths + 'test_multimodal_50.h5', 'r')['scores']
    
    train_boxes = h5py.File(image_paths + 'train_multimodal_50.h5', 'r')['boxes']
    val_boxes = h5py.File(image_paths + 'val_multimodal_50.h5', 'r')['boxes']
    test_boxes = h5py.File(image_paths + 'test_multimodal_50.h5', 'r')['boxes']
    
    #print(train_boxes.shape)
    
    # Create a base/root name for all output files
    base_filename = dataset + '_' + str(max_sentences) + '_sent_' + str(min_word_freq) + '_min_word'

    # Save word map to a JSON
    with open(os.path.join(output_folder, 'WORDMAP_' + base_filename + '.json'), 'w') as j:
        json.dump(word_map, j)

    #print('Loading DenseCap vocabulary...')
    #word_to_idx = os.path.join('/home/xilini/par-data/densecap-reworked/word_to_idx' + '.json')
    #with open(word_to_idx, 'r') as j:
    #    word_map = json.load(j)

    # Sample captions for each image, save images to HDF5 file, and captions and their lengths to JSON files
    seed(456)
    for impaths, imcaps, split in [(train_image_paths, train_image_captions, 'TRAIN'),
                                   (val_image_paths, val_image_captions, 'VAL'),
                                   (test_image_paths, test_image_captions, 'TEST')]:

        # Open DenseCap captions
        #with open(image_paths + split.lower()+'_captions.json', 'r') as f:
        #    densecap_captions = json.load(f)

        with h5py.File(os.path.join(output_folder, split + '_IMAGES_' + base_filename + '.hdf5'), 'a') as h:
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
            phrases = h.create_dataset('phrases', (len(impaths), 50, 15, 512))
            boxes = h.create_dataset('boxes', (len(impaths), 50, 4))

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
                        img_phrases = train_phrases[this_path_index]
                        img_boxes = train_boxes[this_path_index]
                        assert img_feat.shape == (50, 4096)
                        assert img_phrases.shape == (50, 15, 512)
                        assert img_boxes.shape == (50, 4)
                    elif path in val_paths:
                        this_path_index = val_paths.index(path)
                        img_feat = val_feats[this_path_index]
                        img_phrases = val_phrases[this_path_index]
                        img_boxes = val_boxes[this_path_index]
                        assert img_feat.shape == (50, 4096)
                        assert img_phrases.shape == (50, 15, 512)
                        assert img_boxes.shape == (50, 4)
                    elif path in test_paths:
                        this_path_index = test_paths.index(path)
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
                        # if it is not empty sentence
                        enc_c = [word_map['<start>']] + [word_map.get(word, word_map['<unk>']) for word in cap] + [word_map['<end>']] + [word_map['<pad>']] * (max_words - len(cap))
                        c_len = len(cap) + 2
                    elif cap[0] == '<pad>':
                        # if it is an empty sentence, we add 2 pads (two because of source-target decoding scheme)
                        enc_c = [word_map['<pad>']] * (max_words + 2)
                        c_len = 2
                    enc_captions.append(enc_c)
                    caplens.append(c_len)

            # Sanity check
            assert images.shape[0] * max_sentences == len(enc_captions) == len(caplens)

            # Save encoded captions and their lengths to JSON files
            with open(os.path.join(output_folder, split + '_CAPTIONS_' + base_filename + '.json'), 'w') as j:
                json.dump(enc_captions, j)

            with open(os.path.join(output_folder, split + '_CAPLENS_' + base_filename + '.json'), 'w') as j:
                json.dump(caplens, j)

def caplens_eos(caplens, max_sents):
    
    '''caplens = caplens.squeeze(1)
    actual_caplens = []
    for n in range(caplens.shape[0]):
        these_eos = caplens[n, :].tolist()
        labels_eos = []
        prob_eos = []
        for sen_num, elem in enumerate(these_eos, 1):
            if elem != 2:
                labels_eos.append(0)
            else:
                labels_eos.append(1)
        #print(labels_eos)
        actual_length = len([e for e in labels_eos if e != 1])
        #print(actual_length)
        for p in range(1, actual_length+1):
            prob_eos.append(p/actual_length)
        prob_eos = prob_eos + [1.0 for _ in range(6 - len(prob_eos))]
        actual_caplens = actual_caplens + prob_eos
        #print(prob_eos)
        #print(n)
    actual_caplens = torch.FloatTensor(actual_caplens).unsqueeze(1)
    #print(actual_caplens)
    init_inx = np.array([i for i in range(len(actual_caplens)) if i % max_sents == 0])    
    #print(init_inx)
    #print(actual_caplens)'''
    
    # OLD METHOD, USE FOR BINARY
    caplens_f = torch.flatten(caplens).type(torch.LongTensor)
    #print('interm', caplens_f)
    actual_caplens = torch.zeros(caplens_f.shape[0])
    #print(actual_caplens)
    #print(caplens_f)
    for num, item in enumerate(caplens_f):
        if item != 2:
            actual_caplens[num] = 0
        else:
            actual_caplens[num] = 1
    actual_caplens = actual_caplens.unsqueeze(1)
    #print('actual caplens')
    #print(actual_caplens)
    init_inx = np.array([i for i in range(len(actual_caplens)) if i % max_sents == 0])
    
    return actual_caplens, init_inx



def densecap_to_embeddings(captions, w2i, emb):
    embeddings = torch.zeros(len(captions), 512)
    #print(embeddings, embeddings.shape)
    for n, c in enumerate(captions):
        #print('caption', c)
        intermediate_embeddings = []
        for each_phrase in c:
            #phrase_embeddings
            single_phrase = each_phrase.split()
            single_phrase = [w if w != '<UNK>' else '<unk>' for w in single_phrase]
            single_phrase = [w for w in single_phrase if w in w2i.keys()]
            indices = [w2i[w] for w in single_phrase]
            #print('phrase', single_phrase)
            #print('indices', indices)
            this_phrase_embedding = torch.stack([emb[i] for i in indices])
            #print('phrase embedding', this_phrase_embedding, this_phrase_embedding.shape)
            this_phrase_embedding = torch.mean(this_phrase_embedding, 0, False)
            #print(this_phrase_embedding, this_phrase_embedding.shape)
            intermediate_embeddings.append(this_phrase_embedding)
        #print('interm embeddings', intermediate_embeddings)
        full_embeddings = torch.stack(intermediate_embeddings)
        #print(full_embeddings, full_embeddings.shape)
        full_embeddings = torch.mean(full_embeddings, 0, False)
        embeddings[n, :] = full_embeddings
    #print('embeddings', embeddings, embeddings.shape)
    return embeddings


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

    #torch.save(state, pathname + filename)
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

def plot_loss(this_train, this_val, exp_num, loss_type='full'):
    if loss_type == 'full':
        train_label = 'Train Loss'
        val_label = 'Val Loss'
    if loss_type == 'sentence':
        train_label = 'Sentence Train Loss'
        val_label = 'Sentence Val Loss'
    if loss_type == 'word':
        train_label = 'Word Train Loss'
        val_label = 'Word Val Loss'
    fig = plt.figure(figsize=(10, 8))
    plt.plot(range(1, len(this_train) + 1), this_train, label=train_label)
    plt.plot(range(1, len(this_val) + 1), this_val, label=val_label)
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()
    pathlib.Path(f'./checkpoints/{exp_num}').mkdir(parents=True, exist_ok=True)
    pathname = f'./checkpoints/{exp_num}/'
    fig.savefig(os.path.join(pathname, 'loss_' + loss_type + '.png'),
                bbox_inches='tight')
    plt.close(fig)

    
class HierarchicalXEntropyLoss(nn.Module):
    def __init__(self, weight_word_loss=1.0):
        super(HierarchicalXEntropyLoss, self).__init__()
        self.weight_word = torch.Tensor([weight_word_loss]).to(device)
        self.word_loss = nn.CrossEntropyLoss()

    def forward(self, outputs, targets):
        
        #print('OUTPUTS', outputs, len(outputs))
        #print('TARGETS', targets, len(targets))
        
        #print(outputs, outputs.shape)
        #print(targets, len(targets))

        # Max number of sentences in the target mini-batch:
        max_sentences = len(targets)
        #print(max_sentences)
        self.loss_w = torch.Tensor([0]).to(device)

        for j in range(max_sentences):
            #print(outputs[j], outputs[j].shape)
            #print(targets[j], targets[j].shape)
            self.loss_w += self.word_loss(outputs[j], targets[j])

        return self.weight_word * self.loss_w

    def item_terms(self):
        return self.weight_word, self.loss_w
    
    
def prepare_hierarchical_targets(max_sentences, lengths, captions):
    word_rnn_targets = []
    for j in range(max_sentences):
        if lengths[0, j] == 0:
            break
        non_zero_idxs = lengths[:, j] > 0
        packed = pack_padded_sequence(captions[:, j][non_zero_idxs],
                                      lengths[:, j][non_zero_idxs],
                                      batch_first=True, enforce_sorted=False)[0]
        word_rnn_targets.append(packed)
    return word_rnn_targets

def top_k_top_p_filtering(logits, top_k=0, top_p=0.0, filter_value=-float('Inf')):
    """ Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
    https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317
        Args:
            logits: logits distribution shape (..., vocabulary size)
            top_k >0: keep only top k tokens with highest probability (top-k filtering).
            top_p >0.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
    """
    top_k = min(top_k, logits.size(-1))  # Safety check
    
    if top_k > 0:
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p > 0.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs >= top_p
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        indices_to_remove = sorted_indices_to_remove.scatter(dim=-1, index=sorted_indices, src=sorted_indices_to_remove)
        logits[indices_to_remove] = filter_value
        
    return logits



def plot_attention(image, result, attention_plot):
    temp_image = np.array(Image.open(image))
    fig = plt.figure(figsize=(20, 20))
    len_result = len(result)
    #print(len_result)
    #print(temp_image)
    for l in range(len_result):
        temp_att = np.resize(attention_plot[l], (7, 7))
        #temp_att = attention_plot[l]
        ax = fig.add_subplot(len_result, len_result, l+1)
        #print(ax)
        ax.set_title(result[l])
        img = ax.imshow(temp_image)
        ax.imshow(temp_att, cmap='gray', alpha=0.6, extent=img.get_extent())
    plt.tight_layout()
    plt.show()
    fig.savefig('./foo.png', bbox_inches='tight')


def visualize_pred(im_path, res, boxes, att_weights):
    im = cv2.imread(im_path)
    im = cv2.cvtColor(im, cv2.COLOR_RGB2RGBA)
    b,g,r,a = cv2.split(im)           # get b, g, r
    im = cv2.merge([r,g,b,a])
    M = min(len(boxes), len(att_weights))
    boxes = boxes.squeeze(0)
    #print(boxes, boxes.shape)
    #print(att_weights, att_weights.shape)
    img = str(im_path.split('/')[-1].split('.jpg')[0])
    len_result = len(res)
    for l in range(len_result):
        #print('boxes in', boxes, boxes.shape)
        #print('att in', att_weights[l], att_weights[l].shape)
        im_ocr_att = attention_bbox_interpolation(im, boxes, att_weights[l])
        #print(im_ocr_att, type(im_ocr_att))
        #print(plt.imshow(im_ocr_att))
        imageio.imwrite(f'./maps/{img}_{l}.png', im_ocr_att) 

def attention_bbox_interpolation(im, bboxes, att):
    softmax = att
    assert len(softmax) == len(bboxes)
    img_h, img_w = im.shape[:2]
    opacity = np.zeros((img_h, img_w), np.float32)
    for bbox, weight in zip(bboxes, softmax):
        x1, y1, x2, y2 = bbox
        opacity[int(y1):int(y2), int(x1):int(x2)] += weight
    opacity = np.minimum(opacity, 1)
    opacity = opacity[..., np.newaxis]
    vis_im = np.array(Image.fromarray(cmap(opacity, bytes=True), 'RGBA'))
    vis_im = vis_im.astype(im.dtype)
    vis_im = cv2.addWeighted(im, 0.4, vis_im, 0.6, 0.0)
    vis_im = vis_im.astype(im.dtype)
    return vis_im


class BeamSearchNode(object):
    def __init__(self, hiddenstate, previousNode, wordId, logProb, length):
        self.h = hiddenstate
        self.prevNode = previousNode
        self.wordid = wordId
        self.logp = logProb
        self.leng = length
        self.alpha_penalty = 0.9
        
    def __lt__(self, other):
        assert isinstance(other, BeamSearchNode)
        return self.logp < other.logp
    
    def eval(self, alpha=1.0):
        length_penalty = ((5.0 + self.leng) ** self.alpha_penalty) / (6.0 ** self.alpha_penalty)
        reward = 0
        # length normalisation
        #return self.logp / float(self.leng - 1 + 1e-6) + alpha * reward
        return self.logp / float(length_penalty)
    
def block_ngram_repeats(cur_len, log_probs, curr_sequence):
    block_ngram_repeat = 2
    if cur_len > 1:
        curr_seqs = curr_sequence.queue
        alive_sequences = []
        for score, n in sorted(curr_seqs, key=operator.itemgetter(0)):
            utterance = []
            utterance.append(n.wordid.item())
            # back trace
            while n.prevNode != None:
                n = n.prevNode
                utterance.append(n.wordid.item())
            alive_sequences.append(utterance[::-1])
        alive_sequences = alive_sequences[-3:]
        #print('alive seqs', alive_sequences)
        for path_idx in range(len(alive_sequences)):
            hyp = alive_sequences[path_idx]
            ngrams = set()
            fail = False
            gram = []
            for i in range(len(hyp) - 1):
                # Last n tokens, n = block_ngram_repeat
                gram = (gram + [hyp[i]])[-block_ngram_repeat:]
                
                # skip the blocking if any token in gram is excluded
                #if set(gram) & self.exclusion_tokens:
                #    continue
                
                if tuple(gram) in ngrams:
                    #print('fail')
                    fail = True
                ngrams.add(tuple(gram))
            if fail:
                log_probs[0][path_idx] = -10e20
    return log_probs


def ensure_min_length(leng, log_probs, eos_token):
    min_length = 9 + 2
    if leng <= min_length:
        log_probs[:, eos_token] = -1e20
    return log_probs

    
def beam_decode(target_tensor, embs, word_to_idx, word_decoder, decoder_hiddens, beam_width, topk):

    IMAGE_TOKEN = word_to_idx['<sentence_topic>']
    EOS_TOKEN = word_to_idx['<end>']

    decoded_batch = []
    
    # decoding goes sentence by sentence
    for idx in range(target_tensor.size(0)):
        if isinstance(decoder_hiddens, tuple):  # LSTM case
            decoder_hidden = (decoder_hiddens[0][:,idx, :].unsqueeze(0),decoder_hiddens[1][:,idx, :].unsqueeze(0))
        else:
            decoder_hidden = decoder_hiddens[:, idx, :].unsqueeze(0)

        # Start with the image encoding
        decoder_input = torch.LongTensor([[IMAGE_TOKEN]]).to(device)

        # Number of sentence to generate
        endnodes = []
        number_required = min((topk + 1), topk - len(endnodes))

        # starting node -  hidden vector, previous node, word id, logp, length
        node = BeamSearchNode(decoder_hidden, None, decoder_input, 0, 1)
        nodes = PriorityQueue()

        # start the queue
        nodes.put((-node.eval(), node))
        qsize = 1

        # start beam search
        while True:
            # give up when decoding takes too long
            if qsize > 50000: break
                            
            # fetch the best node
            score, n = nodes.get()
            decoder_input = n.wordid
            decoder_hidden = n.h
            
            if n.wordid.item() == EOS_TOKEN and n.prevNode != None:
                endnodes.append((score, n))
                # if we reached maximum # of sentences required
                if len(endnodes) >= number_required:
                    break
                else:
                    continue

            decoder_input = embs[decoder_input]
            decoder_output, decoder_hidden = word_decoder.decode_step(decoder_input, decoder_hidden)
            decoder_output = word_decoder.linear(decoder_output.squeeze(1))
            
            
            #decoder_output = decoder_output.div(0.5)
            decoder_output = F.log_softmax(decoder_output, dim=-1)
            
            #print('len', n.leng)
            #print('prob before', decoder_output[:, EOS_TOKEN])
            
            # MIN LENGTH
            decoder_output = ensure_min_length(n.leng, decoder_output, EOS_TOKEN)
            
            #print('prob after', decoder_output[:, EOS_TOKEN])
            #print()
            
            # PUT HERE REAL BEAM SEARCH OF TOP
            log_prob, indexes = torch.topk(decoder_output, beam_width)
            
            #log_prob = block_ngram_repeats(n.leng, log_prob, nodes)
            
            nextnodes = []
            
            for new_k in range(beam_width):
                
                length_penalty = ((5.0 + n.leng) ** n.alpha_penalty) / (6.0 ** n.alpha_penalty)
                                
                decoded_t = indexes[0][new_k].view(1, -1)
                log_p = log_prob[0][new_k].item()
                
                #new_log = (n.logp + log_p) / (n.leng ** 0.1)
                new_log = n.logp + log_p
                #new_log = (n.logp + log_p) / length_penalty
                
                node = BeamSearchNode(decoder_hidden, n, decoded_t, new_log, n.leng + 1)
                score = -node.eval()
                nextnodes.append((score, node))
                
            # put them into queue
            for i in range(len(nextnodes)):
                score, nn = nextnodes[i]
                nodes.put((score, nn))
                # increase qsize
            qsize += len(nextnodes) - 1

        # choose nbest paths, back trace them
        if len(endnodes) == 0:
            endnodes = [nodes.get() for _ in range(topk)]

        utterances = []
        for score, n in sorted(endnodes, key=operator.itemgetter(0)):
            utterance = []
            utterance.append(n.wordid.item())
            # back trace
            while n.prevNode != None:
                n = n.prevNode
                utterance.append(n.wordid.item())
            
            utterance = utterance[::-1]
            utterances.append(utterance)
            #print(score)
        
        decoded_batch.append(utterances)
        
    return decoded_batch, decoder_hidden


'''############
SOME BEAM IMPLEMENTATIONS HERE
############'''


'''embs = word_decoder.embeddings.weight.data
    embs = torch.cat([embs, inputs.squeeze(1)], dim=0)
    word_to_idx['<topic>'] = 7606
    prev_beam = list()
    prev_beam.append((0.0, [word_to_idx['<topic>']]))
    prefix_len = 1
    beam_width = 2
    all_sam_cap = np.ndarray((1, 50))

    while (prefix_len < args.max_words - 1):

        curr_beam = list()
        prefix_batch = list()
        prob_batch = list()

        for (prefix_prob, prefix) in prev_beam:
            if prefix[-1] == word_to_idx['<end>']:
                heapq.heappush(curr_beam, (prefix_prob, prefix))
            else:
                prefix_batch.append(prefix + [word_to_idx['<pad>']] * (args.max_words - len(prefix)))
                prob_batch.append(prefix_prob)
                prefix_len = len(prefix)

            decoder_input = embs[prefix].unsqueeze(0)
            hiddens, (h_word, c_word) = word_decoder.decode_step(decoder_input, (h_word, c_word))
            outputs = word_decoder.linear(hiddens)
            #outputs = outputs.div(args.temperature_value)
            all_distributions = F.log_softmax(outputs, dim=-1)
            sorted_distributions, indexes_distributions = torch.sort(all_distributions)


            for (prefix_prob1, prefix_arr, indexes_distribution) in zip(prob_batch, prefix_batch, indexes_distributions):
                best_words = indexes_distribution[-1, -beam_width:]
                best_probs = all_distributions[:, -1, -beam_width:]

                print('WORKING WITH', prefix_arr)
                #prefix = list(prefix_arr[:prefix_len])
                print('OBSERVED', list(prefix_arr[:prefix_len]))

                print('best words', best_words)
                print('best probs', best_probs)

                #alpha_penalty = 0.7
                #length_penalty = ((5.0 + prefix_len) ** alpha_penalty) / (6.0 ** alpha_penalty)

                for (next_index, next_prob) in zip(best_words.squeeze(), best_probs.squeeze()):

                    print('next index', next_index)
                    print('new prefix', prefix + [next_index.item()])
                    print(prefix_prob, next_prob)
                    print('new prob', prefix_prob + next_prob)

                    #curr_new_prob = (prefix_prob + next_prob) / length_penalty
                    #curr_new_prob = (prefix_prob + next_prob) / (prefix_len ** 0.4)
                    curr_new_prob = prefix_prob + next_prob

                    heapq.heappush(curr_beam, (curr_new_prob, prefix + [next_index.item()]))

                    if len(curr_beam) > beam_width:
                        print('POPPING')
                        print(curr_beam)
                        heapq.heappop(curr_beam)
                        print(curr_beam)

        print()

        print('curr beam', curr_beam)
        prev_beam = curr_beam
        print('new prev beam', prev_beam)
        best_prob, best_current = max(curr_beam)
        print('best prob and best current in curr beam', best_prob, best_current)
        if best_current[-1] == word_to_idx['<end>']:
            break

    print('LAST BEAM?', prev_beam)
    (_, best_generated) = max(prev_beam)
    all_sam_cap = best_generated

#h_word = torch.zeros(args.num_layers_wordrnn, imgs.shape[0], args.hidden_size).to(device)
#c_word = torch.zeros(args.num_layers_wordrnn, imgs.shape[0], args.hidden_size).to(device)

for item in all_sam_cap:
    this_gen_sentence.append(item)'''
    
    
    
    
'''#inputs = inputs.squeeze(1)
    #embs = word_decoder.embeddings.weight.data
    #embs = torch.cat([embs, inputs], dim=0)
    #word_to_idx['image'] = 7606
    #start = [word_to_idx['image']]

    start_word = [[[8888], 0.0]]
    final_caption = []
    alpha_penalty = 0.5

    #print('curr start word', start_word)
    #print(len(start_word[0]))

    while len(start_word[0][0]) < args.max_words:

        #length_penalty = ((5.0 + len(start_word[-1][0])) ** alpha_penalty) / (6.0 ** alpha_penalty)
        #print('length_penalty', length_penalty)
        temp = []

        #print('START WORD', start_word)
        for s_num, s in enumerate(start_word):
            #print(inputs.shape)
            #inputs = embs[torch.LongTensor([s[0][-1]])].unsqueeze(0)
            hiddens, (h_word, c_word) = word_decoder.decode_step(inputs, (h_word, c_word))
            outputs = word_decoder.linear(hiddens.squeeze(1))
            probs = F.softmax(outputs, dim=-1)
            sorted_scores, idxs = torch.sort(probs[0])
            probs_pred = sorted_scores[-args.beam:]
            words_pred = idxs[-args.beam:]
            print('word preds', probs_pred)
            print('idxs', words_pred)

            for n, w in enumerate(words_pred):

                curr_cap, curr_prob = s[0][:], s[1]
                print('curr cap', curr_cap)
                print('curr prob', curr_prob)

                next_word = w.item()
                next_prob = probs_pred[n]
                print('next word', next_word)
                print('next prob', next_prob)

                new_cap = curr_cap
                new_cap.append(next_word)
                new_prob = curr_prob + next_prob
                #new_prob = (curr_prob + torch.log(next_prob)) / length_penalty
                print('new cap', curr_cap)
                print('new prob', new_prob)

                #new_prob = (next_prob + np.log(prob)) / length_penalty
                temp.append([new_cap, new_prob])

        start_word = temp
        start_word = sorted(start_word, reverse=False, key=lambda l: l[0][1])
        start_word = start_word[-args.beam:]
        if start_word[-1][0][-1] == [word_to_idx['<end>']][0]:
            break

    start_word = start_word[-1][0]
    #print('start_word', start_word)
    intermediate_caption = [i for i in start_word[1:]]

    #for i in intermediate_caption:
    #    if i != '<end>':
    #        final_caption.append(i)
    #    else:
    #        break
    for i in intermediate_caption:
        this_gen_sentence.append(i)

    h_word = torch.zeros(args.num_layers_wordrnn, imgs.shape[0], args.hidden_size).to(device)
    c_word = torch.zeros(args.num_layers_wordrnn, imgs.shape[0], args.hidden_size).to(device)'''
