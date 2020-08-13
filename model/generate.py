'''
generation script
'''

# do not forget to set CUDA_VISIBLE_DEVICES to any of the gpus you would like to use (!!!)

import argparse
from configparser import ConfigParser
import time
from itertools import islice
import ast

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import skimage
import cv2
import imageio

import torch.backends.cudnn as cudnn
import torch.optim
from torch.utils import data
import torchvision.transforms as transforms
from torch import nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from models import RegPool, SentenceRNN, WordRNN
from datasets import *
from utils import *
from evalfunc.bleu.bleu import Bleu
from evalfunc.rouge.rouge import Rouge
from evalfunc.cider.cider import Cider
from evalfunc.meteor.meteor import Meteor
from evalfunc.spice.spice import Spice
from evalfunc.wmd.wmd import WMD

import operator
from queue import PriorityQueue

import numpy as np
from PIL import Image
import heapq

device = torch.device("cuda")
cudnn.benchmark = True

# nipy_spectral, gist_rainbow, seismic, hot
cmap = matplotlib.cm.get_cmap('nipy_spectral')
#cmap.set_bad(color="k", alpha=5.0)


def main(args):
    """
    Generation
    """

    word_to_idx = os.path.join(args.data_folder, 'wordmap_' + args.data_name + '.json')
    with open(word_to_idx, 'r') as j:
        word_to_idx = json.load(j)
    idx_to_word = {v: k for k, v in word_to_idx.items()}
    vocab_size = len(word_to_idx)

    args.vocab_size = vocab_size
        
    encoder = RegPool(args)
    encoder_optimizer = None
        
    val_loader = data.DataLoader(
        ParagraphDataset(args.data_folder, args.data_name, 'TEST'),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    if args.set_size != 0:
        val_iterator = iter(val_loader)
        val_loader = list(islice(val_iterator, args.set_size))
    elif args.set_size == 0:
        val_loader = val_loader

    sentence_decoder = SentenceRNN(args)
    word_decoder = WordRNN(args)

    # Load the trained model parameters
    this_checkpoint = torch.load(args.model_path + args.model_gen)
    encoder.load_state_dict(this_checkpoint['encoder'].state_dict())
    sentence_decoder.load_state_dict(this_checkpoint['sentence_decoder'].state_dict())
    word_decoder.load_state_dict(this_checkpoint['word_decoder'].state_dict())

    # Move to GPU
    sentence_decoder = sentence_decoder.to(device)
    word_decoder = word_decoder.to(device)
    encoder = encoder.to(device)

    train_step = 0
    val_step = 0

    # generation loop
    b1, b2, b3, b4, c, m, s, wmd = generate(val_loader=val_loader,
                                    encoder=encoder,
                                    sentence_decoder=sentence_decoder,
                                    word_decoder=word_decoder,
                                    logger=None,
                                    vocab_size=vocab_size,
                                    word_to_idx=word_to_idx,
                                    idx_to_word=idx_to_word,
                                    args=args)
    print('BLEU 1', b1)
    print('BLEU 2', b2)
    print('BLEU 3', b3)
    print('BLEU 4', b4)
    print('CIDER', c)
    print('METEOR', m)
    print('WMD', wmd)
    #print('SPICE', s)
    
    # POINT
    res_fname = str(args.model_gen).split('/')[0]
    with open(f'./scores/res-{res_fname}-{args.decoding_strategy}_{args.beam}_minlength9.json', 'w') as j:
        j.write(f'BLEU 1 \t {b1} \n')
        j.write(f'BLEU 2 \t {b2} \n')
        j.write(f'BLEU 3 \t {b3} \n')
        j.write(f'BLEU 4 \t {b4} \n')
        j.write(f'CIDEr \t {c} \n')
        j.write(f'METEOR \t {m} \n')

    
def generate(val_loader,
             encoder,
             sentence_decoder,
             word_decoder,
             logger,
             vocab_size,
             word_to_idx,
             idx_to_word,
             args):

    sentence_decoder.eval()
    word_decoder.eval()
    if encoder is not None:
        encoder.eval()

    paragraphs_generated = []
    
    print(sentence_decoder)
    print(word_decoder)
    
    Bleu_1 = 0
    Bleu_2 = 0
    Bleu_3 = 0
    Bleu_4 = 0
    CIDEr = 0
    METEOR = 0
    SPICE = 0
    wmd = 0
        
    scorers = [

        (Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
        (Cider('corpus'), "CIDEr"),
        #vg-test-words
        (Meteor(), "METEOR"),
        #(WMD(), 'WMD')
    ]
    
    #for the purpose of making sure generated texts make sense
    references_batch = dict()
    hypotheses_batch = dict()
    
    with torch.no_grad():
        for image_num, (image_features,
                        image_ids,
                        caps,
                        caplens,
                        phrase_scores,
                        bboxes, phrase_lengths) in enumerate(val_loader):
            
            if image_num % 100 == 0:
                print(image_num)
            
            image_features = image_features.to(device)
            image_ids = image_ids.to(device)
            caps = caps.to(device)
            caplens = caplens.squeeze(1).to(device)
            phrase_scores = phrase_scores.to(device)
            phrase_lengths = phrase_lengths.to(device)
            args.batch_size = image_features.shape[0]

            loss = 0
            word_loss = 0
            
            h_sent = c_sent = h_word = c_word = torch.zeros(1, args.batch_size, args.hidden_size)
            
            h_word = h_word.to(device)
            c_word = c_word.to(device)
            h_sent = h_sent.to(device)
            c_sent = c_sent.to(device)
            
            # initial sentence topic is zero topic
            init_topic = torch.zeros(1, args.batch_size, args.hidden_size)
            
            generated = []
            actual = []
            
            language, vision = encoder(image_features, phrase_scores, phrase_lengths)
            topics = torch.zeros(args.batch_size, args.max_sentences, 512).to(device)
            
            if not args.use_attention:
                
                language = language.unsqueeze(1).expand(-1, args.max_sentences, -1, -1)
                vision = vision.unsqueeze(1).expand(-1, args.max_sentences, -1, -1)
                
                # max-pooling instead of attention
                if args.use_vision and not args.use_language:
                    sentence_input_vector = torch.max(vision, dim=1).values.unsqueeze(1)
                elif args.use_language and not args.use_vision:
                    sentence_inp
                    ut_vector = torch.max(language, dim=1).values.unsqueeze(1)
                elif args.use_language and args.use_vision:
                    vision_input = torch.max(vision, dim=2).values
                    language_input = torch.max(language, dim=2).values
                    sentence_input_vector = torch.cat((language_input, vision_input), dim=2)
                topics, (_, _) = sentence_decoder.sentence_rnn(sentence_input_vector)
                    
            if args.use_attention:
              
                attention_plot = np.zeros((6, 50))
                att_plot_sentence_labels = []
                
                language = language.to(device)
                vision = vision.to(device)
                
                hidden_previous = h_sent.squeeze(0).to(device)
                cell_previous = c_sent.squeeze(0).to(device) # batch_size x hidden_size
                topic_previous = init_topic.squeeze(0).to(device) # concat dimension (batch_size) x hidden_size
                
                alphas = torch.zeros(args.batch_size, args.max_sentences, args.num_boxes).to(device)
                                
                for step in range(args.max_sentences):
                    #print(step)
                    
                    if caplens.squeeze()[step].item() != 0:
                        
                        hidden_map = hidden_previous.unsqueeze(1)
                        hidden_map = sentence_decoder.attention.hidden_mapping(hidden_map)
                        
                        if args.use_vision and args.use_vision:
                        
                            text_att = sentence_decoder.attention.language_full_att(sentence_decoder.attention.tanh(hidden_map + language))
                            vis_att = sentence_decoder.attention.vision_full_att(sentence_decoder.attention.tanh(hidden_map + vision))
                            alpha_text = sentence_decoder.attention.softmax(text_att)
                            alpha_vision = sentence_decoder.attention.softmax(vis_att)
                            text_weighted_encoding = (language * alpha_text).sum(dim=1)
                            vision_weighted_encoding = (vision * alpha_vision).sum(dim=1)
                            multimodal_concat = torch.cat((text_weighted_encoding, vision_weighted_encoding), dim=1)

                        #print('alpha_text', alpha_text, alpha_text.shape)
                        #print('alpha vision', alpha_vision, alpha_vision.shape)
                                                
                        gate = sentence_decoder.sigmoid(sentence_decoder.f_beta(hidden_previous))

                        context = gate * multimodal_concat

                        hidden_previous, cell_previous = sentence_decoder.sentence_rnn(context,
                                                                                       (hidden_previous, cell_previous))
                        
                        topics[:, step, :] = hidden_previous


            for t in range(args.max_sentences):
                
                topic = topics[:, t]

                # generation
                start_token = torch.LongTensor([word_to_idx['<start>']]).to(device)
                end_token = torch.LongTensor([word_to_idx['<end>']]).to(device)
                topic = topic.unsqueeze(1)
                                
                this_gen_sentence = []
                temp = []
                final_caption = []
                
                if caplens.squeeze()[t].item() != 0:
                                        
                    if args.decoding_strategy == 'greedy':
                        decoder_input = topic
                        for i in range(args.max_words - 1):
                            hiddens, (h_word, c_word) = word_decoder.decode_step(decoder_input, (h_word, c_word))
                            outputs = word_decoder.linear(hiddens.squeeze(1))
                            outputs = F.log_softmax(outputs, dim=-1)
                            topv, topi = outputs.data.topk(1)
                            topi = topi.view(-1)
                            this_gen_sentence.append(topi.item())
                            decoder_input = word_decoder.embeddings(topi).unsqueeze(1)
                                                                                    
                    if args.decoding_strategy == 'sampling':
                        decoder_input = topic
                        for i in range(args.max_words):
                            hiddens, (h_word, c_word) = word_decoder.decode_step(decoder_input, (h_word, c_word))
                            outputs = word_decoder.linear(hiddens.squeeze(1))
                            if args.temperature:
                                outputs = outputs.div(args.temperature_value)
                            word_softmax = F.softmax(outputs, dim=-1)
                            word_idx = torch.multinomial(word_softmax, 1)[0].to(device)
                            this_gen_sentence.append(word_idx.item())
                            decoder_input = word_decoder.embeddings(word_idx).unsqueeze(1)
                        
                    if args.decoding_strategy == 'topn_sampling':
                        decoder_input = topic
                        for i in range(args.max_words):
                            hiddens, (h_word, c_word) = word_decoder.decode_step(decoder_input, (h_word, c_word))
                            outputs = word_decoder.linear(hiddens.squeeze(1))
                            if args.temperature:
                                outputs = outputs.div(args.temperature_value)
                            # top_n = 1 is greedy search, top_n = vocab_size is pure sampling
                            # use either nucleus, either top-k
                            # you can use both, just change top_k in the first loop from 0.0 to args.top_n
                            if args.nucleus_sampling:
                                logits = top_k_top_p_filtering(outputs, top_k=0.0, top_p=args.top_p)
                            if not args.nucleus_sampling:
                                logits = top_k_top_p_filtering(outputs, top_k=args.top_n, top_p=0.0)
                            word_softmax = F.softmax(logits, dim=-1)
                            word_idx = torch.multinomial(word_softmax, 1)[0].to(device)
                            this_gen_sentence.append(word_idx.item())
                            decoder_input = word_decoder.embeddings(word_idx).unsqueeze(1)
                        
                    if args.decoding_strategy == 'beam':
                        embs = word_decoder.embeddings.weight.data
                        embs = torch.cat([embs, topic.squeeze(1)], dim=0)
                        word_to_idx['<sentence_topic>'] = 7606
                        target_tensor = torch.zeros(1, 50)
                        x, (h_word, c_word) = beam_decode(target_tensor, embs, word_to_idx, word_decoder, 
                                                          (h_word, c_word), args.beam, 10)
                        #print('curr len', len(x[0]))
                        # question: how to sample from topk sentences ???
                        # the most probable, or t-th most probable ??? t-th most probable seems to introduce more diversity ?
                        # sampling t-th most probable motivation: makes paragraphs more diverse
                        # also, input might not be enough
                        # also, the most probable sentence is supposed to alwas describe scene as a whole
                        # other sentences will most likely describe scene details, and that's what we want to be in the t+1 sentences
                        #print('poss sentence', t)
                        #for item in x[0]:
                        #    print([idx_to_word[int(w)] for w in item[1:]])
                        for item in x[0][-1]:
                            this_gen_sentence.append(item)
                        #print()
                            
                    generated.append(this_gen_sentence)
                    actual.append(caps.squeeze(0)[t])

            image_id = image_ids.item()
            # Get references
            for single_paragraph in range(len(actual)):
                if single_paragraph == 0:
                    image_id = image_ids[single_paragraph].item()
                    references_batch[image_id] = []
                img_caps = actual[single_paragraph].tolist()
                img_captions = []
                for i in img_caps:
                    if i not in {word_to_idx['<start>'], word_to_idx['<pad>'], word_to_idx['<end>']}:
                        img_captions.append(i)                
                paragraph_text = []
                this_sent_text = [idx_to_word[w] for w in img_captions]
                this_sent_text[-2:] = [''.join(this_sent_text[-2:])]
                paragraph_text.append(' '.join(this_sent_text))
                for this_sent_text in paragraph_text:
                    if ' '.join(this_sent_text) != '':
                        references_batch[image_id].append(' '.join(paragraph_text))
                # each reference is a single hypothesis
                references_batch[image_id] = [' '.join(references_batch[image_id])]
                        
            # Get hypotheses
            for pred_sentence in range(len(generated)):
                if pred_sentence == 0:
                    hypotheses_batch[image_id] = []
                this_sent_preds = generated[pred_sentence]
                if args.decoding_strategy == 'beam':
                    this_sentence_text = []
                    for w in this_sent_preds[2:]:
                        if idx_to_word[w] != '<end>':
                            this_sentence_text.append(idx_to_word[w])
                        else:
                            break
                else:
                    this_sentence_text = [idx_to_word[w] for w in this_sent_preds[1:][:-1]]
                this_sentence_text = [' '.join(this_sentence_text)]
                this_sentence_text = [this_sentence_text[0].replace(' .', '.')]
                hypotheses_batch[image_id].append(''.join(this_sentence_text))
            for img, par in hypotheses_batch.items():
                new_par = ' '.join(par)
                hypotheses_batch[img] = [new_par]
                
            # ATTENTION PLOTS
            # NEEDS TO BE ADAPTED FOR MULTIPLE FILES
            #if args.use_attention:
            #    this_image_id = list(references_batch.keys())[0]
            #    attention_plot = attention_plot[:len(att_plot_sentence_labels), :]
            #    image_path = show_image(this_image_id)
            #    #print(image_path)
            #    #plot_attention(image_path, res, attention_plot)
            #    visualize_pred(image_path, att_plot_sentence_labels, bboxes, attention_plot)
            

    assert len(references_batch.keys()) == len(hypotheses_batch.keys())

    score = []
    method = []
    for scorer, method_i in scorers:

        score_i, _ = scorer.compute_score(references_batch, hypotheses_batch)
        score.extend(score_i) if isinstance(score_i, list) else score.append(score_i)
        method.extend(method_i) if isinstance(method_i, list) else method.append(method_i)

    score_dict = dict(zip(method, score))

    Bleu_1 += score_dict['Bleu_1']
    Bleu_2 += score_dict['Bleu_2']
    Bleu_3 += score_dict['Bleu_3']
    Bleu_4 += score_dict['Bleu_4']
    CIDEr += score_dict['CIDEr']
    METEOR += score_dict['METEOR']
    #wmd += score_dict['WMD']
    #SPICE += score_dict['SPICE']

    #print(references_batch)
    for item in references_batch:
        pars = {}
        pars['image_id'] = item
        pars['references'] = ' '.join(references_batch[item])
        pars['hypotheses'] = ' '.join(hypotheses_batch[item])
        paragraphs_generated.append(pars)
    
    # POINT
    res_fname = str(args.model_gen).split('/')[0]
    with open(f'./scores/{res_fname}-{args.decoding_strategy}_{args.beam}_minlength9.json', 'w') as f:
        json.dump(paragraphs_generated, f)
        
    return Bleu_1, Bleu_2, Bleu_3, Bleu_4, CIDEr, METEOR, SPICE, wmd

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    config_parser = ConfigParser()
    config_parser.read('../config.ini')

    # Get data paths
    densecap_path = config_parser.get('BASEPATH', 'densecap_path')
    out_path = config_parser.get('BASEPATH', 'data_folder')
    data_name = config_parser.get('BASEPATH', 'data_name')
    checkpoint_path = config_parser.get('BASEPATH', 'models_save')
    data_folder_densecap = config_parser.get('BASEPATH', 'data_folder_densecap')

    # Get model hyperparameters
    hidden_size = config_parser.get('PARAMS-SENTENCE', 'hidden_size')
    num_layers_sentencernn = config_parser.get('PARAMS-SENTENCE', 'num_layers_sentencernn')
    densecap_feat_dim = config_parser.get('PARAMS-SENTENCE', 'densecap_feat_dim')
    sentence_lr = config_parser.get('PARAMS-SENTENCE', 'sentence_lr')

    num_layers_wordrnn = config_parser.get('PARAMS-WORD', 'num_layers_wordrnn')
    max_sentences = config_parser.get('PARAMS-WORD', 'max_sentences')
    max_words = config_parser.get('PARAMS-WORD', 'max_words')
    word_decoder_lr = config_parser.get('PARAMS-WORD', 'word_decoder_lr')
    lambda_word = config_parser.get('PARAMS-WORD', 'lambda_word')

    num_boxes = config_parser.get('PARAMS-MODELS', 'num_boxes')
    embeddings_pretrained = config_parser.getboolean('PARAMS-MODELS', 'embeddings_pretrained')
    start_epoch = config_parser.get('PARAMS-MODELS', 'start_epoch')
    num_epochs = config_parser.get('PARAMS-MODELS', 'num_epochs')
    workers = config_parser.get('PARAMS-MODELS', 'workers')
    print_freq = config_parser.get('PARAMS-MODELS', 'print_freq')
    best_val_loss = config_parser.get('EVAL-SCORES', 'best_val_loss')
    exp_num = config_parser.get('PARAMS-MODELS', 'exp_num')
    word_weight_decay = config_parser.get('PARAMS-WORD', 'word_weight_decay')

    use_attention = config_parser.getboolean('PARAMS-MODELS', 'use_attention')
    use_vision = config_parser.getboolean('PARAMS-MODELS', 'use_vision')
    use_language = config_parser.getboolean('PARAMS-MODELS', 'use_language')
    use_multimodal = config_parser.getboolean('PARAMS-MODELS', 'use_multimodal')
    
    set_size = config_parser.get('PARAMS-MODELS', 'set_size')
    model_gen = config_parser.get('PARAMS-MODELS', 'model_gen')
    decoding_strategy = config_parser.get('PARAMS-MODELS', 'decoding_strategy')
    beam = config_parser.get('PARAMS-MODELS', 'beam')

    api_key = config_parser.get('COMET', 'api_key')
    project_name = config_parser.get('COMET', 'project_name')
    workspace = config_parser.get('COMET', 'workspace')

    # Paths
    parser.add_argument('--data_folder', type=str, default=out_path, help='path to all input data')
    parser.add_argument('--data_name', type=str, default=data_name, help='shared name among target files')
    parser.add_argument('--model_path', type=str, default=checkpoint_path, help='path for keeping model"s checkpoints')
    parser.add_argument('--densecap_path', type=str, default=densecap_path, help='path to the pretrained DenseCap materials')
    parser.add_argument('--data_folder_densecap', type=str, default=data_folder_densecap, help='path to the data when using DenseCap encoder')

    # Model parameters
    parser.add_argument('--num_boxes', type=int, default=num_boxes, help='number of regions to look at in the image')
    parser.add_argument('--max_sentences', type=int, default=max_sentences, help='maximum number of sentences per paragraph')
    parser.add_argument('--max_words', type=int, default=max_words, help='maximum number of words per sentence')
    parser.add_argument('--num_layers_sentencernn', type=int, default=num_layers_sentencernn, help='number of layers in sentenceRNN')
    parser.add_argument('--sentence_lr', type=float, default=sentence_lr, help='learning rate for sentenceRNN')
    parser.add_argument('--word_lr', type=float, default=word_decoder_lr, help='learning rate for wordRNN')
    parser.add_argument('--hidden_size', type=int, default=hidden_size, help='size of hidden dimension')
    parser.add_argument('--densecap_feat_dim', type=int, default=densecap_feat_dim, help='dimension of projected features when using DenseCap encoder')
    parser.add_argument('--lambda_word', type=float, default=lambda_word, help='word normalisation parameter')

    parser.add_argument('--batch_size', type=int, default=1, help='batch size')
    parser.add_argument('--start_epoch', type=int, default=start_epoch, help='start from epoch number')
    parser.add_argument('--num_epochs', type=int, default=num_epochs, help='number of epochs')
    parser.add_argument('--workers', type=int, default=workers, help='number of workers')
    parser.add_argument('--print_freq', type=int, default=print_freq, help='print information after X batches')
    parser.add_argument('--exp_num', type=str, default=exp_num, help='track number of experiments and various model settings, used to create directories for corresponding experiments within ./models')
    parser.add_argument('--best_val_loss', type=float, default=best_val_loss, help='initial best validation loss')
    parser.add_argument('--word_weight_decay', type=float, default=word_weight_decay, help='weight decay for the word LSTM')
    
    parser.add_argument('--use_attention', type=bool, default=use_attention, help='use attention')
    parser.add_argument('--use_vision', type=bool, default=use_vision, help='use vision only')
    parser.add_argument('--use_language', type=bool, default=use_language, help='use language only')
    parser.add_argument('--use_multimodal', type=bool, default=use_multimodal, help='fuse two modalities into a single vector')
    
    parser.add_argument('--set_size', type=int, default=set_size, help='generate paragraphs for this number of images')
    parser.add_argument('--model_gen', type=str, default=model_gen, help='model to use for generation')
    parser.add_argument('--decoding_strategy', type=str, default=decoding_strategy, help='decoding strategy')
    parser.add_argument('--beam', type=int, default=beam, help='beam size')

    parser.add_argument('--api_key', type=str, default=api_key, help='key for the Comet logger')
    parser.add_argument('--project_name', type=str, default=project_name, help='name of the project')
    parser.add_argument('--workspace', type=str, default=workspace, help='owner of the Comet workspace')

    arguments = parser.parse_args()
    main(arguments)