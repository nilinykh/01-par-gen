'''
generation script
'''

# do not forget to set CUDA_VISIBLE_DEVICES to any of the gpus you would like to use (!!!)

import argparse
from configparser import ConfigParser
import time
from itertools import islice
import ast

import torch.backends.cudnn as cudnn
import torch.optim
from torch.utils import data
import torchvision.transforms as transforms
from torch import nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from models import Encoder, RegPool, SentenceRNN, WordRNN
from datasets import *
from utils import *
from evalfunc.bleu.bleu import Bleu
from evalfunc.rouge.rouge import Rouge
from evalfunc.cider.cider import Cider
from evalfunc.meteor.meteor import Meteor

import numpy as np

device = torch.device("cuda")
cudnn.benchmark = True


def main(args):
    """
    Generation
    """

    if args.with_densecap_captions:
        print('Loading DenseCap embeddings...')
        word_to_idx = os.path.join(args.densecap_path, 'word_to_idx' + '.json')
        dc_embeddings = torch.load(os.path.join(args.densecap_path, 'densecap_emb.pt'))
    else:
        word_to_idx = os.path.join(args.data_folder, 'WORDMAP_' + args.data_name + '.json')
        dc_embeddings = None

    if args.embeddings_pretrained:
        print('Loading DenseCap vocabulary...')
        word_to_idx = os.path.join(args.densecap_path, 'word_to_idx' + '.json')
        idx_to_word = os.path.join(args.densecap_path, 'idx_to_word' + '.json')
        with open(word_to_idx, 'r') as j:
            word_to_idx = json.load(j)
        with open(idx_to_word, 'r') as j:
            idx_to_word = json.load(j)
            vocab_size = len(word_to_idx)
    else:
        word_to_idx = os.path.join(args.data_folder, 'WORDMAP_' + args.data_name + '.json')
        with open(word_to_idx, 'r') as j:
            word_to_idx = json.load(j)
        idx_to_word = {v: k for k, v in word_to_idx.items()}
        vocab_size = len(word_to_idx)

    args.vocab_size = vocab_size


    # Initialize / load checkpoint
    #if not args.checkpoint:

    if args.encoder_type == 'resnet512':
        encoder = Encoder()
        encoder_optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad,
                                                           encoder.parameters()),
                                             lr=args.encoder_lr)
        # Normalisation for ResNet
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])

        val_loader = data.DataLoader(
            ParagraphDataset(args.data_folder, args.data_name, 'VAL',
                             transform=transforms.Compose([normalize])),
            batch_size=args.batch_size, shuffle=False,
            num_workers=args.workers, pin_memory=True)

        #test_loader = data.DataLoader(
        #    ParagraphDataset(args.data_folder, args.data_name, 'TEST',
        #                     transform=transforms.Compose([normalize])),
        #    batch_size=args.batch_size, shuffle=True,
        #    num_workers=args.workers, pin_memory=True)

    elif args.encoder_type == 'densecap':
        encoder = RegPool(args)
        # Load train and val datasets
        print('Loading DenseCap features...')

        val_loader = data.DataLoader(
            ParagraphDataset(args.data_folder, args.data_name, 'TEST'),
            batch_size=args.batch_size, shuffle=False,
            num_workers=args.workers, pin_memory=True)

        encoder_optimizer = None

        # pick X elements for generation only
        #val_iterator = iter(val_loader)
        #val_loader = list(islice(val_iterator, 10))

        #test_loader = data.DataLoader(
        #    ParagraphDataset(args.data_folder, args.data_name, 'TEST',
        #                     transform=transforms.Compose([normalize])),
        #    batch_size=args.batch_size, shuffle=True,
        #    num_workers=args.workers, pin_memory=True)

    else:
        raise Exception('Unrecognized encoder type.')

    sentence_decoder = SentenceRNN(args)
    word_decoder = WordRNN(args)

    # Load the trained model parameters
    this_checkpoint = torch.load(args.model_path + args.model_trained)
    encoder.load_state_dict(this_checkpoint['encoder'].state_dict())
    sentence_decoder.load_state_dict(this_checkpoint['sentence_decoder'].state_dict())
    word_decoder.load_state_dict(this_checkpoint['word_decoder'].state_dict())

    # Move to GPU
    sentence_decoder = sentence_decoder.to(device)
    word_decoder = word_decoder.to(device)
    encoder = encoder.to(device)

    train_step = 0
    val_step = 0
    # Epochs
    #for epoch in range(args.start_epoch, args.num_epochs):

    # One epoch's validation
    b1, b2, b3, b4, c, m = generate(val_loader=val_loader,
                                    encoder=encoder,
                                    sentence_decoder=sentence_decoder,
                                    word_decoder=word_decoder,
                                    logger=None,
                                    vocab_size=vocab_size,
                                    word_to_idx=word_to_idx,
                                    idx_to_word=idx_to_word,
                                    dc_embeddings=dc_embeddings,
                                    args=args)
    print('BLEU 1', b1)
    print('BLEU 2', b2)
    print('BLEU 3', b3)
    print('BLEU 4', b4)
    print('CIDER', c)
    print('METEOR', m)
    
    
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

    
def generate(val_loader,
             encoder,
             sentence_decoder,
             word_decoder,
             logger,
             vocab_size,
             word_to_idx,
             idx_to_word,
             dc_embeddings,
             args):
    """
    Performs generation for the single epoch.

    :param val_loader: DataLoader for validation data.
    :param encoder: encoder model
    :param decoder: decoder model
    :param criterion: loss layer
    :return: BLEU-4 score
    """

    sentence_decoder.eval()
    word_decoder.eval()
    if encoder is not None:
        encoder.eval()

    paragraphs_generated = []
    
    Bleu_1 = 0
    Bleu_2 = 0
    Bleu_3 = 0
    Bleu_4 = 0
    CIDEr = 0
    METEOR = 0

    with torch.no_grad():
        # Batches
        for image_num, (imgs, image_ids, caps, caplens) in enumerate(val_loader):
            
            if image_num % 50 == 0:
                print(image_num)

            pars = {}

            #for the purpose of making sure generated texts make sense
            references_batch = dict()
            hypotheses_batch = dict()

            # Move to GPU, if available
            imgs = imgs.to(device)
            image_ids = image_ids.to(device)
            caps = caps.to(device)
            caplens = caplens.to(device)

            imgs = encoder(imgs)
            args.batch_size = imgs.shape[0]

            caplens_f, init_inx = caplens_eos(caplens, args.max_sentences)

            if args.encoder_type == 'resnet512':
                # Prepare images for sentence decoder
                imgs = imgs.view(args.batch_size, -1, args.resnet512_feat_dim)
                #num_pixels = imgs.size(1)
                imgs = imgs.mean(dim=1)


            h_sent = torch.zeros(args.num_layers_sentencernn, imgs.shape[0], args.hidden_size)
            c_sent = torch.zeros(args.num_layers_sentencernn, imgs.shape[0], args.hidden_size)
            h_word = torch.zeros(args.num_layers_wordrnn, imgs.shape[0], args.hidden_size)
            c_word = torch.zeros(args.num_layers_wordrnn, imgs.shape[0], args.hidden_size)
            h_word = h_word.to(device)
            c_word = c_word.to(device)
            
            generated = []
            actual = []

            for sent_num in range(args.max_sentences):
                

                p_source, topic, ht_sent, ct_sent = sentence_decoder(imgs, (h_sent, c_sent))

                h_sent = ht_sent
                c_sent = ct_sent

                p_predicted = nn.Sigmoid()(p_source)
                p_target = torch.LongTensor(caplens_f[init_inx].long().squeeze(1)).to(device)
                p_target = p_target.type_as(p_source)
                init_inx += 1

                #print(init_inx)
                #print(p_source)
                #print(p_target)
                #print(p_predicted)
                #print(sentrnn_loss)
                
                if p_predicted > 0.4:
                    break

                # WordRNN
                current_captions = caps[:, sent_num, :]
                current_caplens = caplens.squeeze(1)[:, sent_num]
                max_seq_length = current_caplens[torch.argmax(current_caplens, dim=0)]

                # generation
                start_token = torch.LongTensor([word_to_idx['<start>']]).to(device)
                end_token = torch.LongTensor([word_to_idx['<end>']]).to(device)
                #print(start_token)
                #print(end_token)
                #if args.use_fc:
                #    topic = word_decoder.non_lin(word_decoder.image_to_hidden(topic))
                inputs = topic
                
                this_gen_sentence = []
                temp = []
                final_caption = []
                
                if args.decoding_strategy == 'greedy':
                    for i in range(args.max_words):
                        #if i == 1:
                        #    inputs = word_decoder.embeddings(start_token).unsqueeze(1)
                        hiddens, (h_word, c_word) = word_decoder.decode_step(inputs, (h_word, c_word))
                        outputs = word_decoder.linear(hiddens)
                        outputs = F.softmax(outputs, dim=-1)
                        topv, topi = outputs.topk(1)
                        word_idx = topi.squeeze(0).detach()
                        this_gen_sentence.append(word_idx.item())
                        inputs = word_decoder.embeddings(word_idx)
                     
                if args.decoding_strategy == 'sampling':
                    for i in range(args.max_words):
                        #if i == 1:
                        #    inputs = word_decoder.embeddings(start_token).unsqueeze(1)
                        hiddens, (h_word, c_word) = word_decoder.decode_step(inputs, (h_word, c_word))
                        outputs = word_decoder.linear(hiddens)
                        if args.temperature:
                            outputs = outputs.div(args.temperature_value)
                        word_softmax = F.softmax(outputs, dim=-1).squeeze()
                        word_idx = torch.multinomial(word_softmax, 1)[0].to(device)
                        this_gen_sentence.append(word_idx.item())
                        #print(word_decoder.embeddings(word_idx).shape)
                        inputs = word_decoder.embeddings(word_idx).unsqueeze(0).unsqueeze(0)

                if args.decoding_strategy == 'topn_sampling':
                    for i in range(args.max_words):
                        #if i == 1:
                        #    inputs = word_decoder.embeddings(start_token).unsqueeze(1)
                        hiddens, (h_word, c_word) = word_decoder.decode_step(inputs, (h_word, c_word))
                        outputs = word_decoder.linear(hiddens)
                        if args.temperature:
                            outputs = outputs.div(args.temperature_value)
                        # top_n = 1 is greedy search, top_n = vocab_size is pure sampling
                        # controlling for nucleus sampling
                        if args.nucleus_sampling:
                            logits = top_k_top_p_filtering(outputs, top_k=args.top_n, top_p=args.top_p)
                        if not args.nucleus_sampling:
                            logits = top_k_top_p_filtering(outputs, top_k=args.top_n, top_p=0.0)
                        word_softmax = F.softmax(logits, dim=-1).squeeze()
                        word_idx = torch.multinomial(word_softmax, 1)[0].to(device)
                        this_gen_sentence.append(word_idx.item())
                        inputs = word_decoder.embeddings(word_idx).unsqueeze(0).unsqueeze(0)
                        
                if args.decoding_strategy == 'beam':

                    embs = word_decoder.embeddings.weight.data
                    embs = torch.cat([embs, inputs.squeeze(0)], dim=0)
                    #print(embs)
                    #print(embs[7604])
                    
                    word_to_idx['image'] = 7604
                    start = [word_to_idx['image']]
                    start_word = [[start, 0.0]]
                    final_caption = []
                    
                    #print('start', start)
                    #print('start word', start_word)
                    
                    while len(start_word[0][0]) < args.max_words:
                        
                        temp = []
                        
                        for s in start_word:
                            
                            #print('S', s)
                            
                            word_in = embs[torch.LongTensor([s[0][-1]])].unsqueeze(0)
                            hiddens, (h_word, c_word) = word_decoder.decode_step(word_in, (h_word, c_word))
                            outputs = word_decoder.linear(hiddens)
                            probs = F.softmax(outputs, dim=-1).squeeze(0)
                            probs = probs.cpu()
                            word_preds = np.argsort(probs[0])[-args.beam:]
                            #print('word preds', word_preds)

                            for w in word_preds:
                                #print('SSSS', s)
                                next_cap, prob = s[0][:], s[1]
                                #print('curr cap', next_cap)
                                #print('curr prob', prob)
                                next_cap.append(w.item())
                                new_prob = prob + probs[0][w]
                                #print(w)
                                #print(prob, probs[0][w])
                                #print('final prob', new_prob)
                                temp.append([next_cap, new_prob])
                            
                        start_word = temp
                        start_word = sorted(start_word, reverse=False, key=lambda l: l[1])
                        #print('after sorting', start_word)
                        start_word = start_word[-args.beam:]
                        #print('after removing', start_word)
                        
                    start_word = start_word[-1][0]
                    #print('FINAL START WORD', start_word)
                    intermediate_caption = [i for i in start_word[1:]]
                
                    for i in intermediate_caption:
                        if i != '<end>':
                            final_caption.append(i)
                        else:
                            break
                    for i in final_caption:
                        this_gen_sentence.append(i)
                        
                        
                        
                gen_sentence_filtered = []
                for i in this_gen_sentence:
                    if i != end_token.item():
                        gen_sentence_filtered.append(i)
                    else:
                        gen_sentence_filtered.append(end_token.item())
                        break
                
                generated.append(gen_sentence_filtered)
                actual.append(current_captions)
                
            # EVALUATION
            
            # str(int(w))
            
            # Get references
            for single_paragraph in range(len(actual)):
                if single_paragraph == 0:
                    image_id = image_ids[single_paragraph].item()
                    references_batch[image_id] = []
                img_caps = actual[single_paragraph].tolist()
                img_captions = list(
                    map(lambda c: [w for w in c if w not in {word_to_idx['<start>'], word_to_idx['<pad>'], word_to_idx['<end>']}],
                        img_caps))
                paragraph_text = []
                for sent in img_captions:
                    this_sent_text = [idx_to_word[w] for w in sent]
                    paragraph_text.append(' '.join(this_sent_text))
                #if paragraph_text != [' ']:
                if ' '.join(this_sent_text) != '':
                    #references_batch[image_id].append(paragraph_text)
                    references_batch[image_id].append(' '.join(this_sent_text))
            
            # Get hypotheses
            for predicted_paragraph in range(len(generated)):
                if predicted_paragraph == 0:
                    image_id = image_ids[predicted_paragraph].item()
                    hypotheses_batch[image_id] = []
                par_preds = generated[predicted_paragraph]
                this_sentence_text = [idx_to_word[w] for w in par_preds[1:]]
                hypotheses_batch[image_id].append(' '.join(this_sentence_text))
                
            #print(hypotheses_batch)

            assert len(references_batch.keys()) == len(hypotheses_batch.keys())

            # Calculate BLEU & CIDEr & METEOR & ROUGE scores
            # WARNING: at the moment, only BLEU is calculated: one hyp sentence is compared to all ref sentences

            scorers = [

                (Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
                (Cider('vg-test-words'), "CIDEr"),
                (Meteor(), "METEOR"),
                #(Rouge(), "ROUGE_L")
            ]

            score = []
            method = []
            for scorer, method_i in scorers:
                
                score_i, _ = scorer.compute_score(references_batch, hypotheses_batch)
                score.extend(score_i) if isinstance(score_i, list) else score.append(score_i)
                method.extend(method_i) if isinstance(method_i, list) else method.append(method_i)

            score_dict = dict(zip(method, score))
            #print(score_dict)
            
            Bleu_1 += score_dict['Bleu_1']
            Bleu_2 += score_dict['Bleu_2']
            Bleu_3 += score_dict['Bleu_3']
            Bleu_4 += score_dict['Bleu_4']
            CIDEr += score_dict['CIDEr']
            METEOR += score_dict['METEOR']
            
            pars['image_id'] = list(references_batch.keys())[0]
            pars['references'] = ' '.join(list(references_batch.values())[0])
            pars['hypotheses'] = ' '.join(list(hypotheses_batch.values())[0])

            paragraphs_generated.append(pars)

    Bleu1_av = Bleu_1 / len(val_loader)
    Bleu2_av = Bleu_2 / len(val_loader)
    Bleu3_av = Bleu_3 / len(val_loader)
    Bleu4_av = Bleu_4 / len(val_loader)
    Cider_av = CIDEr / len(val_loader)
    Meteor_av = METEOR / len(val_loader)

    with open('./generated_paragraphs.json', 'w') as f:
        json.dump(paragraphs_generated, f)
        
    return Bleu1_av, Bleu2_av, Bleu3_av, Bleu4_av, Cider_av, Meteor_av

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
    sentence_pooling_dim = config_parser.get('PARAMS-SENTENCE', 'pooling_dim')
    resnet512_feat_dim = config_parser.get('PARAMS-SENTENCE', 'resnet512_feat_dim')
    densecap_feat_dim = config_parser.get('PARAMS-SENTENCE', 'densecap_feat_dim')
    eos_classes = config_parser.get('PARAMS-SENTENCE', 'eos_classes')
    sentence_decoder_lr = config_parser.get('PARAMS-SENTENCE', 'sentence_decoder_lr')
    lambda_sentence = config_parser.get('PARAMS-SENTENCE', 'lambda_sentence')
    sent_grad_clip = config_parser.get('PARAMS-SENTENCE', 'sent_grad_clip')

    embed_dim = config_parser.get('PARAMS-WORD', 'embed_dim')
    num_layers_wordrnn = config_parser.get('PARAMS-WORD', 'num_layers_wordrnn')
    max_sentences = config_parser.get('PARAMS-WORD', 'max_sentences')
    max_words = config_parser.get('PARAMS-WORD', 'max_words')
    word_decoder_lr = config_parser.get('PARAMS-WORD', 'word_decoder_lr')
    lambda_word = config_parser.get('PARAMS-WORD', 'lambda_word')
    word_grad_clip = config_parser.get('PARAMS-WORD', 'word_grad_clip')

    batch_size = config_parser.get('PARAMS-MODELS', 'batch_size')
    num_boxes = config_parser.get('PARAMS-MODELS', 'num_boxes')
    word_dropout = config_parser.get('PARAMS-WORD', 'word_dropout')
    encoder_dropout = config_parser.get('PARAMS-MODELS', 'encoder_dropout')
    dropout_fc = config_parser.get('PARAMS-MODELS', 'dropout_fc')
    dropout_stopping = config_parser.get('PARAMS-MODELS', 'dropout_stopping')
    encoder_type = config_parser.get('PARAMS-MODELS', 'encoder_type')
    encoder_lr = config_parser.get('PARAMS-MODELS', 'encoder_lr')
    rnn_hidden_init = config_parser.get('PARAMS-MODELS', 'rnn_hidden_init')
    embeddings_pretrained = config_parser.getboolean('PARAMS-MODELS', 'embeddings_pretrained')
    freeze = config_parser.getboolean('PARAMS-MODELS', 'freeze')
    rnn_arch = config_parser.get('PARAMS-MODELS', 'rnn_arch')
    start_epoch = config_parser.get('PARAMS-MODELS', 'start_epoch')
    num_epochs = config_parser.get('PARAMS-MODELS', 'num_epochs')
    rnn_arch = config_parser.get('PARAMS-MODELS', 'rnn_arch')
    start_epoch = config_parser.get('PARAMS-MODELS', 'start_epoch')
    workers = config_parser.get('PARAMS-MODELS', 'workers')
    print_freq = config_parser.get('PARAMS-MODELS', 'print_freq')
    checkpoint = config_parser.getboolean('PARAMS-MODELS', 'checkpoint')
    best_cider = config_parser.get('EVAL-SCORES', 'best_cider')
    best_val_loss = config_parser.get('EVAL-SCORES', 'best_val_loss')
    layer_norm = config_parser.getboolean('PARAMS-MODELS', 'layer_norm')
    exp_num = config_parser.get('PARAMS-MODELS', 'exp_num')
    clipping = config_parser.getboolean('PARAMS-MODELS', 'clipping')
    encoder_weight_decay = config_parser.get('PARAMS-MODELS', 'encoder_weight_decay')
    sentence_weight_decay = config_parser.get('PARAMS-SENTENCE', 'sentence_weight_decay')
    word_weight_decay = config_parser.get('PARAMS-WORD', 'word_weight_decay')
    sentence_nonlin = config_parser.getboolean('PARAMS-SENTENCE', 'sentence_nonlin')
    use_fc2 = config_parser.getboolean('PARAMS-SENTENCE', 'use_fc2')
    use_fc = config_parser.getboolean('PARAMS-SENTENCE', 'use_fc')
    bn = config_parser.getboolean('PARAMS-MODELS', 'bn')
    wordlstm_dropout = config_parser.get('PARAMS-WORD', 'wordlstm_dropout')
    feature_linear = config_parser.getboolean('PARAMS-MODELS', 'feature_linear')
    model_trained = config_parser.get('PARAMS-MODELS', 'model_trained')
    topic_hidden = config_parser.getboolean('PARAMS-WORD', 'topic_hidden')
    with_densecap_captions = config_parser.getboolean('PARAMS-MODELS', 'with_densecap_captions')
    decoding_strategy = config_parser.get('PARAMS-MODELS', 'decoding_strategy')
    temperature = config_parser.getboolean('PARAMS-MODELS', 'temperature')
    temperature_value = config_parser.get('PARAMS-MODELS', 'temperature_value')
    nucleus_sampling = config_parser.getboolean('PARAMS-MODELS', 'nucleus_sampling')
    top_n = config_parser.get('PARAMS-MODELS', 'top_n')
    top_p = config_parser.get('PARAMS-MODELS', 'top_p')
    beam = config_parser.get('PARAMS-MODELS', 'beam')
    attention = config_parser.getboolean('PARAMS-MODELS', 'attention')

    api_key = config_parser.get('COMET', 'api_key')
    project_name = config_parser.get('COMET', 'project_name')
    workspace = config_parser.get('COMET', 'workspace')

    # Paths
    parser.add_argument('--data_folder', type=str, default=out_path, help='path to all input data')
    parser.add_argument('--data_name', type=str, default=data_name, help='shared name among target files')
    parser.add_argument('--model_path', type=str, default=checkpoint_path, help='path for keeping model"s checkpoints')
    parser.add_argument('--model_trained', type=str, default=model_trained, help='actual trained model to use for generation')
    parser.add_argument('--densecap_path', type=str, default=densecap_path, help='path to the pretrained DenseCap materials')
    parser.add_argument('--data_folder_densecap', type=str, default=data_folder_densecap, help='path to the data when using DenseCap encoder')

    # Model parameters
    parser.add_argument('--num_boxes', type=int, default=num_boxes, help='number of regions to look at in the image')
    parser.add_argument('--max_sentences', type=int, default=max_sentences, help='maximum number of sentences per paragraph')
    parser.add_argument('--max_words', type=int, default=max_words, help='maximum number of words per sentence')
    parser.add_argument('--num_layers_sentencernn', type=int, default=num_layers_sentencernn, help='number of layers in sentenceRNN')
    parser.add_argument('--num_layers_wordrnn', type=int, default=num_layers_wordrnn, help='number of layers in wordRNN')
    parser.add_argument('--word_dropout', type=float, default=word_dropout, help='dropout rate for word LSTM')
    parser.add_argument('--encoder_dropout', type=float, default=encoder_dropout, help='dropout for the image encoding')
    parser.add_argument('--dropout_fc', type=float, default=dropout_fc, help='dropout for the topic vector')
    parser.add_argument('--dropout_stopping', type=float, default=dropout_stopping, help='dropout for stopping distribution')
    parser.add_argument('--sentence_lr', type=float, default=sentence_decoder_lr, help='learning rate for sentenceRNN')
    parser.add_argument('--word_lr', type=float, default=word_decoder_lr, help='learning rate for wordRNN')
    parser.add_argument('--hidden_size', type=int, default=hidden_size, help='size of hidden dimension')
    parser.add_argument('--pooling_dim', type=int, default=sentence_pooling_dim, help='dimension of projected features')
    parser.add_argument('--resnet512_feat_dim', type=int, default=resnet512_feat_dim, help='dimension of projected features when using ResNet enoder')
    parser.add_argument('--densecap_feat_dim', type=int, default=densecap_feat_dim, help='dimension of projected features when using DenseCap encoder')
    parser.add_argument('--eos_classes', type=int, default=eos_classes, help='number of classes of eos prediction (CONTINUE/STOP)')
    parser.add_argument('--lambda_sentence', type=float, default=lambda_sentence, help='sentence normalisation parameter')
    parser.add_argument('--embed_dim', type=int, default=embed_dim, help='word embedding dimension')
    parser.add_argument('--lambda_word', type=float, default=lambda_word, help='word normalisation parameter')

    parser.add_argument('--batch_size', type=int, default=1, help='batch size')
    parser.add_argument('--encoder_type', type=str, default=encoder_type, help='which encoder to use')
    parser.add_argument('--encoder_lr', type=float, default=encoder_lr, help='learning rate for the encoder')
    parser.add_argument('--rnn_hidden_init', type=str, default=rnn_hidden_init, help='initialise wordRNN initial states with zeros or with features')
    parser.add_argument('--embeddings_pretrained', type=bool, default=embeddings_pretrained, help='use DenseCap embeddings or not')
    parser.add_argument('--freeze', type=bool, default=freeze, help='if pretrained embeddings, freeze them or not')
    parser.add_argument('--rnn_arch', type=str, default=rnn_arch, help='LSTM or GRU as wordRNN')
    parser.add_argument('--start_epoch', type=int, default=start_epoch, help='start from epoch number')
    parser.add_argument('--num_epochs', type=int, default=num_epochs, help='number of epochs')
    parser.add_argument('--workers', type=int, default=workers, help='number of workers')
    parser.add_argument('--print_freq', type=int, default=print_freq, help='print information after X batches')
    parser.add_argument('--checkpoint', type=bool, default=checkpoint, help='start training from checkpoint or not')
    parser.add_argument('--exp_num', type=str, default=exp_num, help='track number of experiments and various model settings, used to create directories for corresponding experiments within ./models')
    parser.add_argument('--best_cider', type=float, default=best_cider, help='initial best cider score')
    parser.add_argument('--best_val_loss', type=float, default=best_val_loss, help='initial best validation loss')
    parser.add_argument('--layer_norm', type=bool, default=layer_norm, help='use batch normalisation or not')
    parser.add_argument('--sent_grad_clip', type=float, default=sent_grad_clip, help='value for clipping gradients for sentence LSTM')
    parser.add_argument('--word_grad_clip', type=float, default=word_grad_clip, help='value for clipping gradients for word LSTM')
    parser.add_argument('--clipping', type=bool, default=clipping, help='use clipping or not')
    parser.add_argument('--encoder_weight_decay', type=float, default=encoder_weight_decay, help='weight decay for the encoder')
    parser.add_argument('--sentence_weight_decay', type=float, default=sentence_weight_decay, help='weight decay for the sentence LSTM')
    parser.add_argument('--word_weight_decay', type=float, default=word_weight_decay, help='weight decay for the word LSTM')
    parser.add_argument('--sentence_nonlin', type=bool, default=sentence_nonlin, help='use non-linearity in sentence LSTM or not')
    parser.add_argument('--use_fc2', type=bool, default=use_fc2, help='use 2 linear layers for topic creating or 1 linear layer')
    parser.add_argument('--use_fc', type=bool, default=use_fc, help='use fully connected layer or not for topic modelling')
    parser.add_argument('--bn', type=bool, default=bn, help='apply batch normalisation in the encoder or not')
    parser.add_argument('--wordlstm_dropout', type=float, default=wordlstm_dropout, help='dropout for embedding layer')
    parser.add_argument('--feature_linear', type=bool, default=feature_linear, help='add linear layer for image features or not')
    parser.add_argument('--topic_hidden', type=bool, default=topic_hidden, help='initialise word LSTM from image topic or not')
    parser.add_argument('--with_densecap_captions', type=bool, default=with_densecap_captions, help='use densecap captions to create language topic or not')
    parser.add_argument('--temperature', type=bool, default=temperature, help='use temperature in decoding or not')
    parser.add_argument('--temperature_value', type=float, default=temperature_value, help='value for temperature if using it')
    parser.add_argument('--decoding_strategy', type=str, default=decoding_strategy, help='decoder strategy')
    parser.add_argument('--nucleus_sampling', type=bool, default=nucleus_sampling, help='use nucleus sampling or not')
    parser.add_argument('--top_n', type=int, default=top_n, help='top n candidates when using top n sampling')
    parser.add_argument('--top_p', type=float, default=top_p, help='top p for nucleus sampling')
    parser.add_argument('--beam', type=int, default=beam, help='beam search depth')
    parser.add_argument('--attention', type=bool, default=attention, help='use attention or not')

    parser.add_argument('--api_key', type=str, default=api_key, help='key for the Comet logger')
    parser.add_argument('--project_name', type=str, default=project_name, help='name of the project')
    parser.add_argument('--workspace', type=str, default=workspace, help='owner of the Comet workspace')

    arguments = parser.parse_args()
    main(arguments)
