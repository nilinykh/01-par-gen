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

    if args.encoder_type == 'resnet52':
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
            batch_size=args.batch_size, shuffle=True,
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
            batch_size=args.batch_size, shuffle=True,
            num_workers=args.workers, pin_memory=True)

        encoder_optimizer = None
        
        # pick X elements for generation only
        val_iterator = iter(val_loader)
        val_loader = list(islice(val_iterator, 100))

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
    _ = generate(val_loader=val_loader,
                 encoder=encoder,
                 sentence_decoder=sentence_decoder,
                 word_decoder=word_decoder,
                 logger=None,
                 vocab_size=vocab_size,
                 word_to_idx=word_to_idx,
                 idx_to_word=idx_to_word,
                 dc_embeddings=dc_embeddings,
                 args=args)


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

    cider_epoch = {}
    cider_epoch['CIDEr'] = list()
    
    paragraphs_generated = []

    with torch.no_grad():
        # Batches
        for _, (imgs, image_ids, caps, caplens, densecap_captions) in enumerate(val_loader):
            
            pars = {}
            
            #for the purpose of making sure generated texts make sense
            references_batch = dict()
            hypotheses_batch = dict()

            # Move to GPU, if available
            densecap_captions = [ast.literal_eval(elem) for elem in densecap_captions]
            imgs = imgs.to(device)
            image_ids = image_ids.to(device)
            caps = caps.to(device)
            caplens = caplens.to(device)
            
            #print(caps)
            
            if args.with_densecap_captions:
                phrase_embeddings = densecap_to_embeddings(densecap_captions, word_to_idx, dc_embeddings)
            else:
                phrase_embeddings = None
            
            imgs = encoder(imgs)
            args.batch_size = imgs.shape[0]
            
            caplens_f, init_inx = caplens_eos(caplens, args.max_sentences)
            
            if args.encoder_type == 'resnet512':
                # Prepare images for sentence decoder
                imgs = imgs.view(args.batch_size, -1, args.resnet512_feat_dim)
                num_pixels = imgs.size(1)
                imgs = imgs.mean(dim=1)

            scores_all = torch.zeros(imgs.shape[0], args.max_sentences, args.max_words + 2, vocab_size).to(device)
            targets_all = torch.zeros(imgs.shape[0], args.max_sentences, args.max_words + 2).to(device)

            h_sent = torch.zeros(args.num_layers_sentencernn, imgs.shape[0], args.hidden_size)
            c_sent = torch.zeros(args.num_layers_sentencernn, imgs.shape[0], args.hidden_size)
            h_word = torch.zeros(args.num_layers_wordrnn, imgs.shape[0], args.hidden_size)
            c_word = torch.zeros(args.num_layers_wordrnn, imgs.shape[0], args.hidden_size)

            for sent_num in range(args.max_sentences):

                p_source, topic, ht_sent, ct_sent = sentence_decoder(imgs, phrase_embeddings, (h_sent, c_sent))
                
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

                # WordRNN
                current_captions = caps[:, sent_num, :]
                current_caplens = caplens.squeeze(1)[:, sent_num]
                max_seq_length = current_caplens[torch.argmax(current_caplens, dim=0)]

                scores,\
                sorted_caplens,\
                caps_decoder,\
                sort_ind,\
                ht_word, ct_word = word_decoder(topic, current_captions, current_caplens, imgs.shape[0], (h_word, c_word))
                
                h_word = ht_word
                c_word = ct_word

                if args.topic_hidden:
                    targets = caps_decoder[:, 1:max_seq_length]
                else:
                    targets = pack_padded_sequence(caps_decoder, sorted_caplens, batch_first=True)[0]
                    #targets = caps_decoder[:, :max_seq_length]

                if args.topic_hidden:
                    scores_all[:scores.shape[0], sent_num, :max_seq_length-1, :] = scores
                    targets_all[:targets.shape[0], sent_num, :max_seq_length-1] = targets
                else:
                    scores_all[:scores.shape[0], sent_num, :max_seq_length, :] = scores
                    targets_all[:targets.shape[0], sent_num, :max_seq_length] = targets

            # EVALUATION
            # Get references
            for single_paragraph in range(targets_all.shape[0]):
                img_caps = targets_all[single_paragraph].tolist()
                #print(img_caps)
                #print(word_to_idx['<start>'], word_to_idx['<pad>'])
                img_captions = list(
                    map(lambda c: [w for w in c if w not in {word_to_idx['<start>'], word_to_idx['<pad>'], word_to_idx['raining']}],
                        img_caps))  # remove <start> and pads
                paragraph_text = []
                for sent in img_captions:
                    this_sent_text = [idx_to_word[str(int(w))] for w in sent]
                    #print(this_sent_text)
                    #print()
                    paragraph_text.append(' '.join(this_sent_text))
                references_batch[image_ids[single_paragraph].item()] = paragraph_text

            # Get hypotheses
            # 64 x 6 x 52 x vocab size
            scores_probs = F.softmax(scores_all, dim=3)
            preds = torch.argmax(scores_probs, dim=3)
            caplens_squeezed = caplens.squeeze(1)
            for predicted_paragraph in range(preds.shape[0]):
                par_preds = preds[predicted_paragraph].tolist()
                par_text = []
                for sent_num, sent in enumerate(par_preds):
                    this_sentence = sent[:caplens_squeezed[predicted_paragraph, sent_num]]
                    # remove first prediction (start token)
                    this_sentence_text = [idx_to_word[str(int(w))] for w in this_sentence][1:]
                    text_with_end = []
                    for elem in this_sentence_text:
                        if elem != '<end>':
                            text_with_end.append(elem)
                        elif elem == '<end>':
                            text_with_end.append('<end>')
                            break
                    par_text.append(' '.join(text_with_end))
                hypotheses_batch[image_ids[predicted_paragraph].item()] = par_text

            assert len(references_batch.keys()) == len(hypotheses_batch.keys())

            #print('REFERENCES\t', references_batch)
            #with open('./results_base.json', 'w') as f:
            #    json.dump(hypotheses_batch, f)
            #print('HYPOTHESES\t', hypotheses_batch)
            #print()

            # Calculate BLEU & CIDEr & METEOR & ROUGE scores
            # WARNING: at the moment, only BLEU is calculated: one hyp sentence is compared to all ref sentences

            #scorers = [

            #    (Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
            #    (Cider('coco-val-df'), "CIDEr"),
                #(Meteor(), "METEOR"),
                #(Rouge(), "ROUGE_L")
            #]

            #score = []
            #method = []
            #for scorer, method_i in scorers:
            #    score_i, _ = scorer.compute_score(references_batch, hypotheses_batch)
            #    score.extend(score_i) if isinstance(score_i, list) else score.append(score_i)
            #    method.extend(method_i) if isinstance(method_i, list) else method.append(method_i)

            #score_dict = dict(zip(method, score))
            #print(score_dict)
            
            pars['image_id'] = list(references_batch.keys())[0]
            pars['references'] = list(references_batch.values())[0]
            pars['hypotheses'] = list(hypotheses_batch.values())[0]
    
            print()
            print('IMAGE ID', list(references_batch.keys())[0])
            print('GROUND TRUTH')
            print(references_batch.values())
            print('GENERATED')
            print(hypotheses_batch.values())
            print('------------------')
            
            paragraphs_generated.append(pars)
            
    with open('./generated_paragraphs_densecap.json', 'w') as f:
        json.dump(paragraphs_generated, f)
            
    return None

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

    parser.add_argument('--api_key', type=str, default=api_key, help='key for the Comet logger')
    parser.add_argument('--project_name', type=str, default=project_name, help='name of the project')
    parser.add_argument('--workspace', type=str, default=workspace, help='owner of the Comet workspace')

    arguments = parser.parse_args()
    main(arguments)