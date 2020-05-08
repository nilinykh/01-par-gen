'''
full training and validation loop
'''

# do not forget to set CUDA_VISIBLE_DEVICES to any of the gpus you would like to use (!!!)

import argparse
import sys
from configparser import ConfigParser

from comet_ml import Experiment
import torch.backends.cudnn as cudnn
import torch.optim
from torch.utils import data
import torchvision.transforms as transforms
from torch import nn
from models import Encoder, RegPool, SentenceRNN, WordRNN
from datasets import *
from utils import *
from train import *
from validate import *
device = torch.device("cuda")
cudnn.benchmark = True

sys.path.append('/home/xilini/par-gen/01-par-gen')

def main(args):
    """
    Training and validation.
    """

    experiment = Experiment(api_key=args.api_key,
                            project_name=args.project_name, workspace=args.workspace)

    hyper_params = {
        'batch_size': args.batch_size,
        'num_epochs': args.num_epochs,
        'rnn_type': args.rnn_arch,
        'encoder_type': args.encoder_type,
        'rnn_hidden_init': args.rnn_hidden_init,
        'embeddings_pretrained': args.embeddings_pretrained,
        'freeze': args.freeze,
        'encoder_lr': args.encoder_lr,
        'layers_wordrnn': args.num_layers_wordrnn,
        'sentence_decoder_lr': args.sentence_lr,
        'word_decoder_lr': args.word_lr,
        'lambda_sentence': args.lambda_sentence,
        'lambda_word': args.lambda_word,
        'max_sentence': args.max_sentences,
        'max_words': args.max_words,
        'encoder_dropout': args.encoder_dropout,
        'word_dropout': args.word_dropout,
        'wordlstm_dropout': args.wordlstm_dropout,
        'dropout_fc': args.dropout_fc,
        'dropout_stopping': args.dropout_stopping,
        'layer_norm': args.layer_norm,
        'encoder_weight_decay': args.encoder_weight_decay,
        'sentence_weight_decay': args.sentence_weight_decay,
        'word_weight_decay': args.word_weight_decay,
        'clipping': args.clipping
    }

    experiment.log_parameters(hyper_params)

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

    if args.encoder_type == 'resnet512':
        encoder = Encoder()
        encoder_optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad,
                                                           encoder.parameters()),
                                             lr=args.encoder_lr)
        # Normalisation for ResNet
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])

        train_loader = data.DataLoader(
            ParagraphDataset(args.data_folder, args.data_name, 'VAL',
                             transform=transforms.Compose([normalize])),
            batch_size=args.batch_size, shuffle=True,
            num_workers=args.workers, pin_memory=True)

        val_loader = data.DataLoader(
            ParagraphDataset(args.data_folder, args.data_name, 'VAL',
                             transform=transforms.Compose([normalize])),
            batch_size=args.batch_size, shuffle=True,
            num_workers=args.workers, pin_memory=True)

    elif args.encoder_type == 'densecap':
        encoder = RegPool(args)
        if args.feature_linear:
            encoder_optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad,
                                                               encoder.parameters()),
                                                 lr=args.encoder_lr)
        elif not args.feature_linear:
            encoder_optimizer = None

        # Load train and val datasets
        print('Loading DenseCap features...')

        train_loader = data.DataLoader(
            ParagraphDataset(args.data_folder, args.data_name, 'TRAIN'),
            batch_size=args.batch_size, shuffle=True,
            num_workers=args.workers, pin_memory=True)

        val_loader = data.DataLoader(
            ParagraphDataset(args.data_folder, args.data_name, 'VAL'),
            batch_size=args.batch_size, shuffle=True,
            num_workers=args.workers, pin_memory=True)

        # datasets for testing overfitting
        #train_iterator = iter(train_loader)
        # 5 batches of X samples
        #train_loader = list(islice(train_iterator, 5))
        #val_loader = train_loader

    else:
        raise Exception('Unrecognized encoder type.')

    #print(len(train_loader))
    #print(len(val_loader))

    sentence_decoder = SentenceRNN(args)
    word_decoder = WordRNN(args)

    if args.sentence_weight_decay != 0 and args.word_weight_decay != 0:
        print('Using weight decay, L2 regularisation')

    sentence_optimizer = torch.optim.Adam(list(sentence_decoder.parameters()), lr=args.sentence_lr,
                                          weight_decay=args.sentence_weight_decay)
    word_optimizer = torch.optim.Adam(list(word_decoder.parameters()), lr=args.word_lr,
                                      weight_decay=args.word_weight_decay)
    sentence_decoder.to(device)
    word_decoder.to(device)
    encoder.to(device)

    #densecap_embedding_layer = nn.Embedding(len(word_to_idx), 512)

    print('IMAGE ENCODER', encoder)
    print('SENTENCE DECODER', sentence_decoder)
    print('WORD DECODER', word_decoder)
    print('SENTENCE OPTIMIZER', sentence_optimizer)
    print('WORD OPTIMIZER', word_optimizer)

    epochs_since_improvement = 0

    # Move to GPU

    #sentence_decoder = sentence_decoder.to(device)
    #word_decoder = word_decoder.to(device)
    #encoder = encoder.to(device)

    # Loss functions
    criterion_sent = nn.BCEWithLogitsLoss(reduction='none').to(device)
    criterion_word = nn.CrossEntropyLoss(reduction='none', ignore_index=0).to(device)

    #sentence_lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(sentence_optimizer, 'min',
    #                                                                   verbose=True, patience=1)
    #word_lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(word_optimizer, 'min',
    #                                                               verbose=True, patience=1)

    initial_best_loss = args.best_val_loss

    train_epoch_loss = []
    train_sentence_epoch_loss = []
    train_word_epoch_loss = []
    val_epoch_loss = []
    val_sentence_epoch_loss = []
    val_word_epoch_loss = []
    
    curr_val_sentence_loss = 500

    # Epochs
    for epoch in range(args.start_epoch, args.num_epochs):

        # One epoch's training
        epoch_loss,\
        this_epoch_sentence,\
        this_epoch_word = train(train_loader=train_loader,
                                encoder=encoder,
                                sentence_decoder=sentence_decoder,
                                word_decoder=word_decoder,
                                criterion_sent=criterion_sent,
                                criterion_word=criterion_word,
                                encoder_optimizer=encoder_optimizer,
                                sentence_optimizer=sentence_optimizer,
                                word_optimizer=word_optimizer,
                                word_to_idx=word_to_idx,
                                dc_embeddings=dc_embeddings,
                                epoch=epoch,
                                logger=experiment,
                                args=args)

        #print(epoch_loss, this_epoch_sentence, this_epoch_word)

        experiment.log_metric("train_loss", epoch_loss, step=epoch)
        experiment.log_metric("train_sentence_loss", this_epoch_sentence, step=epoch)
        experiment.log_metric("train_word_loss", this_epoch_word, step=epoch)

        # One epoch's validation
        this_val_epoch_loss,\
        val_this_epoch_sentence,\
        val_this_epoch_word = validate(val_loader=val_loader,
                                       encoder=encoder,
                                       sentence_decoder=sentence_decoder,
                                       word_decoder=word_decoder,
                                       criterion_sent=criterion_sent,
                                       criterion_word=criterion_word,
                                       logger=experiment,
                                       vocab_size=vocab_size,
                                       word_to_idx=word_to_idx,
                                       idx_to_word=idx_to_word,
                                       dc_embeddings=dc_embeddings,
                                       args=args)

        #print(val_epoch_loss, val_this_epoch_sentence, val_this_epoch_word)
        
        #if curr_val_sentence_loss > val_this_epoch_sentence:
        #    curr_val_sentence_loss = val_this_epoch_sentence
        #    for param in sentence_decoder.parameters():
        #        param.requires_grad = True
        #    print('Sentence RNN is trained!')
        #elif curr_val_sentence_loss < val_this_epoch_sentence:
        #    for param in sentence_decoder.parameters():
        #        param.requires_grad = False
        #    print('Sentence RNN is frozen!')

        experiment.log_metric("val_loss", this_val_epoch_loss, step=epoch)
        experiment.log_metric("val_sentence_loss", val_this_epoch_sentence, step=epoch)
        experiment.log_metric("val_word_loss", val_this_epoch_word, step=epoch)

        train_epoch_loss.append(epoch_loss)
        train_sentence_epoch_loss.append(this_epoch_sentence)
        train_word_epoch_loss.append(this_epoch_word)
        val_epoch_loss.append(this_val_epoch_loss)
        val_sentence_epoch_loss.append(val_this_epoch_sentence)
        val_word_epoch_loss.append(val_this_epoch_word)

        plot_loss(train_epoch_loss, val_epoch_loss, args.exp_num, loss_type='full')
        plot_loss(train_sentence_epoch_loss, val_sentence_epoch_loss, args.exp_num,
                  loss_type='sentence')
        plot_loss(train_word_epoch_loss, val_word_epoch_loss, args.exp_num, loss_type='word')

        #sentence_lr_scheduler.step(val_this_epoch_sentence)
        #word_lr_scheduler.step(val_this_epoch_word)

        # Check if there was an improvement
        if initial_best_loss == 0 and initial_best_loss < this_val_epoch_loss:
            is_best = True
            initial_best_loss = this_val_epoch_loss
        elif initial_best_loss > this_val_epoch_loss:
            is_best = True
            initial_best_loss = this_val_epoch_loss
        else:
            is_best = False

        # Save checkpoint
        save_checkpoint(data_name,
                        epoch,
                        epochs_since_improvement,
                        encoder,
                        sentence_decoder,
                        word_decoder,
                        encoder_optimizer,
                        sentence_optimizer,
                        word_optimizer,
                        this_val_epoch_loss,
                        is_best,
                        args.exp_num)


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
    topic_hidden = config_parser.getboolean('PARAMS-WORD', 'topic_hidden')
    with_densecap_captions = config_parser.getboolean('PARAMS-MODELS', 'with_densecap_captions')
    attention = config_parser.getboolean('PARAMS-MODELS', 'attention')

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

    parser.add_argument('--batch_size', type=int, default=batch_size, help='batch size')
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
    parser.add_argument('--attention', type=bool, default=attention, help='use attention or not')

    parser.add_argument('--api_key', type=str, default=api_key, help='key for the Comet logger')
    parser.add_argument('--project_name', type=str, default=project_name, help='name of the project')
    parser.add_argument('--workspace', type=str, default=workspace, help='owner of the Comet workspace')

    arguments = parser.parse_args()
    main(arguments)
