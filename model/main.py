'''

training and validation loops

'''

# set CUDA_VISIBLE_DEVICES

import argparse
import sys
from configparser import ConfigParser
import pickle

from comet_ml import Experiment
import torch.backends.cudnn as cudnn
import torch.optim
from torch.utils import data
import torchvision.transforms as transforms
from torch import nn
from models import RegPool, SentenceRNN, WordRNN
from datasets import *
from utils import *
from train import *
device = torch.device("cuda")
cudnn.benchmark = True

sys.path.append('/home/xilini/par-gen/01-par-gen')

def main(args):

    experiment = Experiment(api_key=args.api_key,
                            project_name=args.project_name, workspace=args.workspace)

    hyper_params = {
        'batch_size': args.batch_size,
        'num_epochs': args.num_epochs,
        'word_decoder_lr': args.word_lr,
        'lambda_word': args.lambda_word,
        'max_sentences': args.max_sentences,
        'max_words': args.max_words,
        'word_weight_decay': args.word_weight_decay,
    }

    experiment.log_parameters(hyper_params)

    word_to_idx = os.path.join(args.data_folder, 'wordmap_' + args.data_name + '.json')
    with open(word_to_idx, 'r') as j:
        word_to_idx = json.load(j)
    idx_to_word = {v: k for k, v in word_to_idx.items()}
    vocab_size = len(word_to_idx)

    args.vocab_size = vocab_size

    encoder = RegPool(args)
    encoder_optimizer = None

    # Load train and val datasets
    print('Loading Datasets...')

    train_loader = data.DataLoader(
        ParagraphDataset(args.data_folder, args.data_name, 'TRAIN'),
        batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True)

    val_loader = data.DataLoader(
        ParagraphDataset(args.data_folder, args.data_name, 'VAL'),
        batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True)

    sentence_decoder = SentenceRNN(args)
    word_decoder = WordRNN(args)
    
    '''word_decoder_weights = word_decoder.state_dict()
    with open('/home/xilini/par-data/densecap-reworked/rnn_densecap.pkl', 'rb') as f:
        densecap_weights = pickle.load(f)
    pretrained_dict = {k: v for k, v in densecap_weights.items() if k in word_decoder_weights}
    word_decoder_weights.update(pretrained_dict)
    word_decoder.load_state_dict(word_decoder_weights)
    
    word_decoder_weights = word_decoder.state_dict()
    #print(word_decoder_weights)
    for name, param in word_decoder.named_parameters():
        #print(name, param, param.requires_grad)
        if name == 'decode_step.weight_ih_l0':
            param.requires_grad = False
        if name == 'decode_step.weight_hh_l0':
            param.requires_grad = False
        if name == 'decode_step.bias_ih_l0':
            param.requires_grad = False'''

    sentence_optimizer = torch.optim.Adam(list(sentence_decoder.parameters())
                                          + list(encoder.parameters()),
                                          lr=args.sentence_lr)  
    word_optimizer = torch.optim.Adam(list(word_decoder.parameters()), lr=args.word_lr,
                                      weight_decay=args.word_weight_decay)
    sentence_decoder.to(device)
    word_decoder.to(device)
    encoder.to(device)
    
    #print('IMAGE ENCODER', encoder)
    #print('SENTENCE DECODER', sentence_decoder)
    #print('WORD DECODER', word_decoder)
    #print('SENTENCE OPTIMIZER', sentence_optimizer)
    #print('WORD OPTIMIZER', word_optimizer)

    epochs_since_improvement = 0

    # Loss functions
    criterion_sent = nn.NLLLoss().to(device)
    criterion_word = nn.CrossEntropyLoss(ignore_index=0).to(device)

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
        print('Training...')
        this_epoch_word = forward_pass(data_loader=train_loader,
                                       encoder=encoder,
                                       sentence_decoder=sentence_decoder,
                                       word_decoder=word_decoder,
                                       criterion_word=criterion_word,
                                       encoder_optimizer=encoder_optimizer,
                                       sentence_optimizer=sentence_optimizer,
                                       word_optimizer=word_optimizer,
                                       word_to_idx=word_to_idx,
                                       epoch=epoch,
                                       logger=experiment,
                                       args=args,
                                       mode='train')

        experiment.log_metric("train_word_loss", this_epoch_word, step=epoch)

        # One epoch's validation
        print('Validating...')
        val_this_epoch_word = forward_pass(data_loader=val_loader,
                                           encoder=encoder,
                                           sentence_decoder=sentence_decoder,
                                           word_decoder=word_decoder,
                                           criterion_word=criterion_word,
                                           encoder_optimizer=None,
                                           sentence_optimizer=None,
                                           word_optimizer=None,
                                           word_to_idx=word_to_idx,
                                           epoch=epoch,
                                           logger=experiment,
                                           args=args,
                                           mode='validate')


        experiment.log_metric("val_word_loss", val_this_epoch_word, step=epoch)

        train_word_epoch_loss.append(this_epoch_word)
        val_word_epoch_loss.append(val_this_epoch_word)
        plot_loss(train_word_epoch_loss, val_word_epoch_loss, args.exp_num, loss_type='word')

        # Check if there was an improvement
        if initial_best_loss == 0 and initial_best_loss < val_this_epoch_word:
            is_best = True
            initial_best_loss = val_this_epoch_word
        elif initial_best_loss > val_this_epoch_word:
            is_best = True
            initial_best_loss = val_this_epoch_word
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
                        val_this_epoch_word,
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
    densecap_feat_dim = config_parser.get('PARAMS-SENTENCE', 'densecap_feat_dim')
    sentence_lr = config_parser.get('PARAMS-SENTENCE', 'sentence_lr')

    num_layers_wordrnn = config_parser.get('PARAMS-WORD', 'num_layers_wordrnn')
    max_sentences = config_parser.get('PARAMS-WORD', 'max_sentences')
    max_words = config_parser.get('PARAMS-WORD', 'max_words')
    word_decoder_lr = config_parser.get('PARAMS-WORD', 'word_decoder_lr')
    lambda_word = config_parser.get('PARAMS-WORD', 'lambda_word')

    batch_size = config_parser.get('PARAMS-MODELS', 'batch_size')
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

    parser.add_argument('--batch_size', type=int, default=batch_size, help='batch size')
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

    parser.add_argument('--api_key', type=str, default=api_key, help='key for the Comet logger')
    parser.add_argument('--project_name', type=str, default=project_name, help='name of the project')
    parser.add_argument('--workspace', type=str, default=workspace, help='owner of the Comet workspace')

    arguments = parser.parse_args()
    main(arguments)
