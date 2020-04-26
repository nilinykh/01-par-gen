import sys
import torch
from torch import nn
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torchvision
from torch.autograd import Variable
import os
import pickle

device = torch.device("cuda")
cudnn.benchmark = True

sys.path.append('/home/xilini/par-gen/01-par-gen')

'''
image encoder model (paragraph generation)
'''

import numpy as np
from utils import *
import torch
from torch import nn

class RegPool(nn.Module):
    '''
    image representation (aka DenseCap encoder) via pooling its regions into the single vector
    following Krause 2016 paragraph paper
    '''
    def __init__(self, args):

        super(RegPool, self).__init__()
        self.batch_size = args.batch_size
        self.num_boxes = args.num_boxes
        self.feats_dim = args.densecap_feat_dim
        self.project_dim = args.pooling_dim

        if args.feature_linear:
            self.linear_transformation = True
            self.projection_matrix = nn.Linear(self.feats_dim, self.project_dim)
            self.__init_weights()
        elif not args.feature_linear:
            self.linear_transformation = False

        if args.bn:
            self.bn = nn.BatchNorm1d(self.project_dim, momentum=0.9)
        elif not args.bn:
            self.bn = False

    def __init_weights(self):
        nn.init.normal_(self.projection_matrix.weight)
        nn.init.constant_(self.projection_matrix.bias, 0.0)

    def forward(self, features):
        """
        :input: images, a tensor of dimension (batch_size, num_boxes, feats_dim)
        :return: pooled vector
        """
        #features = nn.Dropout()(features)
        if self.linear_transformation:
            project_vec = nn.LeakyReLU()(self.projection_matrix(features))
        else:
            project_vec = features
        project_vec_all = torch.max(project_vec, 1).values
        # L2 normalisation of features is below to test for future
        project_vec_all = F.normalize(project_vec_all, p=2, dim=1)
        if self.bn:
            project_vec_all = self.bn(project_vec_all)
        return project_vec_all



class Encoder(nn.Module):
    """
    Encoder for images,
    uses pretrained ResNet-152 to extract features from the whole image
    """

    def __init__(self, encoded_image_size=14):
        super(Encoder, self).__init__()
        self.enc_image_size = encoded_image_size
        resnet = torchvision.models.resnet152(pretrained=True)
        # Remove linear and pool layers
        modules = list(resnet.children())[:-2]
        self.resnet = nn.Sequential(*modules)
        # Resize image to fixed size to allow input images of variable size
        self.adaptive_pool = nn.AdaptiveAvgPool2d((encoded_image_size, encoded_image_size))
        self.fine_tune()

    def forward(self, images):
        """
        Forward propagation.
        :param images: images, a tensor of dimensions (batch_size, 3, image_size, image_size)
        :return: encoded images
        """
        out = self.resnet(images)  # (batch_size, 2048, image_size/32, image_size/32)
        out = self.adaptive_pool(out)  # (batch_size, 2048, encoded_image_size, encoded_image_size)
        out = out.permute(0, 2, 3, 1)  # (batch_size, encoded_image_size, encoded_image_size, 2048)
        return out

    def fine_tune(self, fine_tune=True):
        """
        Allow or prevent the computation of gradients
        for convolutional blocks 2 through 4 of the encoder.
        :param fine_tune: Allow?
        """
        for param in self.resnet.parameters():
            param.requires_grad = False
        # If fine-tuning, only fine-tune convolutional blocks 2 through 4
        for layer in list(self.resnet.children())[5:]:
            for param in layer.parameters():
                param.requires_grad = fine_tune


class SentenceRNN(nn.Module):
    '''
    sentence rnn module
    :param 1: batch size
    :param 2: hidden size
    :param 3: number of layers
    :param 4: pooling dimension of topic vector
    :param 5: input dimension of visual features
    :param 6: number of classes for logistic regression (continue/stop)
    '''
    def __init__(self, args):
        self.batch_size = args.batch_size
        self.hidden_size = args.hidden_size
        self.num_layers_sentencernn = args.num_layers_sentencernn
        self.pooling_dim = args.pooling_dim
        self.eos_classes = args.eos_classes
        self.embed_size = args.embed_dim
        super(SentenceRNN, self).__init__()

        if args.encoder_type == 'resnet512':
            self.feat_dim = args.resnet512_feat_dim
        elif args.encoder_type == 'densecap':
            if args.feature_linear:
                self.feat_dim = args.pooling_dim
            else:
                self.feat_dim = 4096
        else:
            raise Exception('Unknown encoder type.')

        self.sentence_rnn = nn.LSTM(input_size=self.feat_dim, hidden_size=self.hidden_size,
                                    num_layers=self.num_layers_sentencernn, batch_first=True)
        self.logistic = nn.Linear(self.hidden_size, self.eos_classes)
        self.non_lin = nn.LeakyReLU()

        if args.use_fc:
            if args.use_fc2:
                self.fc1 = nn.Linear(self.hidden_size, self.pooling_dim)
                if args.topic_hidden:
                    self.fc2 = nn.Linear(self.pooling_dim, self.pooling_dim)
                else:
                    self.fc2 = nn.Linear(self.pooling_dim, self.hidden_size)
                self.use_fc2 = True
                self.use_fc = True
            else:
                self.fc1 = nn.Linear(self.hidden_size, self.pooling_dim)
                self.use_fc2 = False
                self.use_fc = True

            if args.sentence_nonlin:
                self.use_non_lin = True
                self.non_lin = nn.SELU()
            else:
                self.use_non_lin = False
            self.no_fc = False

        else:
            self.no_fc = True
            self.use_fc = False
            self.use_fc2 = False

        if args.dropout_stopping != 0:
            print('Stopping: Linear -> LeakyReLU -> Dropout')
            self.dropout_stopping = nn.Dropout(p=args.dropout_stopping)
        elif args.dropout_stopping == 0:
            self.dropout_stopping = False
        self.__init_weights()

    def __init_weights(self):
        nn.init.normal_(self.logistic.weight)
        nn.init.constant_(self.logistic.bias, 0.0)
        if self.use_fc:
            if self.use_fc2:
                nn.init.normal_(self.fc1.weight)
                nn.init.normal_(self.fc2.weight)
                nn.init.constant_(self.fc1.bias, 0.0)
                nn.init.constant_(self.fc2.bias, 0.0)
            else:
                nn.init.normal_(self.fc1.weight)
                nn.init.constant_(self.fc1.bias, 0.0)
        else:
            print('No FC layers are used for topic modelling')
        for name, param in self.sentence_rnn.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0.0)
            elif 'weight' in name:
                nn.init.normal_(param)

    def forward(self, pooling_vector, states):
        '''
        forward topic vector and output
        end of the paragraph probability,
        topic of the sentence to be fed to wordRNN
        '''

        if pooling_vector.shape[0] != self.batch_size:
            self.batch_size = pooling_vector.shape[0]

        h, c = states
        h = h.to(device)
        c = c.to(device)

        pooling_vector = pooling_vector.unsqueeze(1)
        out, (h, c) = self.sentence_rnn(pooling_vector, (h, c))
        prob = self.non_lin(self.logistic(out))
        prob = prob.squeeze(1).squeeze(1)

        if self.use_fc:
            if self.use_fc2:
                topic = self.fc2(nn.ReLU()(self.fc1(out)))
            else:
                topic = self.non_lin(self.fc1(out))
        if self.no_fc:
            topic = out

        return prob, topic, h, c



class WordRNN(nn.Module):
    '''
    word prediction for the current sentence,
    uses topic vector and embedding of the previous word,
    unrolled for X times (depending on the sentence length)
    '''

    def __init__(self, args):
        self.batch_size = args.batch_size
        self.hidden_size = args.hidden_size
        self.embed_dim = args.embed_dim
        self.num_layers_wordrnn = args.num_layers_wordrnn
        self.pooling_dim = args.pooling_dim
        self.max_sentences = args.max_sentences
        self.max_words = args.max_words
        self.dropout_rate = args.wordlstm_dropout
        self.word_dropout = args.word_dropout
        self.vocab_size = args.vocab_size
        self.rnn_hidden_init = args.rnn_hidden_init
        super(WordRNN, self).__init__()

        if args.embeddings_pretrained:
            print('Using pretrained embeddings...')
            pre_trained = self.__load_pretrained_embeddings(args.densecap_path + 'densecap_emb.pt')
            self.embeddings = nn.Embedding.from_pretrained(pre_trained, freeze=args.freeze)
        else:
            self.embeddings = nn.Embedding(self.vocab_size, self.embed_dim)
            self.__init_embeddings()
            self.__fine_tune_embeddings(fine_tune=True)

        self.rnn_arch = args.rnn_arch
        if self.rnn_arch == 'LSTM':
            if self.num_layers_wordrnn != 1:
                self.decode_step = nn.LSTM(input_size=self.embed_dim, hidden_size=self.hidden_size,
                                           num_layers=self.num_layers_wordrnn, dropout=self.dropout_rate,
                                           batch_first=True)
            else:
                self.decode_step = nn.LSTM(input_size=self.embed_dim, hidden_size=self.hidden_size,
                                           num_layers=self.num_layers_wordrnn,
                                           batch_first=True)
        else:
            raise Exception('Unknown RNN type.')

        self.load_my_state_dict(args.densecap_path + 'rnn_densecap.pkl')

        if args.layer_norm:
            #nn.LayerNorm
            self.layer_norm = nn.LayerNorm(self.hidden_size)
        elif not args.layer_norm:
            self.layer_norm = False

        self.linear = nn.Linear(self.hidden_size, self.vocab_size)
        #self.additional_linear = nn.Linear(self.hidden_size, self.hidden_size)

        #self.load_linear_projection(args.densecap_path + 'projection.pkl')

        if self.word_dropout:
            print('Projection: Linear -> Leaky ReLU -> Dropout')
        #self.linear_dropout = nn.Dropout(p=self.word_dropout)

        if args.topic_hidden:
            self.topic_hidden = True
            self.init_h = nn.Linear(self.pooling_dim, self.hidden_size)
            self.init_c = nn.Linear(self.pooling_dim, self.hidden_size)
        else:
            self.topic_hidden = False

        if args.with_densecap_captions:
            self.with_densecap_captions = True
        elif not args.with_densecap_captions:
            self.with_densecap_captions = False

        #if args.use_fc:
        self.image_to_hidden = nn.Linear(self.pooling_dim, self.hidden_size)
        self.non_lin = nn.ReLU()
        self.im2h = True
        #elif not args.use_fc:
        #    self.im2h = False

    def __init_embeddings(self):
        initrange = 0.1
        self.embeddings.weight.data.uniform_(-initrange, initrange)

    def __fine_tune_embeddings(self, fine_tune=True):
        for param in self.embeddings.parameters():
            param.requires_grad = fine_tune

    def load_linear_projection(self, linear_proj_path):
        print(self.linear.weight)
        with open(linear_proj_path, 'rb') as f:
            linear_proj_densecap = pickle.load(f)
        print(linear_proj_densecap['vocab_projection.weight'], type(linear_proj_densecap['vocab_projection.weight']))
        #self.linear.weight = linear_proj_densecap['vocab_projection.weight']
        for name, param in self.linear.state_dict().items():
            if name == 'weight':
                self.linear.state_dict()['weight'].copy_(linear_proj_densecap['vocab_projection.weight'])
        print(self.linear.weight)

    def load_my_state_dict(self, state_dict_path):
        own_state = self.state_dict()
        with open(state_dict_path, 'rb') as j:
            state_dict = pickle.load(j)
        for name, param in state_dict.items():
            if name == 'word_rnn.0.weight_ih':
                param = param.permute(1, 0)
                own_state['decode_step.weight_ih_l0'].copy_(param[:512].permute(1, 0))
                own_state['decode_step.weight_hh_l0'].copy_(param[512:].permute(1, 0))
            if name == 'word_rnn.0.bias_ih':
                own_state['decode_step.bias_ih_l0'].copy_(param)

        for name1, param1 in own_state.items():
            #print(name1)
            if name1 == 'decode_step.weight_ih_l0':
                param1.requires_grad = True
                #print(param1.requires_grad)
            if name1 == 'decode_step.weight_hh_l0':
                param1.requires_grad = True
            if name1 == 'decode_step.bias_ih_l0':
                param1.requires_grad = True
            else:
                param1.requires_grad = True

        #for name1, param1 in own_state.items():
        #    print(name1, param1.requires_grad)
        #print(own_state)

    @classmethod
    def __load_pretrained_embeddings(cls, weights_matrix):
        """
        load DenseCap pretrained word embeddings
        """
        densecap_embeddings = torch.load(weights_matrix)
        return densecap_embeddings

    def init_hidden_state(self, topic):
        """
        Creates the initial hidden and cell states for the decoder's LSTM based on the encoded images.
        :param encoder_out: encoded images, a tensor of dimension (batch_size, num_pixels, encoder_dim)
        :return: hidden state, cell state
        """
        mean_topic = topic.mean(dim=1)
        h = self.init_h(mean_topic).unsqueeze(0)  # (batch_size, decoder_dim)
        c = self.init_c(mean_topic).unsqueeze(0)
        return h, c


    def forward(self, topic, caps, caplens, new_batch_size, states):
        '''
        decode image features with word vectors and generate captions
        '''

        if self.topic_hidden:
            h, c = self.init_hidden_state(topic)
            caplens = caplens - 1
        else:
            h, c = states

        h = h.to(device)
        c = c.to(device)
        if new_batch_size != self.batch_size:
            self.batch_size = new_batch_size

        caplens, sort_ind = caplens.sort(dim=0, descending=True)
        caps = caps[sort_ind]
        topic = topic[sort_ind]
        
        embeddings = self.embeddings(caps)

        topic = self.non_lin(self.image_to_hidden(topic))
        if not self.topic_hidden:
            embeddings = torch.cat([topic, embeddings], 1)

        packed = pack_padded_sequence(embeddings, caplens, batch_first=True)
        hiddens, (h, c) = self.decode_step(packed, (h, c))
        if self.layer_norm:
            outputs = self.layer_norm(hiddens.data)
        outputs = self.linear(hiddens.data)
        #print(outputs, outputs.shape)
        
        #----WITHOUT TEACHER FORCING (TEST CODE, NOT WORKING ATM)----#
        #topic = self.non_lin(self.image_to_hidden(topic))
        #inputs = topic
        #predicted_caps = torch.zeros(self.batch_size, caplens[0], self.vocab_size)
        #for i in range(caplens[0]):
        #    hiddens, (h, c) = self.decode_step(inputs, (h, c))
        #    outputs = self.linear(hiddens)
        #    scores_probs = F.log_softmax(outputs, dim=2)
        #    if mode == 'val':
        #        predicted_caps[:, i] = scores_probs.squeeze(1)
        #        topv, topi = scores_probs.topk(1)
        #        decoder_input = topi.squeeze(1).detach()
        #        inputs = self.embeddings(decoder_input)
        #    if mode == 'train':
        #        predicted_caps[:, i] = scores_probs.squeeze(1)
        #        inputs = self.embeddings(caps[:, i]).unsqueeze(1)
        #return predicted_caps, caps, caplens, sort_ind, h, c
        
        return outputs, caps, caplens, sort_ind, h, c
