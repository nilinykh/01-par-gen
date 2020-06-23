import os
import sys
import pickle
import numpy as np
from collections import OrderedDict

import torch
from torch import nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.nn.utils.rnn import pack_padded_sequence

#from utils import *

device = torch.device("cuda")
cudnn.benchmark = True

sys.path.append('/home/xilini/par-gen/01-par-gen')


class Attention(nn.Module):
    def __init__(self, encoder_dim, decoder_dim, attention_dim):
        super(Attention, self).__init__()
        self.encoder_att = nn.Linear(encoder_dim, attention_dim)
        self.decoder_att = nn.Linear(decoder_dim, attention_dim)
        self.full_att = nn.Linear(attention_dim, 1)
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, encoder_out, decoder_hidden):
        att1 = self.encoder_att(encoder_out)
        att2 = self.decoder_att(decoder_hidden)
        att = self.full_att(self.tanh(att1 + att2.unsqueeze(1))).squeeze(2)
        alpha = self.softmax(att)
        attention_weighted_encoding = (encoder_out * alpha.unsqueeze(2)).sum(dim=1)
        return attention_weighted_encoding, alpha


class RegPool(nn.Module):
    def __init__(self, args):
        super(RegPool, self).__init__()
        self.batch_size = args.batch_size
        self.num_boxes = args.num_boxes
        self.feats_dim = args.densecap_feat_dim
        self.project_dim = args.pooling_dim
        self.phrase_dim = args.hidden_size
        self.multimodal = args.multimodal
        self.image_pass = nn.Sequential(OrderedDict([('fc1', nn.Linear(self.feats_dim, self.project_dim)),
                                                     ('relu', nn.ReLU())]))
        self.image_pass.apply(self.__init_weights)
        self.non_lin = nn.ReLU()

    def __init_weights(self, m):
        if type(m) == nn.Linear:
            nn.init.xavier_uniform_(m.weight)
            nn.init.constant_(m.bias, 0.0)  
        
    def forward(self, features, phrases):
        lang_vector = torch.mean(phrases, 2) # batch_size x 50 x 512
        vis_vector = self.image_pass(features)
        if self.multimodal:
            multimodal_vector = self.non_lin(torch.mul(vis_vector, lang_vector))
        else:
            multimodal_vector = vis_vector
        return multimodal_vector
    

class SentenceRNN(nn.Module):
    def __init__(self, args):
        super(SentenceRNN, self).__init__()
        self.batch_size = args.batch_size
        self.hidden_size = args.hidden_size
        self.pooling_dim = args.pooling_dim
        self.embed_size = args.embed_dim
        self.phrase_dim = args.hidden_size
        self.feat_dim = args.pooling_dim
        self.use_attention = args.use_attention
        self.max_sentences = args.max_sentences
        self.num_regions = args.num_boxes
        
        if self.use_attention:
            self.sentence_rnn = nn.LSTMCell(input_size=self.phrase_dim*2, hidden_size=self.hidden_size)
            self.non_lin = nn.ReLU()
            self.attention = Attention(self.phrase_dim, self.phrase_dim, self.phrase_dim + self.phrase_dim)
            self.f_beta = nn.Linear(self.phrase_dim, self.hidden_size)
            self.sigmoid = nn.Sigmoid()
            
        elif not self.use_attention:
            self.sentence_rnn = nn.LSTM(input_size=self.phrase_dim, hidden_size=self.hidden_size, batch_first=True)
        
        self.__init_weights()

    def __init_weights(self):
        for name, param in self.sentence_rnn.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0.0)
            elif 'weight' in name:
                nn.init.xavier_uniform_(param)
        if self.use_attention:
            nn.init.xavier_uniform_(self.f_beta.weight)
            nn.init.constant_(self.f_beta.bias, 0.0)

    def forward(self, input_vector, topic_previous, hidden_previous, cell_previous, paragraph_lengths, sorting_order, caplens):
        if input_vector.shape[0] != self.batch_size:
            self.batch_size = input_vector.shape[0]
        if self.use_attention:
            # attention on previous topics -> learning context model
            hidden_previous = hidden_previous.squeeze(0).to(device) # batch_size x hidden_size
            cell_previous = cell_previous.squeeze(0).to(device) # batch_size x hidden_size
            topic_previous = topic_previous.squeeze(0) # concat dimension (batch_size) x hidden_size
            topic_previous = topic_previous.to(device)
            hiddens_all = torch.zeros(self.batch_size, self.max_sentences, self.hidden_size).to(device)
            alphas = torch.zeros(self.batch_size, self.max_sentences, self.num_regions).to(device)
            
            for step in range(input_vector.shape[1]):
                non_zero_idxs = caplens[:, step] > 0
                if (non_zero_idxs == 0).sum(dim=0) == self.batch_size:
                    break
                batch_size_step = sum([l > step for l in paragraph_lengths])
                attention_topic, alpha = self.attention(input_vector[:batch_size_step, step, :, :], hidden_previous[:batch_size_step])
                gate = self.sigmoid(self.f_beta(hidden_previous[:batch_size_step]))
                context = gate * attention_topic
                topic_input = torch.cat((topic_previous[:batch_size_step], context), dim=1)
                hidden_previous, cell_previous = self.sentence_rnn(topic_input[:batch_size_step], (hidden_previous[:batch_size_step], cell_previous[:batch_size_step]))
                new_topic = hidden_previous
                topic_previous[:batch_size_step] = new_topic
                hiddens_all[:, step, :][:batch_size_step] = new_topic
                alphas[:, step, :][:batch_size_step] = alpha
        else:
            # max-pooling instead of attention
            sentence_input_vector = torch.max(input_vector, dim=2).values
            hiddens_all, (h, c) = self.sentence_rnn(sentence_input_vector)
            alphas = None
        return hiddens_all, alphas



class WordRNN(nn.Module):
    '''
    word prediction for the current sentence,
    uses topic vector and embedding of the previous word,
    unrolled for X times (depending on the sentence length)
    '''

    def __init__(self, args):
        super(WordRNN, self).__init__()
        self.batch_size = args.batch_size
        self.hidden_size = args.hidden_size
        self.embed_dim = args.embed_dim
        self.num_layers_wordrnn = args.num_layers_wordrnn
        self.word_dropout = args.word_dropout
        self.vocab_size = args.vocab_size
        
        self.decode_step = nn.LSTM(input_size=self.embed_dim, hidden_size=self.hidden_size,
                                   num_layers=self.num_layers_wordrnn, dropout=self.word_dropout,
                                   batch_first=True)
        self.linear = nn.Linear(self.hidden_size, self.vocab_size)
        
        if args.embeddings_pretrained:
            print('Using pretrained embeddings...')
            pre_trained = self.__load_pretrained_embeddings(args.densecap_path + 'densecap_emb.pt')
            self.embeddings = nn.Embedding.from_pretrained(pre_trained, freeze=args.freeze)
        else:
            print('Training embeddings from scratch...')
            self.embeddings = nn.Embedding(self.vocab_size, self.embed_dim)
            self.__init_embeddings(fine_tune=True)
            
    def __init_embeddings(self, fine_tune=True):
        initrange = 0.1
        self.embeddings.weight.data.uniform_(-initrange, initrange)
        for param in self.embeddings.parameters():
            param.requires_grad = fine_tune
        for name, param in self.decode_step.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0.0)
            elif 'weight' in name:
                nn.init.xavier_uniform_(param)

    def __load_pretrained_embeddings(cls, weights_matrix):
        densecap_embeddings = torch.load(weights_matrix)
        return densecap_embeddings

    def forward(self, topic, caps, caplens, new_batch_size, non_zero_idxs, states):
        if new_batch_size != self.batch_size:
            self.batch_size = new_batch_size
        h, c = states
        h = h.to(device)
        c = c.to(device)
        embeddings = self.embeddings(caps)
        embeddings = torch.cat([topic.unsqueeze(1), embeddings], 1)
        packed = pack_padded_sequence(embeddings, caplens, batch_first=True, enforce_sorted=False)
        hiddens, (h, c) = self.decode_step(packed, (h, c))
        outputs = self.linear(hiddens.data)
        return outputs, (h, c)
