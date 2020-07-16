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
    def __init__(self,
                 hidden_dim,
                 language_dim,
                 vision_dim,
                 attention_dim,
                 use_vision,
                 use_language,
                 use_multimodal):
        super(Attention, self).__init__()
        
        self.use_vision = use_vision
        self.use_language = use_language
        self.use_multimodal = use_multimodal
        
        self.hidden_mapping = nn.Linear(hidden_dim, attention_dim)
        
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=1)
        
        # vision only
        if self.use_vision and not self.use_multimodal:
            self.vision_full_att = nn.Linear(attention_dim, 1)
        # language only
        elif self.use_language and not self.use_multimodal:
            self.language_full_att = nn.Linear(attention_dim, 1)
        # multimodal
        elif self.use_multimodal:
            self.vision_mapping = nn.Linear(vision_dim, attention_dim)
            self.language_mapping = nn.Linear(language_dim, attention_dim)
            self.vision_full_att = nn.Linear(attention_dim, 1)
            self.language_full_att = nn.Linear(attention_dim, 1)
            self.fusion = nn.Linear(512 + 4096, 512)
        
    def forward(self, language, vision, hidden):
        
        hidden = hidden.unsqueeze(1)
        hidden_map = self.hidden_mapping(hidden)

        if self.use_vision and not self.use_multimodal:
            visual_map = vision
            vis_att = self.vision_full_att(self.tanh(hidden_map + visual_map))
            alpha_vision = self.softmax(vis_att)
            alpha_text = None
            vision_weighted_encoding = (vision * alpha_vision).sum(dim=1)
            return vision_weighted_encoding, alpha_text, alpha_vision
        
        if self.use_language and not self.use_multimodal:
            text_map = language
            text_att = self.language_full_att(self.tanh(hidden_map + text_map))
            alpha_text = self.softmax(text_att)
            alpha_vision = None
            text_weighted_encoding = (language * alpha_text).sum(dim=1)
            return text_weighted_encoding, alpha_text, alpha_vision
            
        if self.use_multimodal:
            text_map = language
            visual_map = vision
            text_att = self.language_full_att(self.tanh(hidden_map + text_map))
            vis_att = self.vision_full_att(self.tanh(hidden_map + visual_map))
            alpha_text = self.softmax(text_att)
            alpha_vision = self.softmax(vis_att)
            text_weighted_encoding = (language * alpha_text).sum(dim=1)
            vision_weighted_encoding = (vision * alpha_vision).sum(dim=1)
            multimodal_concat = torch.cat((text_weighted_encoding, vision_weighted_encoding), dim=1)
            fused_vector = self.tanh(self.fusion(multimodal_concat))
            return fused_vector, alpha_text, alpha_vision


class RegPool(nn.Module):
    
    def __init__(self, args):
        super(RegPool, self).__init__()
        self.batch_size = args.batch_size
        self.feat_dim = args.densecap_feat_dim
        self.num_boxes = args.num_boxes
        self.hidden_size = args.hidden_size
        self.use_attention = args.use_attention
        self.use_language = args.use_language
        self.use_vision = args.use_vision
        self.use_multimodal = args.use_multimodal
        
        if (self.use_vision and not self.use_language and not self.use_multimodal) or \
           (self.use_vision and not self.use_language and self.use_attention):
            self.vision_mapping = nn.Linear(self.feat_dim, self.hidden_size)
            self.vision_mapping.apply(self.__init_weights)
            
        elif (self.use_language and not self.use_vision and not self.use_multimodal) or \
             (self.use_language and not self.use_vision and self.use_attention):
            self.language_mapping = nn.Linear(self.hidden_size, self.hidden_size)
            self.language_mapping.apply(self.__init_weights)
            
        else:
            self.vision_mapping = nn.Linear(self.feat_dim, self.hidden_size)
            self.language_mapping = nn.Linear(self.hidden_size, self.hidden_size)
            self.vision_mapping.apply(self.__init_weights)
            self.language_mapping.apply(self.__init_weights)
            
    def __init_weights(self, m):
        if type(m) == nn.Linear: 
            nn.init.xavier_uniform_(m.weight)
            nn.init.constant_(m.bias, 0.0)  
            
    def forward(self, vision, language, phrase_lengths):
        if vision.shape[0] != self.batch_size:
            self.batch_size = vision.shape[0]
        language_normalised = torch.zeros(self.batch_size,
                                   self.num_boxes,
                                   self.hidden_size).to(device)
        for img_num, image in enumerate(language):
            # 50 x 15 x 512
            for reg_num, region in enumerate(image):
                # 15 x 512
                this_phrase_length = phrase_lengths[img_num, reg_num]
                summed_phrase = torch.sum(region, 0)
                # 512
                mean_phrase = summed_phrase.div(this_phrase_length)
                language_normalised[img_num, reg_num, :] = mean_phrase
        
        if (self.use_vision and not self.use_language and not self.use_multimodal) or \
           (self.use_vision and not self.use_language and self.use_attention):
            visual_map = self.vision_mapping(vision)
            language_map = language_normalised
        elif (self.use_language and not self.use_vision and not self.use_multimodal) or \
             (self.use_language and not self.use_vision and self.use_attention):
            language_map = self.language_mapping(language_normalised)
            visual_map = vision
        else:
            visual_map = self.vision_mapping(vision)
            language_map = self.language_mapping(language_normalised)
        return language_map, visual_map
    

class SentenceRNN(nn.Module):
    def __init__(self, args):
        super(SentenceRNN, self).__init__()
        self.batch_size = args.batch_size
        self.max_sentences = args.max_sentences
        self.num_regions = args.num_boxes
        self.feat_dim = args.densecap_feat_dim
        self.hidden_size = args.hidden_size
        self.use_attention = args.use_attention
        self.use_vision = args.use_vision
        self.use_language = args.use_language
        self.use_multimodal = args.use_multimodal
        
        if self.use_attention:
            self.non_lin = nn.ReLU()
            self.attention = Attention(self.hidden_size,
                                       self.hidden_size,
                                       self.feat_dim,
                                       self.hidden_size,
                                       self.use_vision,
                                       self.use_language,
                                       self.use_multimodal)
            self.sentence_rnn = nn.LSTMCell(input_size=self.hidden_size, hidden_size=self.hidden_size)
            self.f_beta = nn.Linear(self.hidden_size, self.hidden_size)
            self.sigmoid = nn.Sigmoid()
            
        elif not self.use_attention:
            if (self.use_vision and self.use_language) or self.use_multimodal:
                self.sentence_rnn = nn.LSTM(input_size=self.hidden_size*2, hidden_size=self.hidden_size, batch_first=True)
            else:
                self.sentence_rnn = nn.LSTM(input_size=self.hidden_size, hidden_size=self.hidden_size, batch_first=True)

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

    def forward(self, language, vision,
                topic_previous, hidden_previous, cell_previous,
                paragraph_lengths, sorting_order, caplens):
        
        if vision.shape[0] != self.batch_size:
            self.batch_size = vision.shape[0]
                
        language = language.unsqueeze(1).expand(-1, self.max_sentences, -1, -1)
        vision = vision.unsqueeze(1).expand(-1, self.max_sentences, -1, -1)
        
        if self.use_attention:
            # attention on previous topics -> learning context model
            hidden_previous = hidden_previous.squeeze(0).to(device) # batch_size x hidden_size
            cell_previous = cell_previous.squeeze(0).to(device) # batch_size x hidden_size
            hiddens_all = torch.zeros(self.batch_size, self.max_sentences, self.hidden_size).to(device)
            alphas = torch.zeros(self.batch_size, self.max_sentences, self.num_regions).to(device)
            
            for step in range(language.shape[1]):
                non_zero_idxs = caplens[:, step] > 0
                if (non_zero_idxs == 0).sum(dim=0) == self.batch_size:
                    break
                batch_size_step = sum([l > step for l in paragraph_lengths])
                
                attention_topic, alpha_text, alpha_vision = self.attention(language[:batch_size_step, step, :, :], 
                                                                   vision[:batch_size_step, step, :, :],
                                                                   hidden_previous[:batch_size_step])
                
                gate = self.sigmoid(self.f_beta(hidden_previous[:batch_size_step]))
                context = gate * attention_topic
                
                hidden_previous, cell_previous = self.sentence_rnn(context[:batch_size_step],
                                                                   (hidden_previous[:batch_size_step],
                                                                    cell_previous[:batch_size_step]))
                
                hiddens_all[:, step, :][:batch_size_step] = hidden_previous
                #alphas[:, step, :][:batch_size_step] = alpha
        else:
            # max-pooling instead of attention
            if self.use_vision and not self.use_language and not self.use_multimodal:
                sentence_input_vector = torch.max(vision, dim=2).values
            elif self.use_language and not self.use_vision and not self.use_multimodal:
                sentence_input_vector = torch.max(language, dim=2).values
            else:
                vision_input = torch.max(vision, dim=2).values
                language_input = torch.max(language, dim=2).values
                sentence_input_vector = torch.cat((language_input, vision_input), dim=2)
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
        self.vocab_size = args.vocab_size
        
        self.decode_step = nn.LSTM(input_size=self.hidden_size, hidden_size=self.hidden_size,
                                   batch_first=True)
        self.linear = nn.Linear(self.hidden_size, self.vocab_size)

        self.embeddings = nn.Embedding(self.vocab_size, self.hidden_size)
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
