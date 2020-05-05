import sys
import time
import json
import pathlib
import ast

from torch import nn
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence

sys.path.append('/home/xilini/par-gen/01-par-gen')
from utils import *

device = torch.device("cuda")
cudnn.benchmark = True

def validate(val_loader,
             encoder,
             sentence_decoder,
             word_decoder,
             criterion_sent,
             criterion_word,
             logger,
             vocab_size,
             word_to_idx,
             idx_to_word,
             dc_embeddings,
             args):

    '''
    single epoch validation
    '''

    sentence_decoder.eval()
    word_decoder.eval()
    encoder.eval()

    batch_time = AverageMeter()
    start = time.time()

    this_epoch_loss = 0
    this_epoch_sentence = 0
    this_epoch_word = 0

    with logger.validate():
        with torch.no_grad():
            # Batches
            for i, (imgs, image_ids, caps, caplens) in enumerate(val_loader):

                #references_batch = dict()
                #hypotheses_batch = dict()

                imgs = imgs.to(device)
                caps = caps.to(device)
                caplens = caplens.to(device)
                image_ids = image_ids.to(device)

                # Forward prop.
                imgs = encoder(imgs)
                args.batch_size = imgs.shape[0]

                if args.encoder_type == 'resnet512':
                    # Prepare images for sentence decoder
                    imgs = imgs.view(args.batch_size, -1, args.resnet512_feat_dim)
                    #num_pixels = imgs.size(1)
                    imgs = imgs.mean(dim=1)

                caplens_f, init_inx = caplens_eos(caplens, args.max_sentences)

                loss = 0
                sentence_loss = 0
                word_loss = 0

                h_sent = torch.zeros(args.num_layers_sentencernn, imgs.shape[0], args.hidden_size)
                c_sent = torch.zeros(args.num_layers_sentencernn, imgs.shape[0], args.hidden_size)
                h_word = torch.zeros(args.num_layers_wordrnn, imgs.shape[0], args.hidden_size)
                c_word = torch.zeros(args.num_layers_wordrnn, imgs.shape[0], args.hidden_size)

                for sent_num in range(args.max_sentences):

                    p_source, topic, ht_sent, ct_sent = sentence_decoder(imgs, (h_sent, c_sent))

                    h_sent = ht_sent
                    c_sent = ct_sent

                    p_target = torch.LongTensor(caplens_f[init_inx].long().squeeze(1)).to(device)
                    p_target = p_target.type_as(p_source)
                    sentrnn_loss = criterion_sent(p_source, p_target)
                    init_inx += 1

                    #print(p_source)
                    #print(p_target)
                    #print(sentrnn_loss)

                    # WordRNN
                    current_captions = caps[:, sent_num, :]
                    current_caplens = caplens.squeeze(1)[:, sent_num]
                    max_seq_length = current_caplens[torch.argmax(current_caplens, dim=0)]

                    sorted_scores,\
                    sorted_caps,\
                    sorted_caplens,\
                    _,\
                    ht_word, ct_word = word_decoder(topic,
                                                    current_captions,
                                                    current_caplens,
                                                    imgs.shape[0], (h_word, c_word))

                    h_word = ht_word
                    c_word = ct_word

                    if args.topic_hidden:
                        sorted_targets = caps_decoder[:, 1:max_seq_length]
                    else:
                        sorted_targets = pack_padded_sequence(sorted_caps,
                                                              sorted_caplens,
                                                              batch_first=True)[0]

                    wordrnn_loss = criterion_word(sorted_scores, sorted_targets)


                    #sentence_loss += torch.sum(sentrnn_loss) / imgs.shape[0]
                    #word_loss += torch.sum(wordrnn_loss) / imgs.shape[0]

                    sentence_loss += torch.mean(sentrnn_loss)
                    word_loss += torch.mean(wordrnn_loss)

                batch_time.update(time.time() - start)
                start = time.time()

                loss += (sentence_loss * args.lambda_sentence) + (word_loss * args.lambda_word)

                this_epoch_sentence += sentence_loss.item() * args.lambda_sentence
                this_epoch_word += word_loss.item() * args.lambda_word
                this_epoch_loss += (this_epoch_sentence + this_epoch_word)

                batch_time.update(time.time() - start)
                start = time.time()

                if i % int(args.print_freq) == 0:
                    print('Validation: [{0}/{1}]\t'
                          'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                          'Loss {loss:.4f}\t'.format(i, len(val_loader), batch_time=batch_time,
                                                     loss=loss.item()))

    return this_epoch_loss / len(val_loader),\
           this_epoch_sentence / len(val_loader),\
           this_epoch_word / len(val_loader)
