
import sys
import time
import torch.backends.cudnn as cudnn
import ast
from torch.nn.utils.rnn import pack_padded_sequence

sys.path.append('/home/xilini/par-gen/01-par-gen')
from utils import *

device = torch.device("cuda")
cudnn.benchmark = True

def train(train_loader,
          encoder,
          sentence_decoder,
          word_decoder,
          criterion_sent,
          criterion_word,
          encoder_optimizer,
          sentence_optimizer,
          word_optimizer,
          word_to_idx,
          dc_embeddings,
          epoch,
          logger,
          args):

    '''
    single epoch training
    '''

    encoder.train()
    sentence_decoder.train()
    word_decoder.train()

    # Set timers
    batch_time = AverageMeter()  # forward prop. + back prop. time
    data_time = AverageMeter()  # data loading time
    start = time.time()

    this_epoch_loss = 0
    this_epoch_sentence = 0
    this_epoch_word = 0

    with logger.train():
        for i, (imgs, image_ids, caps, caplens) in enumerate(train_loader):

            data_time.update(time.time() - start)

            imgs = imgs.to(device)
            caps = caps.to(device)
            caplens = caplens.to(device)
            image_ids = image_ids.to(device)

            imgs = encoder(imgs)

            if args.encoder_type == 'resnet512':
                # Prepare images for sentence decoder
                imgs = imgs.view(imgs.shape[0], -1, args.resnet512_feat_dim)
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

            #print('START')
            #print(h_sent, c_sent)
            #print(h_word, c_word)

            for sent_num in range(args.max_sentences):
                
                #print(imgs.shape)
                
                p_source, topic, ht_sent, ct_sent = sentence_decoder(imgs, (h_sent, c_sent))
                #(h_word[-1].unsqueeze(0), c_word[-1].unsqueeze(0))
                
                h_sent = ht_sent
                c_sent = ct_sent

                p_target = torch.LongTensor(caplens_f[init_inx].long().squeeze(1)).to(device)
                p_target = p_target.type_as(p_source)
                sentrnn_loss = criterion_sent(p_source, p_target)
                init_inx += 1

                #print('p source', p_source)
                #print('p target', p_target)
                #print('sentence loss', sentrnn_loss)
                #print('h sentence', h_sent)
                #print('c sentence', c_sent)

                # WordRNN
                current_captions = caps[:, sent_num, :]
                current_caplens = caplens.squeeze(1)[:, sent_num]
                max_seq_length = current_caplens[torch.argmax(current_caplens, dim=0)]

                #print('topic', topic)
                #print('current captions', current_captions)
                #print('current caplens', current_caplens)
                #print('current max seq length', max_seq_length)

                sorted_scores,\
                sorted_caps,\
                sorted_caplens,\
                _,\
                ht_word, ct_word = word_decoder(topic,
                                                current_captions,
                                                current_caplens,
                                                imgs.shape[0], (h_word, c_word))

                #print('returned scores', sorted_scores, sorted_scores.shape)
                #print('returned captions', sorted_caps, sorted_caps.shape)
                #print('returned sort ind', sort_ind)
                #print('returned caplens', sorted_caplens)
                
                h_word = ht_word
                c_word = ct_word

                if args.topic_hidden:
                    sorted_targets = sorted_caps[:, 1:max_seq_length]
                else:
                    sorted_targets = pack_padded_sequence(sorted_caps,
                                                   sorted_caplens,
                                                   batch_first=True)[0]

                wordrnn_loss = criterion_word(sorted_scores, sorted_targets)
                
                #print(criterion_word)
                #print(sorted_scores, sorted_scores.shape)
                #print(sorted_targets, sorted_targets.shape)
                #print('scores', scores, scores.shape)
                #print('targets', targets, targets.shape)
                #print('word loss', wordrnn_loss)

                #print('END')
                #print('====================')
                #print()

                #sentence_loss += torch.sum(sentrnn_loss) / imgs.shape[0]
                #word_loss += torch.sum(wordrnn_loss) / imgs.shape[0]
                
                #print(sentrnn_loss, sentrnn_loss.shape)
                #print(wordrnn_loss, wordrnn_loss.shape)
                
                sentence_loss += torch.mean(sentrnn_loss)
                word_loss += torch.mean(wordrnn_loss)
                
                #print('sentence', sentence_loss)
                #print('word', word_loss)

            batch_time.update(time.time() - start)
            start = time.time()

            loss += (sentence_loss * args.lambda_sentence) + (word_loss * args.lambda_word)

            #print('loss', loss)
            this_epoch_sentence += sentence_loss.item() * args.lambda_sentence
            this_epoch_word += word_loss.item() * args.lambda_word
            this_epoch_loss += (this_epoch_sentence + this_epoch_word)

            sentence_optimizer.zero_grad()
            word_optimizer.zero_grad()
            if args.feature_linear:
                encoder_optimizer.zero_grad()

            loss.backward()

            if args.clipping:
                torch.nn.utils.clip_grad_norm_(sentence_decoder.parameters(), args.sent_grad_clip)
                torch.nn.utils.clip_grad_norm_(word_decoder.parameters(), args.word_grad_clip)

            sentence_optimizer.step()
            word_optimizer.step()
            if args.feature_linear:
                encoder_optimizer.step()

            # Print status
            if i % int(args.print_freq) == 0:
                print('Epoch: [{0}][{1}/{2}]\t'
                      'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Data Load Time {data_time.val:.3f} ({data_time.avg:.3f})\t'
                      'Loss {loss:.4f}\t'.format(epoch, i, len(train_loader),
                                                 batch_time=batch_time,
                                                 data_time=data_time, loss=loss.item()))

    # return average loss / sentence loss / word loss for current epoch
    return this_epoch_loss / len(train_loader),\
           this_epoch_sentence / len(train_loader),\
           this_epoch_word / len(train_loader)
