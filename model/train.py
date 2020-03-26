
import sys
import time
import torch.backends.cudnn as cudnn

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
                num_pixels = imgs.size(1)
                imgs = imgs.mean(dim=1)

            caplens_f, init_inx = caplens_eos(caplens, args.max_sentences)

            loss = 0
            sentence_loss = 0
            word_loss = 0

            h_sent = torch.zeros(args.num_layers_sentencernn, imgs.shape[0], args.hidden_size)
            c_sent = torch.zeros(args.num_layers_sentencernn, imgs.shape[0], args.hidden_size)
            h_word = torch.zeros(args.num_layers_wordrnn, imgs.shape[0], args.hidden_size)
            c_word = torch.zeros(args.num_layers_wordrnn, imgs.shape[0], args.hidden_size)
            #print('HERE')
            #print(h_sent, c_sent)
            #print(h_word, c_word)
            #print()

            for sent_num in range(args.max_sentences):

                p_source, topic, h_sent, c_sent = sentence_decoder(imgs, (h_word, c_word))
                #(h_word[-1].unsqueeze(0), c_word[-1].unsqueeze(0))

                p_target = torch.LongTensor(caplens_f[init_inx].long().squeeze(1)).to(device)
                p_target = p_target.type_as(p_source)
                sentrnn_loss = criterion_sent(p_source, p_target) * args.lambda_sentence
                init_inx += 1
                
                #print(p_source)
                #print(p_target)
                #print(sentrnn_loss)

                # WordRNN
                current_captions = caps[:, sent_num, :]
                current_caplens = caplens.squeeze(1)[:, sent_num]
                max_seq_length = current_caplens[torch.argmax(current_caplens, dim=0)]
                
                #print(current_captions)
                #print(current_caplens)
                #print(topic)
                
                # ignore empty, non-existing sentences in calculations
                #nonzero_indices = (current_caplens!=0).nonzero()
                #nonzero_indices = nonzero_indices.flatten()
                #print(nonzero_indices)
                #current_captions = torch.index_select(current_captions, 0, nonzero_indices)
                #current_caplens = torch.index_select(current_caplens, 0, nonzero_indices)
                #topic = torch.index_select(topic, 0, nonzero_indices)
                
                #print(current_captions)
                #print(current_caplens)
                #print(topic)
                
                #if nonzero_indices.shape[0] == 0:
                    
                #    sentence_loss += torch.mean(sentrnn_loss)
                #    word_loss += 0

                #else:
                    
                scores,\
                _,\
                caps_decoder,\
                _,\
                h_word, c_word = word_decoder(topic, current_captions, current_caplens, imgs.shape[0], (h_word, c_word))

                #print(max_seq_length)
                if args.topic_hidden:
                    targets = caps_decoder[:, 1:max_seq_length]
                else:
                    targets = caps_decoder[:, :max_seq_length]
                #print(targets)
                wordrnn_loss = criterion_word(scores.permute(0, 2, 1), targets) * args.lambda_word

                #print(scores, scores.shape)
                #print(targets, targets.shape)
                #print(wordrnn_loss)

                sentence_loss += torch.mean(sentrnn_loss)
                word_loss += torch.mean(wordrnn_loss)

                    #print()
                    
                #print('NOW HERE')
                #print(h_sent, c_sent)
                #print(h_word, c_word)
                #print()

            batch_time.update(time.time() - start)
            start = time.time()

            loss += sentence_loss
            loss += word_loss

            this_epoch_sentence += sentence_loss
            this_epoch_word += word_loss
            this_epoch_loss += loss

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
    return this_epoch_loss.item()/len(train_loader),\
           this_epoch_sentence.item()/len(train_loader),\
           this_epoch_word.item()/len(train_loader)
