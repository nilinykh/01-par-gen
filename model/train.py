import sys
import time
import torch.backends.cudnn as cudnn

sys.path.append('/home/xilini/par-gen/01-par-gen')
from utils import *

device = torch.device("cuda")
cudnn.benchmark = True

alpha_c = 0.5

criterion = HierarchicalXEntropyLoss(weight_word_loss=1.0)

def forward_pass(data_loader,
                 encoder,
                 sentence_decoder,
                 word_decoder,
                 criterion_word,
                 encoder_optimizer,
                 sentence_optimizer,
                 word_optimizer,
                 word_to_idx,
                 epoch,
                 logger,
                 args,
                 mode):
    
    if mode == 'train':
        encoder.train()
        sentence_decoder.train()
        word_decoder.train()
    elif mode == 'validate':
        encoder.eval()
        sentence_decoder.eval()
        word_decoder.eval()

    # Set timers
    batch_time = AverageMeter()
    data_time = AverageMeter()
    start = time.time()

    this_epoch_word = 0

    #with logger.train():
        
    for batch_num, (image_features, image_ids, caps, caplens, phrase_scores, _, phrase_lengths) in enumerate(data_loader):

        data_time.update(time.time() - start)

        image_features = image_features.to(device)
        image_ids = image_ids.to(device)
        caps = caps.to(device)
        caplens = caplens.squeeze(1).to(device)
        phrase_scores = phrase_scores.to(device)
        phrase_lengths = phrase_lengths.to(device)
        args.batch_size = image_features.shape[0]

        loss = 0
        word_loss = 0

        h_sent = c_sent = h_word = c_word = torch.zeros(1, args.batch_size, args.hidden_size)

        alphas = torch.zeros(args.batch_size, args.max_sentences, args.max_words).to(device)
        #print('alphas', alphas.shape)

        # initial sentence topic is zero topic
        init_topic = torch.zeros(1, args.batch_size, args.hidden_size)

        # lengths of the paragraphs in current minibatch (e.g., 6 sentences, 3 sentence, etc.)
        # note: caplens is the length of sentences in the paragraph with 0 when there is no sentences
        minibatch_lengths = torch.zeros(args.batch_size)
        for sample_num, sample in enumerate(caplens.squeeze(1)):
            unique_values, counts = torch.unique(sample, return_counts=True)
            if 0 in unique_values:
                this_sample_length = counts[1:].sum()
            else:
                this_sample_length = counts.sum()
            minibatch_lengths[sample_num] = this_sample_length

        # we need to look at the topics of sentences which are NON-EMPTY
        sorting_order = torch.zeros(args.batch_size, args.max_sentences).long()
        _, idxs = torch.sort(minibatch_lengths, descending=True)

        image_features = image_features[idxs]
        image_ids = image_ids[idxs]
        caps = caps[idxs]
        caplens = caplens[idxs]
        phrase_scores = phrase_scores[idxs]
        minibatch_lengths = minibatch_lengths[idxs]

        # for word-level prediction scores
        word_rnn_out = []

        # 1. output of the encoder
        language_out, vision_out = encoder(image_features, phrase_scores, phrase_lengths)            
        #print('encoder out', encoder_out, encoder_out.shape)

        # 2. repeat encoder output 6 times
        #features_repeated = encoder_out.unsqueeze(1).expand(-1, args.max_sentences, -1, -1)
        #print('feat repeated', features_repeated, features_repeated.shape)

        # 3. produce 6 different topics
        hiddens, alphas = sentence_decoder(language_out, vision_out,
                                           init_topic,
                                           h_sent,
                                           c_sent,
                                           minibatch_lengths,
                                           sorting_order,
                                           caplens)

        for sent_num in range(args.max_sentences):
            if caplens[0, sent_num] == 0:
                break 
            # 1. pick indexes of existing sentences only
            non_zero_idxs = caplens[:, sent_num] > 0
            # 2. take existing topics based on their indixes
            topic = hiddens[:, sent_num][non_zero_idxs]
            # 3. pick captions and caption lengths for the existing sentences
            current_caps = caps[:, sent_num][non_zero_idxs]
            current_caplens = caplens[:, sent_num][non_zero_idxs]
            # 4. word-level forward pass
            sorted_scores, (h_word, c_word) = word_decoder(topic,
                                                           current_caps,
                                                           current_caplens,
                                                           args.batch_size, non_zero_idxs, (h_word, c_word))
            word_rnn_out.append(sorted_scores)

        # 5. prepare targets
        targets = prepare_hierarchical_targets(args.max_sentences,
                                               caplens,
                                               caps)
        batch_time.update(time.time() - start)
        start = time.time()

        # LOSS
        loss = criterion(word_rnn_out, targets)

        if mode == 'train':
            sentence_optimizer.zero_grad()
            word_optimizer.zero_grad()
            if encoder_optimizer != None:
                encoder_optimizer.zero_grad()
            loss.backward()
            sentence_optimizer.step()
            word_optimizer.step()
            if encoder_optimizer != None:
                encoder_optimizer.step()
                
        this_epoch_word += loss.item()
        
        # Print status
        if batch_num % int(args.print_freq) == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data Load Time {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss:.4f}\t'.format(epoch, batch_num, len(data_loader),
                                             batch_time=batch_time,
                                             data_time=data_time, loss=loss.item()))

    return this_epoch_word  / len(data_loader)