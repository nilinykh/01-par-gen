import sys
import time
import json
import pathlib
import ast

from torch import nn
import torch.backends.cudnn as cudnn
import torch.nn.functional as F

from evalfunc.bleu.bleu import Bleu
from evalfunc.rouge.rouge import Rouge
from evalfunc.cider.cider import Cider
from evalfunc.meteor.meteor import Meteor
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

                references_batch = dict()
                hypotheses_batch = dict()

                #densecap_captions = [ast.literal_eval(elem) for elem in densecap_captions]
                imgs = imgs.to(device)
                caps = caps.to(device)
                caplens = caplens.to(device)
                image_ids = image_ids.to(device)

                #if args.with_densecap_captions:
                #    phrase_embeddings = densecap_to_embeddings(densecap_captions, word_to_idx, dc_embeddings)
                #else:
                #    phrase_embeddings = None

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

                #scores_all = torch.zeros(imgs.shape[0], args.max_sentences, args.max_words + 2, vocab_size).to(device)
                #targets_all = torch.zeros(imgs.shape[0], args.max_sentences, args.max_words + 2).to(device)

                h_sent = torch.zeros(args.num_layers_sentencernn, imgs.shape[0], args.hidden_size)
                c_sent = torch.zeros(args.num_layers_sentencernn, imgs.shape[0], args.hidden_size)
                h_word = torch.zeros(args.num_layers_wordrnn, imgs.shape[0], args.hidden_size)
                c_word = torch.zeros(args.num_layers_wordrnn, imgs.shape[0], args.hidden_size)

                #eos_predicted = torch.zeros(imgs.shape[0], args.max_sentences).to(device)

                for sent_num in range(args.max_sentences):

                    p_source, topic, ht_sent, ct_sent = sentence_decoder(imgs,
                                                                         (h_sent, c_sent))

                    h_sent = ht_sent
                    c_sent = ct_sent

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

                    #print('CAPTIONS ORIG', current_captions)
                    #print(current_caplens)

                    sorted_scores,\
                    sorted_caps,\
                    sorted_caplens,\
                    _,\
                    ht_word, ct_word = word_decoder(topic,
                                                    current_captions,
                                                    current_caplens,
                                                    imgs.shape[0],
                                                    (h_word, c_word))

                    h_word = ht_word
                    c_word = ct_word

                    #print('sort ind', sort_ind)
                    #print('CAPTIONS SORTED', caps_decoder)

                    if args.topic_hidden:
                        # sorted targets here as well as caps (!!!) - control for it
                        targets = caps_decoder[:, 1:max_seq_length]
                    else:
                        #targets = caps_decoder[:, :max_seq_length]
                        sorted_targets = pack_padded_sequence(sorted_caps,
                                                              sorted_caplens,
                                                              batch_first=True)[0]
                    #print(scores.shape)
                    #print(scores_all.shape)

                    #scores = scores[sort_ind]
                    #targets = targets[sort_ind]

                    #print('SORTED BACK T', targets)

                    #if args.topic_hidden:
                    #    scores_all[:scores.shape[0], sent_num, :max_seq_length-1, :] = scores
                    #    targets_all[:targets.shape[0], sent_num, :max_seq_length-1] = targets
                    #else:
                    #    scores_all[:scores.shape[0], sent_num, :max_seq_length, :] = scores
                    #    targets_all[:targets.shape[0], sent_num, :max_seq_length] = targets

                    wordrnn_loss = criterion_word(sorted_scores, sorted_targets) * args.lambda_word
                    #wordrnn_loss = criterion_word(scores.permute(0, 2, 1), targets) * args.lambda_word

                    sentence_loss += sentrnn_loss
                    word_loss += wordrnn_loss


                batch_time.update(time.time() - start)
                start = time.time()

                loss += sentence_loss
                loss += word_loss

                this_epoch_sentence += sentence_loss.item() / args.lambda_sentence
                this_epoch_word += word_loss.item() / args.lambda_word
                this_epoch_loss += (this_epoch_sentence + this_epoch_word)

                #scores_all_copy = scores_all.clone()
                #scores_all_copy = scores_all
                batch_time.update(time.time() - start)
                start = time.time()

                if i % int(args.print_freq) == 0:
                    print('Validation: [{0}/{1}]\t'
                          'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                          'Loss {loss:.4f}\t'.format(i, len(val_loader), batch_time=batch_time,
                                                     loss=loss.item()))

                # GENERATE SENTENCES UNTIL THE END TOKEN IS PICKED
                # Get references

                '''for single_paragraph in range(targets_all.shape[0]):
                    img_caps = targets_all[single_paragraph].tolist()
                    img_captions = list(
                        map(lambda c: [w for w in c if w not in {word_to_idx['<start>'], word_to_idx['<pad>']}],
                            img_caps))  # remove <start> and pads
                    paragraph_text = []
                    for sent in img_captions:
                        this_sent_text = [idx_to_word[w] for w in sent]
                        paragraph_text.append(' '.join(this_sent_text))
                    references_batch[image_ids[single_paragraph].item()] = paragraph_text

                # Get hypotheses
                # batch size x num sents x num words x vocab size
                scores_probs = F.softmax(scores_all_copy, dim=3)
                preds = torch.argmax(scores_probs, dim=3)

                for predicted_paragraph in range(preds.shape[0]):
                    par_preds = preds[predicted_paragraph].tolist()
                    eos = eos_predicted[predicted_paragraph]
                    par_text = []
                    for sent_num, sent in enumerate(par_preds):
                        if eos[sent_num].item() > 0.5:
                            break
                        this_sentence = sent
                        # removed [1:] for now
                        this_sentence_text = [idx_to_word[w] for w in this_sentence]
                        if '<end>' in this_sentence_text:
                            end_token_loc = this_sentence_text.index('<end>') + 1
                            this_sentence_text = this_sentence_text[:end_token_loc]
                        par_text.append(' '.join(this_sentence_text))
                    hypotheses_batch[image_ids[predicted_paragraph].item()] = par_text

                assert len(references_batch.keys()) == len(hypotheses_batch.keys())

                pathlib.Path(f'./generation_validate').mkdir(parents=True, exist_ok=True)

                with open(f'./generation_validate/hyp_{args.exp_num}.json', 'w') as f:
                    json.dump(hypotheses_batch, f)
                with open(f'./generation_validate/ref_{args.exp_num}.json', 'w') as f:
                    json.dump(references_batch, f)'''
                #print(hypotheses_batch)
                #print()

                # EVALUATION
                #scorers = [
                #(Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
                #(Cider('coco-val-df'), "CIDEr"),
                #]
                #score = []
                #method = []
                #for scorer, method_i in scorers:
                #    score_i, _ = scorer.compute_score(references_batch, hypotheses_batch)
                #    score.extend(score_i) if isinstance(score_i, list) else score.append(score_i)
                #    method.extend(method_i) if isinstance(method_i, list) else method.append(method_i)

                #score_dict = dict(zip(method, score))
                #print(score_dict)

    return this_epoch_loss, this_epoch_sentence, this_epoch_word
