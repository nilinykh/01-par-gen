Nikolai


## 2020-03-27

  - Passing the LSTM cell state of the last word of the previous sentence as an input to the sentence LSTM makes a difference in predicting whether this is the last sentence
  - The semantics of the previous sentence (word-embeddings)
  - the effects of the batch size on training
  - The curiousc case of loss:
    * validation loss is initially low, after a few epochs is equal and then subseqwuently slighty higher but still following training loss and decreasing; we cut off training when the validation loss increases
    * large batch sizes require more training, but the overall loss is lower and overfitting happens later; large batch sizes are conservative in updates but they nonetheless achieve overall lower loss
    * experiment: start training with 128 BS and then later increase the batch size; and then increase the BS to 256; large BS as fine tuning? S's expectation is that this will only speed up the training; N's expectation is that we would actually reach lower validation; compare where BS is constant, e.g. 256 with a case where BS is first 128 and then switched to 256; a case of fine tuning
    * experiment: represent dense-cap captions as a document topic vector; NAACL in New Orleans a keynote on poetry egenration; a best paper on story generation; both papers some represnettaion; having denscap texts summarised as a vector would tell us what is interesting in the picture; attention on what is interesting to describe in this picture for humans;
    * pass the densacap through some sentence embeddings encoder; pre-trained embeddings for dense-caps, encode densecaps to a semantics vector which represents the attention through language in the model
    
  


## 2020-03-20

2020-03-20-generation-model.png

  - loss: removed the loss on the padding
  - removed the vector that we considered to be the topic vector
  - diagram showing the model
    * connect the hidden state of the word LSTM as an input to the sentence LSTM as a topic encoder
  - What is the relevant thing we want to say about the image?
    * Currently, the module is based only image features
    * What do want to say about the image?
    * Use the last state of the word LSTM to encode the semantics of this sentence and concatenate this with the sentence LSTM as a topic of the previous sentence (currently the state of the last word LSTM is passed in as the hidden state of the first word of the next sentence
    * What dependencies do we want to encode: each sentence should represent an interesting sepearate topic; but there should also be dependencies between the sentneces, e.g. co-reference
    * Try the current model vs the sentence topic model
    * Then next xet of experiments, also include a topic model from the descriptions from the densecap
  - Document experiments and their results on Github as Wiki
  - ISP study plan
  - Next
    * Daily catchup meetings at 10:00 - 10:15
    * Meet on Tuesday


## 2020-02-26

  - Summary of the model
  - https://github.com/nilinykh/im2p_new
    * Resnet
    * Initialise the RNN with the image or 0s; if 0s then use image as an input; the zero initialisation makes more sense; image as an inout to every word (Karpathy says this reduces påerformance); currently 0 initialisation; Glorot initialisation
    * comet.ml
    * How to compare BLEU sentences; a generates sentence against every sentence
  - Datasets
    * Stanford
    * Your dataset
  - Densecap vs Resnet
  - Different evaluation scores
  - Generated sentences
  - https://cs.stanford.edu/people/ranjaykrishna/im2p/index.html
  - 
  


## 2020-02-18

  - Coling workshops: *SEM, Lantern
  - INLG paper on object linking
  - A Hierarchical Approach for Generating Descriptive Image Paragraphs: https://cs.stanford.edu/people/ranjaykrishna/im2p/index.html
    * DenseCap (Karpathy, small captions of regions), alternatives fasterRNNs, Yolo
    * Recurrent Topic-Transition GAN for Visual Paragraph Generation, Xiaodan Liang, Zhiting Hu, Hao Zhang, Chuang Gan, Eric P. Xing: https://arxiv.org/abs/1703.07022
  - Todo
    * what objects are most relevant; prioritise detected objects and include them in the
  - M. Tanti and A. C. K. Gatt. Quantifying the amount of visual information used by neural caption generators. In Proceedings of the 1st Workshop on Shortcomings in Vision and Language (SiVL’18), ECCV, 2018.
  - M. Tanti, A. Gatt, and K. P. Camilleri. Where to put the image in an image caption generator. Natural Language Engineering, 24(3):467–489, 2018.
  - A. Ramisa, J. Wang, Y. Lu, E. Dellandrea, F. Moreno-Noguer, and R. Gaizauskas. Combining geometric, textual and visual features for predicting prepositions in image descriptions. In Proceedings of the 2015 Conference on Empirical Methods in Natural Language Processing, pages 214–220, Lisbon, Portugal, 7–21 September 2015. Association for Computational Linguistics.

    
