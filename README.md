# 01-par-gen

### Summary of the current state:

Type of models which have been trained:
1. Baseline:
- encoder with one linear layer and max pooling over regions to produce image vector,
- decoder consisting of
-- sentence LSTM (input: image vector, previous sentence LSTM hidden state; output: sentence topic vector) and linear layer for predicting end of the paragraph (input: sentence topic vector; output: 0/1)
-- word LSTM (input: sentence topic vector, word embeddings, previous word LSTM hidden state; output: word scores) for sentence generation, output is passed through linear layer with subsequence softmax applied to it in order to get word probabilities
2. Baseline + 2 linear layers are added: the first one expands sentence topic vector, the second one shrinks it back to the input dimension for word LSTM (512 -> 1024 -> 512)
3. Baseline + 2 layer word LSTM, where the first layer is initialised with DenseCap RNN weights (not frozen) + word embeddings are initialised from DenseCap (not frozen)

Various decoding strategies have been implemented (greedy search, beam search, sampling, top-n sampling, nucleus sampling). For some good description of decoding strategies: https://web.stanford.edu/class/cs224n/slides/cs224n-2019-lecture15-nlg.pdf, https://github.com/huggingface/blog/blob/master/notebooks/02_how_to_generate.ipynb

Current investigations:
1. Short draft with intermediate results (to put everything in the same picture)
2. Implementing attention (on image regions? on pooled image representation? on topic vector?)

### Train model

1. Run `preproc.ipynb` on raw paragraphs (for more details, check the notebook) to build paragraphs in the format which is required for the model.\
  p.s. `splits.json` and `dataset_paragraphs_v1.json` are already included in the `data` directory for your convenience.

2. Run `create_input_files.py` to generate files which include word map, image features, captions and their lengths for all three data splits. Note that image features where extracted in advance, and this script simply distributes them between data splits properly.

3. Run `/model/main.py` to start training the model. This script uses `config.ini`, which is found in the root directory, simply change settings in this file to train model with different configurations.

### Generate paragraphs

1. To generate new paragraphs, run `model/generate.py`. This will also calculate evaluation scores, simply uncomment corresponding lines in the script if you want to get the scores (line ~285, BLEU and CIDEr only are calculated at the moment). Generated paragraphs are displayed in the terminal window.

### Visualise (check paragraphs and images)

1. Run `visualise.ipynb`, this will run generation script and it provides you with function to see image for which paragraph has been generated. You will also see both ground-truth and generated paragraphs. How many images to generate paragraphs for, which splits to use, etc. can be changed in `model/generate.py`.

### Experiments

Dataset information:
Visual Genome images with paragraphs, 14579 train set, 2490 validation set.

| Experiment Name      | BLEU-4 | BLEU-3 | BLEU-2 | BLEU-1 | CIDEr |
|----------------------|--------|--------|--------|--------|-------|
|  baseline            |
|  baseline + linear   |
|  baseline + DenseCap | 

