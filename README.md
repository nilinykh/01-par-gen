# 01-par-gen

### Train model

1. Run `preproc.ipynb` on raw paragraphs (for more details, check the notebook) to build paragraphs in the format which is required for the model.\
  p.s. `splits.json` and `dataset_paragraphs_v1.json` are already included in the `data` directory for your convenience.

2. Run `create_input_files.py` to generate files which include word map, image features, captions and their lengths for all three data splits. Note that image features where extracted in advance, and this script simply distributes them between data splits properly.

3. Run `/model/main.py` to start training the model. This script uses `config.ini`, which is found in the root directory, simply change settings in this file to train model with different configurations. Also, during validation intermediate generation results for the last batch are saved in `generation_validate` folder.

### Generate paragraphs

1. To generate new paragraphs, run `model/generate.py`. This will also calculate evaluation scores (BLEU and CIDEr only at the moment). Generated paragraphs are displayed in the terminal window.

### Experiments

Proj-Matrix: linear layer for image features\
BS: batch size\
LR-S: learning rate for the sentence LSTM\
LR-W: learning rate for the word LSTM\
NL-W: number of layers in word LSTM\
D-W: if NL-W is not 1, then dropout is specified\
L-S: lambda for sentence LSTM loss\
L-W: lambda for word LSTM loss\
C-S: clipping for sentence LSTM\
C-W: clipping for word LSTM\
LN-W: layer normalisation for word LSTM\
BN-E: batch normalisation for encoder\
VL-min: minimal validation set loss that has been achieved
VL-min-W: word validation min loss
VL-min-S: sentence validatin min loss

| Exp Num | Feat-Extractor | Proj-Matrix | BS | LR-S | LR-W | NL-W | D-W | L-S | L-W | C-S | C-W | LN-W | BN-E | VL-min | VL-min-W | VL-min-S |
|---|----------------|---------|----|------|------|------|-----|-----|-----|-----|-----|------|------|------|------|------|
|  simple |       DenseCap         |     -    |  64  |   1e-4   |   1e-3   |   1   |  -   |  1   |  1   |  -   |  -  |  -  |  -    |      |      |      |
|   |                |         |    |      |      |      |     |     |     |     |     |      |      |      |
|   |                |         |    |      |      |      |     |     |     |     |     |      |      |      |
