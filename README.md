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

### Examples

![Image 1](https://cs.stanford.edu/people/rak248/VG_100K/2356347.jpg)

Ground truth:
a large building with bars on the windows in front of it . there is people walking in front of the building . there is a street in front of the building with many cars on it .

Greedy:
a large white building with a green roof on the side of it . the man is wearing a black helmet and a black helmet . the road is covered with snow and there is a large green sign . there is a large white building with a red roof on the side . the sky is blue with white clouds in it .

Beam-2:
a man is wearing a park . a man in front of the bus . a man is wearing a shirt and is wearing a helmet . a lot of a hat is standing on the boy . a man in the man is wearing a hat .

Sampling (No Temp):
a bus is going down a street . there are windows on the ground which is rocky . the street is made of mesh cement and metal rails with large gray dotting . there are dandelions flowering other items items on the laps . another sign can be seen hanging on the building .

Sampling (T = 0.4):
a street has many cars on it . the road is paved and there are cars parked in the street . the street is lined with tall green trees and a small white building . there are trees to the left of the street . the trees are green and some are a few green trees .

Sampling (T = 1.2):
the main train cars are broad , red , orange , and white . setting jam cone is a large gray body area . there are electrical wires may hung made of mailbox with open doors on top . black high down suns alley above place , bushes , and a characteristics way there is a pile of snow covered two players next to it where the man on top is steep

Top-N Sampling (T = 0.5, V = 3000):
a city street with a lot of cars . the sign is red and white with green lettering . the road is covered with one of the people riding bikes . the sky is blue with white clouds . the man is holding a skateboard in his hand .

Nucleus Sampling (P = 0.9, T = 0.5):
a large white building with a lot of windows on the side . there are people walking on the side of the train . there is a large white building behind the woman . the man is wearing a white and black jacket with a black helmet . the sky is blue with white clouds in the sky .


![Image 2](https://cs.stanford.edu/people/rak248/VG_100K_2/2414610.jpg)

Ground truth:
a woman in a blue tennis outfit stands on a green tennis court . she is swinging a blue tennis racket . there is a green tennis ball above her head .

Greedy:
a man is playing tennis . the man is wearing a white shirt and black shorts . the man is wearing a white shirt with a black design on the front . the court is green with white painted lines on it . the man is wearing a white shirt and black shorts .

Beam-2:
a woman is playing tennis racket . the man is wearing a white shirt . a woman is wearing a light blue shirt with blue jeans . there is wearing a black shirt with white design . the boy is wearing a blue shirt with blue design .

Sampling (No Temp):
a child is playing soccer . her ordered stands on the grass across the game . there is a chain link fence behind the fence and is eating a grassy hills . the blue court has a reflection of a fence on it . a large chain link fence can be seen behind the court .

Sampling (T = 0.4):
a man is wearing a black and white tennis outfit . there are several people standing in the stands behind the fence . the woman is holding a racket in her hands . the court is brown and green with white painted lines on it . the court is green and blue with white painted lines on it .

Sampling (T = 1.2):
this photo can a time way . there is brown dirt in front . the vanilla flowering leans smiling in order at her watch . the woman 's hair is tattoos strapped to over september hoses . part of another woman in a white arm and some white sweater .

Top-N Sampling (T = 0.5, V = 3000):
a man stands in the grass . the woman is wearing a white visor with a white skirt and a black band . the man is wearing a blue shirt and black pants . the man is wearing a black and white shirt . the woman is wearing a red shirt .

Nucleus Sampling (P = 0.9, T = 0.5):
a man is playing tennis . there are three women standing behind the fence . the man is wearing a white shirt and black pants . the court is green with white painted lines on it and the court is green . there are also many people standing behind the fence .


![Image 3](https://cs.stanford.edu/people/rak248/VG_100K_2/2388203.jpg)

Ground truth:
the man is taking a photo in the round mirror . he is bald . he is wearing an orange jacket . his camera is black . there is a train in the mirror too .

Greedy:
a train is on the train tracks . the train is red and white and has a black roof . the train is red and white and has a black roof . the sky is blue with white clouds in it . the man is wearing a black shirt and black pants .

Beam-2:
a lot of people are dressed in the room . the boy is wearing blue jeans and is wearing blue jeans . a lot with a black shirt and black pants . the man is wearing a white shirt and black pants . the train is on the train tracks .

Sampling (No Temp):
this photo is taken outside on a sunny day . the picture was taken at night time at night . two men are walking on both sides of the picture . there is a small thing tree to the left of his track . the kites are holding many large poles .

Sampling (T = 0.4):
a train is on the train tracks . the man is wearing a black hat , black shirt and black pants . the building in the background is dark grey and has a red roof . the building has a sign on it that is lit up . the man is wearing a dark jacket .

Sampling (T = 1.2):
an older gathering walkway came from a metal filigree train station . the street , there is a brown and brown stone building . there are other short of smallest crates , bikes towards the city left towards maybe huge trees . in the back there are three <unk> walking on bikes . trash can cross , beyond the homes are seemingly stacks length brown fence .

Top-N Sampling (T = 0.5, V = 3000):
a train is on the tracks . the sky is blue with a few clouds . there are many buildings and cars . the bus is parked on a road with a few cars parked . there is a large white truck parked in the street .

Nucleus Sampling (P = 0.9, T = 0.5):
a train is on the train tracks . the man is wearing a helmet and goggles . the man is wearing a white shirt and black shorts . the sky is blue with white clouds . there is a large windshield on the front of the bus .


![Image 4](https://cs.stanford.edu/people/rak248/VG_100K_2/2396483.jpg)

Ground truth:
people are on a play ground standing in the sand . a boy in a blue shirt and blue pants are climbing the pole . the woman in brown is holding a brown umbrella . there are people shadows on the ground .

Greedy:
a man is standing on a sidewalk . the sky is blue with white clouds in it . the man is wearing a black jacket and black pants with a black helmet . the sky is blue with white clouds in it . the sky is blue with white clouds .

Beam-2:
a lot of people are sitting . the boy is wearing a black shirt . there is wearing a shirt and dark blue jeans . the sun is seen on the the man is wearing a black jacket .

Sampling (No Temp):
there is an old completely empty train in a large city and contains a soccer park . an old group of kites are holding one contained . there is a big yellow light coming down from side the brown beach closest to the camera . the ground they are holding people , as they 're scaffolding . a few buildings can be seen in the distance .

Sampling (T = 0.4):
this is a picture of a street . there are people standing on the grass near the people . the sky is blue with white clouds in it . the sky is blue with white clouds in it . the sky above the buildings is blue with no clouds .

Sampling (T = 1.2):
the image is of a mocha officers bin in fair line barrels rain flatbed or cut - ) and an american <unk> avocados cabbage , crowds with silhouette halfway around the <unk> . the two brightly colored grass 's are going down on the visit area to appear to be adults and gooey , appliances , along with very twenty location of people . a long puff reading green butts customs covering grass and independently globes out from the clear ready to show the engines from the loading walls . the subway car has bike rides with white signs attached to it towards the bottom of the bridge , and behind to the arrange there are five golden wings and also two sides while . racquets sky is blue with white blinds in the background .

Top-N Sampling (T = 0.5, V = 3000):
a woman is walking down the street . the trees are in a line and the other is empty . the man is wearing a black helmet on his head and a blue shirt . the sky is blue with no clouds in the sky . a tall building is in the background of the photo .

Nucleus Sampling (P = 0.9, T = 0.5):
a man is standing outside on a sidewalk . the sky is blue and there are white clouds in the sky . there are many trees to the right of the train . the sky is clear and blue with white clouds . the sky is blue with white clouds in it .


![Image 5](https://cs.stanford.edu/people/rak248/VG_100K/2316231.jpg)

Ground truth:
a person is skiing through the snow . there is loose snow all around them from him jumping . the person is wearing a yellow snow suit . the person is holding two ski poles in their hands .

Greedy:
a man is standing on a snow covered ground . the man is wearing a black wet suit with a black hat . the man is wearing a black shirt and black pants . the sky is blue with white clouds in the sky . the sky is blue with white clouds in it .

Beam-2:
a person is standing on a black and black and black ski poles . the man is wearing a shirt and black helmet . the boy is wearing a hat , and a hat is standing in front . there is wearing a pair of a long sleeve shirt and black hat . the man is wearing a pair of blue jeans .

Sampling (No Temp):
a device for it is heading in a snow resort . the skier is wearing a white helmet next to his held in the air . a large foamy airplane can be seen coming in the sun . the grass next the giraffe has brown patches of it 's stretched out . buildings buildings across the city .

Sampling (T = 0.4):
a person is holding a ski board . the man is wearing a black helmet and goggles on his head . there is a black and white dog in the water . the sky is blue with white clouds in it . the water is a bright blue color with a few waves .

Sampling (T = 1.2):
this is image view of people . the distant legs are plush closely on each wings of some kind . as he dump through the mountain with a lush and white cop habitat . pedestrians line nearby muddy and start . behind them are extremely encased on sidewalks clay through grass that are mostly green with dirt .

Top-N Sampling (T = 0.5, V = 3000):
a man is sitting on a bench . the kite is black and has a red tail . the sky above the man is blue with white clouds . there are also mountains on the mountain . the sky is blue with a few white clouds .

Nucleus Sampling (P = 0.9, T = 0.5):
a man is standing on a beach covered with snow . the sky is blue with white clouds and a white background . the person on the left is wearing a black jacket and gray pants . the kite is red and white in color . the sky is blue with white clouds in it .
