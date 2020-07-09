Nikolai


## 2020-07-08 

Simon

  - Draft of the paper
  - Two models of information fusion:
	* Grounded object model: but the groudning is circular; denscap regions are used to predict the descriptions but then these descriptions are fused back with the dense-cap representations
	* Language as background knowledge model: dense-cap descriptions are used as a background knowledge support for the generation; the model can attend either on densecap regions (something visually interesting to describe) or on a chunked phrase coming from nackground knowledge
  - Evaluate grounded object model for language only, and language and attention
  - Setup human evaluation experiment: categores
	* Naturalness of description
	* Choice of words
	* Syntactic structure
	* Coherence of the text

Nikolai

  - Train some more models: model with background input, model with background input and attention
  - Expected: good generations, probably better than those generated with visual information only
  - Motivation for multimodal features: why do we construct them exactly the way it is done? Grounding motivation: learning linear layer to map two modalities together and ground language and vision into the same space with ReLU. Knowledge motivation: learn to choose between visual and background information, learn to decide which type of modality is more important for the current timestamp. Motivation for choosing strategy to build multimodal vector is crucial.
  - Human evaluation set-up: how natural the paragraph is? Evaluate across different properties: syntactic structure, choice of words, coherence, interestingness (already entailed by naturalness?), repetitiveness (we do not want repetitions)
  - Leave some feedback fields in human evaluation and ask humans to add any comments they would like to add.
  

## 2020-06-26 

  - Interpretation of the beam search and lenght of sentence: https://arxiv.org/pdf/1808.09582.pdf
  - BLEU and CIDeR
	* our BLEU scores are lower than Krause (30 vs 41 for BLEU1) but our CIDeR scores are higher (18 vs 13.5)
	* we are catching less wods, n-grams than the original implementation compared to ground truth
	* those that we do ctach are more unique and identifying for this image; goes with the principles of GRE
  - METEOR
    * Krause: 15.9; ours 12.8; works on synonyms
	* how many content words (vs functional words) did we get; normalised by the length of the prediction and penalised by the fact whether we predict them as single chunks or distributively
  - CIDeR
    * TF/IDF of each generated word
	* create a vector for n...4 grams
	* words not present are 0
	* take cosine similarity between the generated vectors and ground truth vectors
  - Actual generations
	* attention focuses on describing (only) objects in the focus of attention, e.g. the bus
	* backrgound knowledge adds spatial descriptions and relations between objects, information coming from densecaps; tge focus is to mention all the objects in the scene and their relations
	* with attention and background: the attention/focus on the object is still quite strong but we see a combination of both (which is good)
	* the other thing: attention is acting as visual attention
  - Gecko workshop with nice recorded talks https://sites.google.com/view/gecko2020/home (also relevant for this paper)
  - Next
	* Evaluations of generations by others (rank the generations from the best to worst), select images that are easy (one visually focuse dobject and some background objects) and hard (no visually focused object)
	* Skeleton of the headings/argument points for the paper; define the focus of the paper; identify missing arguments



## 2020-06-18 

  - ALPS, the NLP winter school in the Alps, http://lig-alps.imag.fr
  - Evaluation
	* Baseline: visual no attention
	* + background information: improvement
	* What search is used in Krause? This is not clear? Another paper claims that Krause is using beam search.
  - CIDER: TF/IDF on bigrams
  - METEOR: takes into account synonyms and paraphrases
  - CIDER has the highest improvement in our cases: because of TF/IDF CIDER is biased to more specific descriptions which we observed is the case with our descriptions anyway
  - Beam search produces very short sentences: problem with the beamsearch implementtation: uses sum rather than product
  - Nucleus sampling is producing the best results: only choose only from words that jointly represent some probability mass; the threshold is deifned as a parameter; 0.9 works the best; this will affect the number of words we can choose from each time; and then we randomly choose one of those words;
  - Attention is placed on the discourse LSTM; it helps the system tolearn how to realise background knowledge in the paragraph accross different sentences; because we know thta humans focus on different information when they start and proceed with describing; diversity in the sense that there some topic progression from one sentence to another; we hope that the discourse LSTM will learn how to attend to different chunks of background knowledge and visual information related to those; this gives us a discourse model; this is shown by attention maps



## 2020-06-09

Paper 1:

  - Background knowledge is useful for generating interesting descriptions as it provides additional richness of descriptions
	* The background knowledge provides additional information what is worth mentioning in the image; how humans conceputalise the world and devide scenes into objects and relations between objects
	* The model is able to use synonyms of words that it encounters
	* Descriptions do not only refer to what is seen
	* The descriptions provide background information, priming; they are explciit descriptions of individual objects with focus on what is relevant; we have additional information about objects;
	* In addition to information in the generated output text we are also priming the system with information how to describe objects; very similar to Mooney idea of providing explanations
	* The paragraph does provide conceptual information about objects but only those that are referred to in a description; in a single text maybe not all properties of the scene are mentioned but only those that were considered relavant by a describer; a describer could describe other important information which can be extracted from the background information; attention may be subjective
	* The model is able to hallucinate up to a point but halucination must be grounded in image; there is good question where the boundary generating grounded and ungrounded descriptions is
	* We are not necessarily hallucinating since we are basing it on actual descriptions of several objects; we are providing a more complete description of the image and the relations we find there to help to generate the descriptions; the original model is also hallucinating
	* Densecaps are relevant descriptions of objects
  - How do we evaluate?
  - See also J. Wu and R. J. Mooney. Self-critical reasoning for robust visual question answering. arXiv, arXiv:1905.09998 [cs.CV], 2019.
  - X. Liang, Z. Hu, H. Zhang, C. Gan, and E. P. Xing. Recurrent topic-transition gan for visual paragraph generation. arXiv, arXiv:1703.07022 [cs.CV]:1–10, 2017.
  - We have visual and linguistic topic represnetation of each image; then this informaiton is combined to get multi-modal topics; these are then attended to drive to drive the discourse LSTM which is repsonsible for generating discourse units/sentences; the Liang paper uses phrases to generate the words in sentences; they see it as fragments which should be glued together in a sentence; whereas we see it as providing a topic what can be said about the image and then the system is free to generate individual sentences from that
  - Topic modelling: we create visual and textual topics; from these topics the system learnes how to realise these topics in idnvidual paragraph
  - Evaluation:
	* The produces more interesting descriptions given the baseline
	* We need to say something that we really are learning topics; look at the attention maps for the combinaed features; attention there is over visual-language topics and the previous state of the discourse LSTM; it's modelling to what degree we take into account new topics and to what degree we are following from the previous discourse unit
  - Parallel co-attention: visual features are passed tohtough a dense layer and then multiplied with the linguistic features; hence visual features are adapted to linguistci features which cretae our topics (affinity matrix)
  - What to describe when?
	* Adding more to what
	* Adding the ability select when
  - Models
	* Baseline 1: Krause without end of sentence prediction
	* Model 2: Add linguistic topics (VL Topics with co-attention) and discourse attention
	* Model 3: Add only linguistic topics
  - To do:
      * Implement models with VL topics and attention (all combinations)
	  * Visualisation of attention with and without topics
	  * Evaluation (both beam search and samplign); qualitative evaluation to examine whether sampling is better
	  * Measuring diversity in the generated paragraphs (using frequencies of types and other measures, N's slides)


Paper 2:

  - Paragraphs have a discourse model.
  - Humans structure information diffirently accross different sentences.
  - We have a vision and language input model. How is this information structured accross different sentences?
  - Attention map of the


## 2020-05-29

  Nikolai, Simon and Asad

  - 50 visual representations of objects, 50 topic represnetations of descriptions of objects
  - attention on both; the attention is available to the discourse information structure LSTM to help out how to realise the paragraphs
  - top-down and bottom-up attention, cf. Lavie 2014, and S. Dobnik and J. D. Kelleher. A model for attention-driven judgements in Type Theory with Records. In J. Hunter, M. Simons, and M. Stone, editors, JerSem: The 20th Workshop on the Semantics and Pragmatics of Dialogue, volume 20, pages 25–34, New Brunswick, NJ USA, Juley 16–18 2016.
  - NLG:
	* E. Krahmer and K. van Deemter. Computational generation of referring expressions: A survey. Compu- tational Linguistics, 38(1):173–218, 2011.
	* K. v. Deemter. Computational models of referring: a study in cognitive science. The MIT Press, Cam- bridge, Massachusetts and London, England, 2016.
  - Evaluation:
	* the actual distirbution of adaptive attention
	* the attention graph as produced in Nikolai's previous paper over objects mentioned in the generated text
  - end of paragraph prediction removed; replace with a cheat end of paragraph detection: generate the same number of sentences as in the ground truth

Holiday

  - 2 weeks in December and 2 weeks in January (21 December - 18 January)


## 2020-05-22

  Nikolai, Simon and Asad

  - Salince and attention: how humans focus on paragraphs of text; Nikolai's entity linker; how is this captured in the current Stanford corpus; does our model capture it and how can we improve the model
  - A. Zarcone, M. van Schijndel, J. Vogels, and V. Demberg. Salience and attention in surprisal-based accounts of language processing. Frontiers in Psychology, 7:844, 2016. [Paper](https://www.frontiersin.org/articles/10.3389/fpsyg.2016.00844/full)
  - Surprisal modelling; information density, constant surprisal; focus on the herar
  - What is informational density in generated captions? Are we generating too much or too little?
  - The context of annotation; hwo does this affect the information in the description? Experiment where one person does not see an image but only heras descriptions.
  - Remove the sentence generation part; use the number of sentences from the corpus to evaluate the descriptions



## 2020-05-15

  - Extracted DenseCap features
  - Transformer presentation
  - Multi-modal fusion possibilities:
	* pool(visual), pool (backrgound), visual + background = 512 + 512 = 1024; confuses end of sentence detection but produces cool and rich descriptions ==> move end of sentence detection (possibly to the end of the word LSTM) ; remove end of sentence detection completely
	* combinate visual and background into a single 512 vector with multiplication and test; does this improve end of sentence detection and also generates interesting descriptions
	* no pooling, hence 50 x 1024 or 50 x 512; then a dense layer to 512
  - Conceptual model overview:
	* Information fusion (visual + background) and summarisation (pooling)
	* Discourse planning (sentence LSTM)
	* Sentence relaisation (word LSTM)
	* Where does EOP (end of paragraph) fit in?
	* See our diagram doodle
  - Replace DenseCap (2016) with FasterRCNN (bottom-up, Anderson; Detectron2 SoTA for object detection by FB) https://github.com/facebookresearch/detectron2
  - Numeric prediction for EOP which conveys gradience for individual sentences leading to the stop (maybe this is not relevant since the EOP is not an LSTM (the signal about EOP is not fed back into prediction) but this relies on the sentence LSTM which is gradient)



## 2020-05-08

  - Sentence LSTM, add gradient scores for the sentence completion based on the degree to which the sentence is the last sentence
  - Presentation in the Transforms course: S. Herdade, A. Kappeler, K. Boakye, and J. Soares. Image captioning: Transforming objects into words. In Advances in Neural Information Processing Systems, pages 11135–11145, 2019.
  - Paper 1: Intent in image captioning
  - Paper 2: Adding language/conceptual backrgound information to paragrpah description: Coling, deadline 2020-07-01, alternative LANTERN workshop, deadline 2020-08-21
  - Paper 3: Transformer paper, compare the vision and language fusion between trasnformers and adaptive attention
  - Realtion to attention and information fusion
    * S. Dobnik and J. D. Kelleher. A model for attention-driven judgements in Type Theory with Records. In J. Hunter, M. Simons, and M. Stone, editors, JerSem: The 20th Workshop on the Semantics and Pragmatics of Dialogue, volume 20, pages 25–34, New Brunswick, NJ USA, July 16–18 2016.



## 2020-05-01

  - How to add attention (similar to Lu and Ghanimifard):
  * In addition to visual embedding we also have ambedding from the phrases generated by DenseCap
  * attend on a vector that contains visual embeddings and phrases from the DenseCap
  * Motivation: DenseCap descriptions will be descriptions of the most visible objects in the scene; in DenseCap we can control the number of region proposals from which unique objects are identified (there are constraints on how objects are combined); the number of identified objects is not the same for images; (1) we could take the visual features of these identified objects as a vector of attended objects for example; (2) we take the actual LSTM hidden state of the generated descriptions of these proposals; in both cases we are testing the effects of transfer learning
  * MaxPooling is now performing the task of attention; MaxPooling also from DenseCap regions and DenseCap description vectors; maybe in later versions replace with attention
  * Where to introduce attention; before Sentence LSTM to generate sentence topics or at the level of Word LSTM to generate individual words?
  *
  * S. Ullman. Visual routines. Cognition, 18(1–3):97–159, 1984.
  - Where to introduce attention?
	* Attention over visual features and the DenseCap features: what is relevant in the image to describe?
	* Attention on sentence LSTM: how to describe rleevant information accross sentences
	* Attention over words: how to generate individual words in a sentence
  - Next:
	* Extract linguistic information from DenseCap and add it to the model with MaxPooling; (1) visual representations of the identified regions (2) LSTM language model of the identified regions; (3) both
	   * Getting the automatic scores for the current models
	* Calculations of CIDEr: how to interpret the score; experiment with ground truth being slightly randomly modified; the effect on the score
  - Pragmatic descriptions of perceptual stimuli Emiel van Miltenberg: [paper](https://www.aclweb.org/anthology/E17-4001.pdf)
   * Nucleus model with temperature is the best (comparing the generations manually)

On the intent in image description tasks

  - The task of image captioning/descriptions is unnatural; there is no clearly expressed intent which is defined by the task
  - The images are taken with a particular intent but this is not visible to captioners (e.g. the beach and the climbing boy example)
  - This leads to problems:
	* The quality of the descriptions: people are unsure what they should be describing; they resort to very generic descriptions: highly technical, e.g. man woman, etc. quite uninteresting; the problem of evaluation?
	* The quality of the images: they are not representing very typical scenes sometimes and therefore relations are less visual; even describers halucinate/rely on their world knowledge
	* Sometimes the descriptions that are generated are valid but different from the ground truth; we do not know whether the system is looking at what it is describing or halucinating; there may be several different tasks/intention for a particula picture; the system may have identified that from the LM but it is not grounded
	- But even if we know the intent; then the system needs to learn a discousre model of a particular task/intent; what are valid sequences of sentences in a paragrpH, how dow e approach to describe a particular task; generate a general description and then focus on individual details; paragraph discourse model
	_ The role of attention: the model needs to learn what features to attend on;: the attention is at different levels; the image, general topics; how to structure the paragraphs; how to generate individual words in a sentence
	- Keywords:
	  * visual data
	  * task orintedness
	  * attention


## 2020-04-23

  - Baseline model with no pre-training now generates very good descriptions; the problem was in the way the generation was implementaed; it was conditioned on the ground truth rather than previous prediction
  - Different implementations of search
	* Greedy search: generic output, frequenetly incorrect and repetitive
	* Sampling from multinomial distribution https://docs.scipy.org/doc/numpy-1.13.0/reference/generated/numpy.random.multinomial.html for implementation see Softmax-with-temperature-test.ods
	* Top-n sampling (temperature scaling on the top most probable words), https://cs.stackexchange.com/questions/79241/what-is-temperature-in-lstm-and-neural-networks-generally
	* Beam-search


## 2020-04-17

  - Using Densecap embeddings and LSTM language model and fine-tuning them (initilaise only and then update the weights) improves the performance
  - A system without the denscap language model generates fragmented phrases (which seem to be good) but the syntax is fragmented
  - In packing sentences, now we do not calculate loss on padded tokens
  - Results:
	* Baseline 1:
  - Next?
	* Double checking of the code
	* Create Topic/language embeddings: take an image, create denscap descriptions, take the hidden state of the last word of the LSTM and save this as a topic represnetation for each region; 50 regions; then max polling and finally we get Topic embeddings; concatenate topic embeddings and visual embeddings; start generating sentences from that; visual and language pipeline; the sentence genrator needs to learn how to select information and generate from that
	* INLG, 15 August
	* Any other intermediate conferences
	* Semdial, 15 May: a position paper about the state of the art of the paragraph generation
	* Literature: Devi Parikh (Drev Batra), deep learning models for generation; how are the intermediate topics represented from which language is generated
	* A comparison of paragraph generation models: https://helda.helsinki.fi/bitstream/handle/10138/304686/arturs_polis_thesis_final.pdf?sequence=1&isAllowed=y
	* Kevin Knight, The Moment When the Future Fell Asleep 9am-10am, June 3, NAACL 2018
	* Elizabeth Clark, Yangfeng Ji, Noah A. Smith: Neural Text Generation in Stories Using Entity Representations as Context, NAACL 2018 outstanding paper, 4:36PM - 4:54PM, https://www.aclweb.org/anthology/N18-1204/ (includes [video](https://vimeo.com/277672801) of the talk)



## 2020-04-10

  - The graph of the learning configuration, 2020-04-10-model_scheme.pdf
  - Test the effects of the end of sentence predictor by removing this part
  - Remove one of the RuLUs re-econding the visual features
  - End to end model from visual features to sentences; the model does not have any attention; can we help the model to attend; dense-cap descriptions; attention on visual features in the bounding boxes (cf. Mehdi's paper)
  - Training and validation loss: after some point the validation loss explodes and training loss is reduced: over-fitting starts; descriptions at this stage?
  - Pre-training in the original paper:
    * Word-LSTM (embeddings, weights and linear) is pre-trained from Densecap; this is fine-tuned for the current sentences
    * Word-embeddings from the Densecap
    * What happens if we use a pre-trained language model instead of training it?
  - Next?
    * Baseline 1: Plot the training and validation loss, validate the generation of the model with the lowest val loss; save the sentences
    * Baseline 2: Remove the multi-tasking of the sentence loss; improvement? Generate max number of images, i.e. 6
    * Baseline 3: Representation of visual vectors, remove one of the ReLUs
    * Baseline 4: Use Densecap pre-training for the word LSTMs; two options: freeze and add another layer; use the weights and update them in the fine tuning; there may be doing this in the paper
  - INLG: https://www.inlg2020.org,  deadline of 15 May



## 2020-04-03

  - Discussing the ISP



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
