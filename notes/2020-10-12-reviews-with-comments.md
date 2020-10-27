============================================================================ 
INLG 2020 Reviews for Submission #102
============================================================================ 

Title: When an image tells a story: the role of visual and semantic information for generating paragraph descriptions
Authors: Nikolai Ilinykh and Simon Dobnik

> SD: Generally, my advice is to treat all reviewer comments, either positive or negative,  as informative. This is a reation that someone ahd wehn reading the paper. Even if they misunderstand a point, the trick is then to fool-proof the text so that a reader would not get confused in the same way. This can be easily done by adding a sentence here and then to disambiguate.


============================================================================
                            REVIEWER #1
============================================================================

---------------------------------------------------------------------------
Reviewer's Scores
---------------------------------------------------------------------------
                   APPROPRIATENESS (1-5): 4
                            IMPACT (1-5): 4
                       ORIGINALITY (1-5): 4
             MEANINGFUL COMPARISON (1-5): 4
                 TECHNICALLY SOUND (1-5): 4
                           CLARITY (1-5): 5
                    RECOMMENDATION (1-5): 4

What are the main strengths and weaknesses of the paper?
---------------------------------------------------------------------------
Strengths:
-------------
1. Clear and interesting approach for combining visual features and language.
2. Good results.
3. Good experimentation. 

Weaknesses:
-----------------
1. Adaptation of  the hierarchical image paragraph model by Krause et al. (2017) instead of proposing an original approach.
2. No comparison with knowledge-based approaches.
3. No redundancy control in the generation process (in the example provided, the information is repeated).
---------------------------------------------------------------------------

> SD: For (1) Emphasise that this is a deliberate extension so that we can study the effects of adding language information. However, through this extension we effectively propose a new model.

> SD: For (2) Perhaps add some references. Mitchell, etc. But a performance comparison with concerete systems would be out of the scope of this paper.

> SD: For (3) This is probably because of the example. We need to add discussion alongside these lines: we do not control redundancy because in a way redundancy reduction is evaluatied through the incroporation of the linguistic information. We hope that this will reduce redundancy. In a production system additional redundancy control could be incorporated. The success of redundancy control by the linguistic module has been evaluated through the AMT evaluation. Here you can explain how the features we have tested relate to redundancy.


Detailed Comments
---------------------------------------------------------------------------
In general, the paper is very well explained and the idea presented in interesting. The experiments performed are sound and the evaluation conducted is appropriate, combining automatic and human-based evaluation. I think the approach is very promising, and I would recommend the author to integrate some redundancy control in the approach to avoid the repetition of information. They could have a look at the redundancy control techniques that are usually employed in the task of multi-document summarization.
---------------------------------------------------------------------------

> SD: Yay!


============================================================================
                            REVIEWER #2
============================================================================

---------------------------------------------------------------------------
Reviewer's Scores
---------------------------------------------------------------------------
                   APPROPRIATENESS (1-5): 4
                            IMPACT (1-5): 3
                       ORIGINALITY (1-5): 2
             MEANINGFUL COMPARISON (1-5): 3
                 TECHNICALLY SOUND (1-5): 3
                           CLARITY (1-5): 4
                    RECOMMENDATION (1-5): 3

What are the main strengths and weaknesses of the paper?
---------------------------------------------------------------------------
Strengths:
-------------
1. the paper is well written. 
2. it does a good job comparing the proposed approach with the existing literature.
3. results are good.

Weaknesses:
-----------------
1. it is not clear how similar or different the generated descriptions are. 
2. the paper requires a qualitative error analysis section.
3.
---------------------------------------------------------------------------

> SD: For (1) Question about redundancy? The similarity between the description is measured in the experiment where we calculate m-BLEU and self-CIDER in Table 2. Emphasise this more strongly?

> SD: For (2) This is a good point. Rather than cherry-picking only good examples it also good to discuss shortcomings. This is what we are interested in as scientists. I assume Figure 1 shows a good example which comes quite early on but then I do not think we discuss qualitatively these descriptions. Shall we add another bad example and then discuss how the systems prodice good an bad sentences. I suspect the system will produce bad sentences when the image is challenging, as we were discussing earlier, when even a human had a difficulty to see what the image is about.



Detailed Comments
---------------------------------------------------------------------------
This paper introduces a model for generating multi-sentence image descriptions. It investigates the following question: to what extend using visual and/or textual features can improve the accuracy fo generated image descriptions? It shows that the model that uses both visual and language input can be used to generate accurate paragraphs. 
The paper is well written and the claims are supported. 

My question is, how do you make sure that the generated sentences are not very similar? that is to say, how do you make sure that the model describes different parts of the image or communicates different facts about the image in different sentences? I would like to see a careful analysis of the diversity of the sentences. Do they describe different content or they are just stating the same fact in different words? 
The paper also needs a qualitative error analysis section that describes the mistakes that the model makes.
---------------------------------------------------------------------------

> SD: Second paragraph: Self-measures, human evaluation and also qualitative analysis of some good and bad examples. Emphasise in the discussion/conclusion. Ah, I see, here we could measure the diversity of objects related that you suggested.


============================================================================
                            REVIEWER #3
============================================================================

---------------------------------------------------------------------------
Reviewer's Scores
---------------------------------------------------------------------------
                   APPROPRIATENESS (1-5): 5
                            IMPACT (1-5): 3
                       ORIGINALITY (1-5): 3
             MEANINGFUL COMPARISON (1-5): 4
                 TECHNICALLY SOUND (1-5): 4
                           CLARITY (1-5): 3
                    RECOMMENDATION (1-5): 4

What are the main strengths and weaknesses of the paper?
---------------------------------------------------------------------------
Strengths:
-------------
1. Comprehensive initial investigation into the performance of various DNN approaches to multi-sentence image description. 
2. Useful finding on the disparity between human evaluation and automated measures. 
3.

Weaknesses:
-----------------
1. Preliminary investigation, did not address issues such as how multi-sentence descriptions should be modulated by task-context or other situation-dependent factors.
2.
3.
---------------------------------------------------------------------------

> SD: Emphasise in the discussion/consluisons that task-context will be evaluated in the next step/future work.


Detailed Comments
---------------------------------------------------------------------------
This paper investigates the problem of multi-sentence image description using recent DNN techniques, focusing testing a variety of approaches on both automatic and human evaluation measures. Overall, the paper is clearly written and organized, and appears technically sound. One key contribution is highlighting that descriptions evaluated most positively by people may not be the best according to automated measures. Additionally, the results also suggest that multi-modal integration of information improves description quality. As the authors point out, this work described in the submission is a foundation for future work in description. Given the preliminary nature of this work, one limitation is that it does not address issues such as how multi-sentence descriptions should be modulated by task-context or other situation-dependent factors. When people freely view images, they will attend and encode differently than when they have a specific task in mind. How could the DNN app!
 roaches present be adapted and trained to account for this?
---------------------------------------------------------------------------

> SD: All good points that raise from what we already meantion in the introduction. Perhaps this paper is too big for three tasks (integration of linguistic knowledge, differences between automatic and human evaluation and the effects of the context) and therefore they call the work preliminary. Ideally, we would write a paper on each and test each question properly. Overall, what the reviewer says here is valid and we could add a similar discussion to discussion/conclusion describing what we are going to do next.
