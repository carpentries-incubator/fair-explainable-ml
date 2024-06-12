---
title: "Explainability methods overview"
teaching: 0
exercises: 0
---

# Fantastic Explainability Methods and Where to Use Them

We will now take a bird's eye view of explainability methods that are widely applied on complex models like neural networks. 
We will get a sense of when to use which kind of method, and what the tradeoffs between these methods are. 


## Three axes of use cases for understanding model behavior

When deciding which explainability method to use, it is helpful to define your setting along three axes. 
This helps in understanding the context in which the model is being used, and the kind of insights you are looking to gain from the model.

### Inherently Interpretable vs Post Hoc Explainable

Understanding the tradeoff between interpretability and complexity is crucial in machine learning. 
Simple models like decision trees, random forests, and linear regression offer transparency and ease of understanding, making them ideal for explaining predictions to stakeholders. 
In contrast, neural networks, while powerful, lack interpretability due to their complexity. 
Post hoc explainable techniques can be applied to neural networks to provide explanations for predictions, but it's essential to recognize that using such methods involves a tradeoff between model complexity and interpretability. 

Striking the right balance between these factors is key to selecting the most suitable model for a given task, considering both its predictive performance and the need for interpretability.

![The tradeoff between Interpretability and Complexity](https://raw.githubusercontent.com/carpentries-incubator/fair-explainable-ml/main/images/e5-interpretability-vs-complexity.png)
_Credits: AAAI 2021 Tutorial on Explaining Machine Learning Predictions: State of the Art, Challenges, Opportunities._

### Local vs Global Explanations
Local explanations focus on describing model behavior within a specific neighborhood, providing insights into individual predictions. 
Conversely, global explanations aim to elucidate overall model behavior, offering a broader perspective. 
While global explanations may be more comprehensive, they run the risk of being overly complex. 

Both types of explanations are valuable for uncovering biases and ensuring that the model makes predictions for the right reasons. 
The tradeoff between local and global explanations has a long history in statistics, with methods like linear regression (global) and kernel smoothing (local) illustrating the importance of considering both perspectives in statistical analysis.

### Black box vs White Box Approaches
Even without access to the model weights, black box or top down approaches can shed a lot of light on model behavior. 
For example, by simply evaluating the model on certain kinds of data, high level biases or trends in the model’s decision making process can be unearthed. 

White box approaches use the weights and activations of the model to understand its behavior. 
These classes or methods are more complex and diverse, and we will be discussing them in more detail later in this chapter.
Some large models are closed-source due to commercial or safety concerns; for example, users can’t get access to the weights of GPT-4. This limits the use of white box explanations to such models.



## Classes of Explainability Methods for Understanding Model Behavior

### Diagnostic Testing

This is the simplest approach towards explaining model behavior. 
This involves applying a series of unit tests to the model. 
By developing test examples that break the heuristics the model relies on (called counterfactuals), you can gain insights into the high-level behavior of the model.

**Use-Case:** Black box, post-hoc explainable, global

**Example Methods:** [Counterfactuals](https://arxiv.org/abs/1902.01007), [Unit tests](https://arxiv.org/abs/2005.04118)

**Pros and Cons:**
These methods allow for gaining insights into the high-level behavior of the model without the needing access to model weights.
This is especially useful with recent powerful closed-source models like GPT-4. 
One challenge with this approach is that it is hard to identify in advance what heuristics a model may depend on.


### Baking interpretability into models

Some recent research has focused on tweaking highly complex models like neural networks, towards making them more interpretable inherently. 
One such example with language models involves training the model to generate rationales for its prediction, in addition to its original prediction.
This approach has gained some traction, and there are even [public benchmarks](https://arxiv.org/abs/1911.03429) for evaluating the quality of these generated rationales.

**Use Case:** Interpretable, Local, White Box

**Example methods:** [Rationales with WT5](https://arxiv.org/abs/2004.14546), [Older approaches for rationales](https://arxiv.org/abs/1606.04155)   

**Pros and cons:**
These models hope to achieve the best of both worlds: complex models that are also inherently interpretable.
However, research in this direction is still new, and there are no established and reliable approaches for real world applications just yet. 


### Identifying Decision Rules of the Model:

In this class of methods, we try find a set of rules that generally explain the decision making process of the model. 
Loosely, these rules would be of the form "if a specific condition is met, then the model will predict a certain class".
Explaining the decision making process of an _entire_ neural network can be challenging.
Instead, these rules are built over a small set of inputs, for example, a single dataset. 

**Use Case:** White box, post-hoc explainable, global or local

**Example methods:** [Anchors](https://aaai.org/papers/11491-anchors-high-precision-model-agnostic-explanations/), [Universal Adversarial Triggers](https://arxiv.org/abs/1908.07125)

**Pros and cons:**
Some global rules help find "bugs" in the model, or identify high level biases. But finding such broad coverage rules is challenging. 
Furthermore, these rules only showcase the model's weaknesses, but give next to no insight as to why these weaknesses exist.



### Visualizing model weights or representations
Just like how a picture tells a thousand words, visualizations can help encapsulate complex model behavior in a simple image. 
Visualizations are commonly used in explaining neural networks, where the weights or data representations of the model are directly visualized.
Many such approaches involve reducing the high-dimensional weights or representations to a 2D or 3D space, using techniques like PCA, tSNE, or UMAP.
Alternatively, these visualizations can retain their high dimensional representation, but use color or size to identify which dimensions or neurons are more important.

**Use Case:** Black box, Post Hoc, Global

**Example methods:** [Visualizing attention heatmaps](https://arxiv.org/abs/1612.08220), Weight visualizations, Model activation visualizations

**Pros and cons:**
Gleaning model behaviour from visualizations is very intuitive and user-friendly, and visualizations sometimes have interactive interfaces.
However, visualizations can be misleading, especially when high-dimensional vectors are reduced to 2D, leading to a loss of information (crowding issue).

An iconic debate exemplifying the validity of visualizations has centered around attention heatmaps. 
Research has shown them to be [unreliable](https://arxiv.org/abs/1902.10186), and then [reliable again](https://arxiv.org/abs/1908.04626). (Check out the titles of these papers!)
Thus, visualization can only be used as an additional step in an analysis, and not as a standalone method.


### Understanding impact of training examples
Which of the many datapoints seen during training a model, caused it to generate a specific prediction? 
Identifying this helps in finding annotation artifacts in the data, and correcting them. 

**Use Case:** White box, Post Hoc, Local

**Example methods:** [Influence functions](https://arxiv.org/abs/1703.04730), [Representer point selection](https://arxiv.org/abs/1811.09720)

**Pros and cons:**
The insights from these approaches are actionable - by identifying the data responsible for a prediction, it can help correct labels/annotation artifacts/labels in that data.
Unfortunately, these methods scale poorly with the size of the model and training data, quickly becoming computationally expensive.
Furthermore, even knowing which datapoints had a high influence on a prediction, we don’t know what in the datapoint caused that that influence.


### Understanding the impact of a single example:
For a single input, what parts of the input were most important in generating the model's prediction? 
These methods study the signal sent by various features to the model, and observe how the model reacts to changes in these features.

**Use Case:** Local, black/white box (depending on specific method), post hoc.

**Example methods:** [Saliency Maps](https://arxiv.org/abs/1312.6034), [LIME](https://arxiv.org/abs/1602.04938)/[SHAP](https://arxiv.org/abs/1705.07874), Perturbations ([Input reduction](https://arxiv.org/abs/1804.07781), [Adversarial Perturbations](https://arxiv.org/abs/1712.06751))

**Pros and cons:**
These methods are fast to compute, and flexible in their use across models.
However, the insights gained from these methods are not actionable - knowing which part of the input caused the prediction does not highlight why that part caused it.
On finding issues in the prediction process, it is also hard to pick up on if there is an underlying issue in the model, or just the specific inputs tested on.


### Probing internal representations

As the name suggests, this class of methods aims to probe the internals of a model, to discover what kind of information or knowledge is stored inside the model. 
Probes are often administered to a specific component of the model, like a set of neurons or layers within a neural network. 

**Use Case:** Global/local, white box, post hoc

**Example methods:** [Probing classifiers](https://direct.mit.edu/coli/article/48/1/207/107571/Probing-Classifiers-Promises-Shortcomings-and), [Causal tracing](https://proceedings.neurips.cc/paper/2020/hash/92650b2e92217715fe312e6fa7b90d82-Abstract.html)

**Pros and cons:**
Probes have shown that it is possible to find highly interpretable components in a complex model, e.g., MLP layers in transformers have been shown to store factual knowledge in a structured manner.
However, there is no systematic way of finding interpretable components, and many components may remain elusive to humans to understand.
Furthermore, the model components that have been shown to contain certain knowledge may not actually play a role in the model's prediction.


### TODO: Methods related to graph based structures


## Summary

To summarize the methods discussed above, we can categorize them based on their approach, interpretability, and scope.

| Approach                                                                                         | Post Hoc or Inherently Interpretable? | Local or Global? | White Box or Black Box? |
|--------------------------------------------------------------------------------------------------|---------------------------------------|------------------|-------------------------|
| [Diagnostic Testing](#diagnostic-testing)                                                        | Post Hoc                              | Global           | Black Box               |
| [Baking interpretability into models](#baking-interpretability-into-models)                      | Inherently Interpretable              | Local            | White Box               |
| [Identifying Decision Rules of the Model](#identifying-decision-rules-of-the-model)              | Post Hoc                              | Both             | White Box               | 
| [Visualizing model weights or representations](#visualizing-model-weights-or-representations)    | Post Hoc                              | Global           | White Box               |
| [Understanding the impact of training examples](#understanding-the-impact-of-training-examples)  | Post Hoc                              | Local            | White Box               |
| [Understanding the impact of a single example](#understanding-the-impact-of-a-single-example)    | Post Hoc                              | Local            | Both                    |
| [Probing internal representations of a model](#probing-internal-representations)                 | Post Hoc                              | Global/Local     | White Box               |




:::::::::::::::::::::::::::::::::::::: challenge

Think about the following scenarios and suggest which explainability method would be most appropriate to use, and what information could be gained from that method. Furthermore, think about the limitations of your findings.

_Note:_ These are open-ended questions, and there is no correct answer. Feel free to break into discussion groups to discuss the scenarios.

[//]: # ([These are open-ended questions. Participants are encouraged to discuss if current explainability methods are sufficient to provide guidance.])
[//]: # (Given a series of scenarios, suggest which explainability method would be most appropriate to use, and what information could be gained from that method. Furthermore, highlight the limitations of your findings. )
[//]: # ([Use responses from survey for scenarios])
[//]: # ([Vision + Text + Tabular examples])

**Scenario 1**: Suppose that you are an ML engineer working at a tech company. A fast-food chain company consults with you about sentimental analysis based on feedback they collected on Yelp and their survey. You use an open sourced LLM such as Llama-2 and finetune it on the review text data. The fast-food company asks to provide explanations for the model: 
Is there any outlier review? How does each review in the data affect the finetuned model?
Which part of the language in the review indicates that a customer likes or dislikes the food? Can you score the food quality according to the reviews?
Does the review show a trend over time? What item is gaining popularity or losing popularity?
Q: Can you suggest a few explainability methods that may be useful for answering these questions?

[//]: # ([These are open-ended questions. Participants are encouraged to discuss if current explainability methods are sufficient to provide guidance.])

**Scenario 2**: Suppose that you are a radiologist who analyzes medical images of patients with the help of machine learning models. You use black-box models (e.g., CNNs, Vision Transformers)  to complement human expertise and get useful information before making high-stake decisions. 
Which areas of a medical image most likely explains the output of a black-box? 
Can we visualize and understand what features are captured by the intermediate components of the black-box models?
How do we know if there is a distribution shift? How can we tell if an image is an out-of-distribution example?
Q: Can you suggest a few explainability methods that may be useful for answering these questions?

**Scenario 3**: Suppose that you work on genomics and you just collected samples of single-cell data into a table: each row records gene expression levels, and each column represents a single cell. You are interested in scientific hypotheses about evolution of cells. You believe that only a few genes are playing a role in your study. 
What exploratory data analysis techniques would you use to examine the dataset?
How do you check whether there are potential outliers, irregularities in the dataset?
You believe that only a few genes are playing a role in your study. What can you do to find the set of most explanatory genes?
How do you know if there is clustering, and if there is a trajectory of changes in the cells? 
Q: Can you explain the decisions you make for each method you use?

::::::::::::::::::::::::::::::::::::::::::::::::::

### References and Further Reading

This lesson provides a gentle overview into the world of explainability methods. If you'd like to know more, here are some resources to get you started:

- Tutorials on Explainability:
  - [Wallace, E., Gardner, M., & Singh, S. (2020, November). Interpreting predictions of NLP models. In Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing: Tutorial Abstracts (pp. 20-23).](https://github.com/Eric-Wallace/interpretability-tutorial-emnlp2020/blob/master/tutorial_slides.pdf)
  - [Lakkaraju, H., Adebayo, J., & Singh, S. (2020). Explaining machine learning predictions: State-of-the-art, challenges, and opportunities. NeurIPS Tutorial.](https://explainml-tutorial.github.io/aaai21)
  - [Belinkov, Y., Gehrmann, S., & Pavlick, E. (2020, July). Interpretability and analysis in neural NLP. In Proceedings of the 58th annual meeting of the association for computational linguistics: tutorial abstracts (pp. 1-5).](https://sebastiangehrmann.github.io/assets/files/acl_2020_interpretability_tutorial.pdf)
- Research papers:
  - [Holtzman, A., West, P., & Zettlemoyer, L. (2023). Generative Models as a Complex Systems Science: How can we make sense of large language model behavior?. arXiv preprint arXiv:2308.00189.](https://arxiv.org/abs/2308.00189)
  
