---
title: "Explainability methods: deep dive"
teaching: 0
exercises: 0
---

:::::::::::::::::::::::::::::::::::::: questions 

- TODO

::::::::::::::::::::::::::::::::::::::::::::::::

::::::::::::::::::::::::::::::::::::: objectives

- TODO

::::::::::::::::::::::::::::::::::::::::::::::::

## A Deep Dive into Methods for Understanding Model Behaviour

In the previous section, we scratched the surface of explainability methods, introducing you to the broad classes of methods designed to understand different aspects of a model's behavior.

Now, we will dive deeper into two widely used methods, each one which answers one key question: 

## What part of my input causes this prediction?

When a model makes a prediction, we often want to know which parts of the input were most important in generating that prediction.
This helps confirm if the model is making its predictions for the right reasons. 
Sometimes, models use features totally unrelated to the task for their prediction - these are known as 'spurious correlations'.
For example, a model might predict that a picture contains a dog because it was taken in a park, and not because there is actually a dog in the picture.

**[Saliency Maps](https://arxiv.org/abs/1312.6034)** are among the most simple and popular methods used towards this end. 
We will be working with a more sophisticated version of this method, known as **[GradCAM](https://arxiv.org/abs/1610.02391)**.

#### Method and Examples

A saliency map is a kind of visualization - it is a heatmap across the input that shows which parts of the input are most important in generating the model's prediction.
They can be calculated using the gradients of a neural network, or by perturbing the input to any ML model and observing how the model reacts to these perturbations.
The key intuition is that if a small change in a part of the input causes a large change in the model's prediction, then that part of the input is important for the prediction.
Gradients are useful in this because they provide a signal towards how much the model's prediction would change if the input was changed slightly.

For example, in an image classification task, a saliency map can be used to highlight the parts of the image that the model is focusing on to make its prediction.
In a text classification task, a saliency map can be used to highlight the words or phrases that are most important for the model's prediction.

GradCAM is an extension of this idea, which uses the gradients of the final layer of a convolutional neural network to generate a heatmap that highlights the important regions of an image.
This heatmap can be overlaid on the original image to visualize which parts of the image are most important for the model's prediction.

Other variants of this method include [Integrated Gradients](https://arxiv.org/abs/1703.01365), [SmoothGrad](https://arxiv.org/pdf/1806.03000), and others, which are designed to provide more robust and reliable explanations for model predictions.
However, GradCAM is a good starting point for understanding how saliency maps work, and is a popularly used approach.

Alternative approaches, which may not directly generate heatmaps, include [LIME](https://arxiv.org/abs/1602.04938) and [SHAP](https://arxiv.org/abs/1705.07874), which are also popular and recommended for further reading. 

#### Limitations and Extensions

Gradient based saliency methods like GradCam are fast to compute, requiring only a handful of backpropagation steps on the model to generate the heatmap.
The method is also model-agnostic, meaning it can be applied to any model that can be trained using gradient descent.
Additionally, the results obtained from these methods are intuitive and easy to understand, making them useful for explaining model predictions to non-experts.

However, their use is limited to models that can be trained using gradient descent, and have white-box access. 
It is also difficult to apply these methods to tasks beyond classification, making their application limited with many recent
generative models (think LLMs).

Another limitation is that the insights gained from these methods are not actionable - knowing which part of the input caused the prediction does not highlight why that part caused it.
On finding issues in the prediction process, it is also hard to pick up on if there is an underlying issue in the model, or just the specific inputs tested on.


## What part of my model causes this prediction?

When a model makes a correct prediction on a task it has been trained on (known as a 'downstream task'), 
**[Probing classifiers](https://direct.mit.edu/coli/article/48/1/207/107571/Probing-Classifiers-Promises-Shortcomings-and)** can be used to identify if the model actually contains the relevant information or knowledge required 
to make that prediction, or if it is just making a lucky guess.
Furthermore, probes can be used to identify the specific components of the model that contain this relevant information, 
providing crucial insights for developing better models over time.

#### Method and Examples

A neural network takes its input as a series of vectors, or representations, and transforms them through a series of layers to produce an output.
The job of the main body of the neural network is to develop representations that are as useful for the downstream task as possible, 
so that the final few layers of the network can make a good prediction.

This essentially means that a good quality representation is one that _already_ contains all the information required to make a good prediction. 
In other words, the features or representations from the model are easily separable by a simple classifier. And that classifier is what we call 
a 'probe'. A probe is a simple model that uses the representations of the model as input, and tries to learn the downstream task from them.
The probe itself is designed to be too easy to learn the task on its own. This means, that the only way the probe get perform well on this task is if 
the representations it is given are already good enough to make the prediction.

These representations can be taken from any part of the model. Generally, using representations from the last layer of a neural network help identify if
the model even contains the information to make predictions for the downstream task. 
However, this can be extended further: probing the representations from different layers of the model can help identify where in the model the
information is stored, and how it is transformed through the model.

Probes have been frequently used in the domain of NLP, where they have been used to check if language models contain certain kinds of linguistic information. 
These probes can be designed with varying levels of complexity. For example, simple probes have shown language models to contain information 
about simple syntactical features like [Part of Speech tags](https://aclanthology.org/D15-1246.pdf), and more complex probes have shown models to contain entire [Parse trees](https://aclanthology.org/N19-1419.pdf) of sentences.

#### Limitations and Extensions

One large challenge in using probes is identifying the correct architectural design of the probe. Too simple, and 
it may not be able to learn the downstream task at all. Too complex, and it may be able to learn the task even if the 
model does not contain the information required to make the prediction.

Another large limitation is that even if a probe is able to learn the downstream task, it does not mean that the model
is actually using the information contained in the representations to make the prediction. 
So essentially, a probe can only tell us if a part of the model _can_ make the prediction, not if it _does_ make the prediction.

A new approach known as **[Causal Tracing](https://proceedings.neurips.cc/paper/2020/hash/92650b2e92217715fe312e6fa7b90d82-Abstract.html)** 
addresses this limitation. The objective of this approach is similar to probes: attempting to understand which part of a model contains 
information relevant to a downstream task. The approach involves iterating through all parts of the model being examined (e.g. all layers
of a model), and disrupting the information flow through that part of the model. (This could be as easy as adding some kind of noise on top of the 
weights of that model component). If the model performance on the downstream task suddenly drops on disrupting a specific model component, 
we know for sure that that component not only contains the information required to make the prediction, but that the model is actually using that
information to make the prediction.


:::::::::::::::::::::::::::::::::::::: challenge

Now, it's time to try implementing these methods yourself! Pick one of the following problems to work on:

- [Train your own linear probe to check if BERT stores the required knowledge for sentiment analysis.](https://carpentries-incubator.github.io/fair-explainable-ml/5c-probes.html)
- [Use GradCAM on a trained model to check if the model is using the right features to make predictions.](https://carpentries-incubator.github.io/fair-explainable-ml/5d-gradcam.html)

It's time to get your hands dirty now. Good luck, and have fun!


::::::::::::::::::::::::::::::::::::::::::::::::::

