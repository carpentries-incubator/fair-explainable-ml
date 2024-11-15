---
title: "OOD detection: overview"
teaching: 0
exercises: 0
---

:::::::::::::::::::::::::::::::::::::::: questions

- What are out-of-distribution (OOD) data and why is detecting them important in machine learning models?
- What are the two broad classes of OOD detection methods: threshold-based and training-time regularization?

::::::::::::::::::::::::::::::::::::::::::::::::::

::::::::::::::::::::::::::::::::::::::: objectives

- Understand the concept of out-of-distribution data and its significance in building trustworthy machine learning models.
- Learn the two main approaches to OOD detection: threshold-based methods and training-time regularization.
- Identify the strengths and limitations of these approaches at a high level.

::::::::::::::::::::::::::::::::::::::::::::::::::

# Introduction to Out-of-Distribution (OOD) Data
## What is OOD data?
Out-of-distribution (OOD) data refers to data that significantly differs from the training data on which a machine learning model was built. For example, the image below compares the training data distribution of CIFAR-10, a popular dataset used for image classification, with the vastly broader and more diverse distribution of images found on the internet:

![OpenAI: CIFAR-10 training distribution vs. internet](https://raw.githubusercontent.com/carpentries-incubator/fair-explainable-ml/main/images/OOD-internet-vs-CIFAR10.jpg)

CIFAR-10 contains 60,000 images across 10 distinct classes (e.g., airplanes, dogs, trucks), with carefully curated examples for each class. However, the internet features an essentially infinite variety of images, many of which fall outside these predefined classes or include unseen variations (e.g., new breeds of dogs or novel vehicle designs). This contrast highlights the challenges models face when they encounter data that significantly differs from their training distribution.

## How OOD data manifests in ML pipelines
The difference between in-distribution (ID) and OOD data can arise from:

- **Semantic shift**: The OOD sample belongs to a class that was not present during training.
- **Covariate shift**: The OOD sample comes from a domain where the input feature distribution is drastically different from the training data.

## Why does OOD data matter?
Models trained on a specific distribution might make incorrect predictions on OOD data, leading to unreliable outputs. In critical applications (e.g., healthcare, autonomous driving), encountering OOD data without proper handling can have severe consequences.

### Ex1: Tesla crashes into jet
In April 2022, a [Tesla Model Y crashed into a $3.5 million private jet](https://www.newsweek.com/video-tesla-smart-summon-mode-ramming-3m-jet-viewed-34m-times-1700310 ) at an aviation trade show in Spokane, Washington, while operating on the "Smart Summon" feature. The feature allows Tesla vehicles to autonomously navigate parking lots to their owners, but in this case, it resulted in a significant mishap.
- The Tesla was summoned by its owner using the Tesla app, which requires holding down a button to keep the car moving. The car continued to move forward even after making contact with the jet, pushing the expensive aircraft and causing notable damage.
- The crash highlighted several issues with Tesla's Smart Summon feature, particularly its object detection capabilities. The system failed to recognize and appropriately react to the presence of the jet, a problem that has been observed in other scenarios where the car's sensors struggle with objects that are lifted off the ground or have unusual shapes.

### Ex2: IBM Watson for Oncology
Around a decade ago, the excitement surrounding AI in healthcare often exceeded its actual capabilities. In 2016, IBM launched Watson for Oncology, an AI-powered platform for treatment recommendations, to much public enthusiasm. However, it soon became apparent that the system was both costly and unreliable, frequently generating flawed advice while operating as an opaque "black box. IBM Watson for Oncology faced several issues due to OOD data. The system was primarily trained on data from Memorial Sloan Kettering Cancer Center (MSK), which did not generalize well to other healthcare settings. This led to the following problems:

1. Unsafe Recommendations: Watson for Oncology provided treatment recommendations that were not safe or aligned with standard care guidelines in many cases outside of MSK. This happened because the training data was not representative of the diverse medical practices and patient populations in different regions
2. Bias in Training Data: The system's recommendations were biased towards the practices at MSK, failing to account for different treatment protocols and patient needs elsewhere. This bias is a classic example of an OOD issue, where the model encounters data (patients and treatments) during deployment that significantly differ from its training data
   
By 2022, IBM had taken Watson for Oncology offline, marking the end of its commercial use.

### Ex3: Doctors using GPT3
#### Misdiagnosis and inaccurate medical advice
In various studies and real-world applications, GPT-3 has been shown to generate inaccurate medical advice when faced with OOD data. This can be attributed to the fact that the training data, while extensive, does not cover all possible medical scenarios and nuances, leading to hallucinations or incorrect responses when encountering unfamiliar input.

A [study published by researchers at Stanford](https://hai.stanford.edu/news/generating-medical-errors-genai-and-erroneous-medical-references) found that GPT-3, even when using retrieval-augmented generation, provided unsupported medical advice in about 30% of its statements. For example, it suggested the use of a specific dosage for a defibrillator based on monophasic technology, while the cited source only discussed biphasic technology, which operates differently.

#### Fake medical literature references
Another critical OOD issue is the generation of fake or non-existent medical references by LLMs. When LLMs are prompted to provide citations for their responses, they sometimes generate references that sound plausible but do not actually exist. This can be particularly problematic in academic and medical contexts where accurate sourcing is crucial.

In [evaluations of GPT-3's ability to generate medical literature references](https://hai.stanford.edu/news/generating-medical-errors-genai-and-erroneous-medical-references) , it was found that a significant portion of the references were either entirely fabricated or did not support the claims being made. This was especially true for complex medical inquiries that the model had not seen in its training data.


## Detecting and handling OOD data
Given the problems posed by OOD data, a reliable model should identify such instances, and then:

1. Reject them during inference
2. Ideally, hand these OOD instances to a model trained on a more similar distribution (an in-distribution).
  
The second step is much more complicated/involved since it requires matching OOD data to essentially an infinite number of possible classes. For the current scope of this workshop, we will focus on just the first step.

How can we determine whether a given instance is OOD or ID? Over the past several years, there have been a wide assortment of new methods developed to tackle this task. In this episode, we will cover a few of the most common approaches and discuss advantages/disadvantages of each.

### Threshold-based methods
Threshold-based methods are one of the simplest and most intuitive approaches for detecting out-of-distribution (OOD) data. The central idea is to define a threshold on a certain score or confidence measure, beyond which the data point is considered out-of-distribution. Typically, these scores are derived from the model's output probabilities or other statistical measures of uncertainty. There are two general classes of threshold-based methods: output-based and distance-based.

#### Output-based thresholds
Output-based Out-of-Distribution (OOD) detection refers to methods that determine whether a given input is out-of-distribution based on the output of a trained model. These methods typically analyze the model’s confidence scores, energy scores, or other output metrics to identify data points that are unlikely to belong to the distribution the model was trained on. The main approaches within output-based OOD detection include:

- **Softmax scores**: The softmax output of a neural network represents the predicted probabilities for each class. A common threshold-based method involves setting a confidence threshold, and if the maximum softmax score of an instance falls below this threshold, it is flagged as OOD.
- **Energy**: The energy-based method also uses the network's output but measures the uncertainty in a more nuanced way by calculating an energy score. The energy score typically captures the confidence more robustly, especially in high-dimensional spaces, and can be considered a more general and reliable approach than just using softmax probabilities.

#### Distance-based thresholds
Distance-based methods calculate the distance of an instance from the distribution of training data features learned by the model. If the distance is beyond a certain threshold, the instance is considered OOD. Common distance-based approaches include:

- **Mahalanobis distance:** This method calculates the Mahalanobis distance of a data point from the mean of the training data distribution. A high Mahalanobis distance indicates that the instance is likely OOD.
- **K-nearest neighbors (KNN):** This method involves computing the distance to the k-nearest neighbors in the training data. If the average distance to these neighbors is high, the instance is considered OOD.

We will focus on output-based methods (softmax and energy) in the next episode and then do a deep dive into distance-based methods in a later next episode.

### Training-time regularization methods
Training-time regularization methods modify the training process to explicitly improve the model's ability to detect out-of-distribution (OOD) data. These methods often involve enhancing the model's generalization or robustness by incorporating OOD scenarios during training or optimizing the model's feature representations. 

#### Outlier exposure
Outlier Exposure (OE) is a common technique where the model is trained with additional auxiliary datasets containing OOD samples. The goal is to teach the model to assign low confidence to OOD data while maintaining high confidence for in-distribution (ID) data.

**How it works**:

- Use a curated auxiliary dataset of OOD samples that differ from the training distribution.
- Augment the training loss function to penalize high confidence on these auxiliary samples.
- Resulting models are less likely to misclassify OOD inputs as ID.

**Advantages**:

- Simple to implement when auxiliary datasets are available.
- Significantly improves OOD detection performance without adding much computational overhead.

**Limitations**:

- Requires access to high-quality, diverse OOD datasets during training.
- Performance may degrade for OOD samples dissimilar to the auxiliary dataset.

#### Contrastive learning
Contrastive learning encourages the model to learn more discriminative feature representations, improving its ability to distinguish between ID and OOD data. By pushing apart embeddings of dissimilar samples and pulling together embeddings of similar ones, contrastive learning creates a representation space where OOD samples are less likely to overlap with ID data.

**How it works**:

- Define a contrastive loss that minimizes the distance between embeddings of similar samples (e.g., belonging to the same class).
- Simultaneously maximize the distance between embeddings of dissimilar samples (e.g., ID vs. synthetic or auxiliary OOD samples).
- Often uses data augmentation or self-supervised techniques to generate "positive" and "negative" pairs.

**Advantages**:

- Does not require labeled auxiliary OOD data, as augmentations or unsupervised data can be used.
- Improves the quality of learned representations, benefiting other downstream tasks.

**Limitations**:

- Computationally expensive, especially with large datasets.
- Requires careful tuning of the contrastive loss and data augmentation strategy.

#### Regularization-based methods
Some training-time regularization methods directly modify the loss function to penalize overconfidence or encourage robustness to data variations:

- **Confidence penalties**: Add regularization terms that penalize overconfidence in predictions, especially on ambiguous samples.
- **Adversarial training**: Generate adversarial examples (slightly perturbed ID samples) to improve robustness against small variations that could make ID samples appear OOD.

#### Summary of strengths and weaknesses

**Strengths**:

  - Incorporating OOD detection into training enhances performance without relying solely on post-hoc thresholds.
  - Often leads to better generalization for both ID and OOD scenarios.
- **Weaknesses**:
  - Requires careful design of the training procedure.
  - Computationally intensive and may require access to additional datasets or resources.

We will explore these techniques in detail in later episodes, focusing on implementation strategies and trade-offs.


:::::::::::::::::::::::::::::::::::::::: keypoints

- Out-of-distribution (OOD) data significantly differs from training data and can lead to unreliable model predictions.
- Threshold-based methods use model outputs or distances in feature space to detect OOD instances by defining a score threshold.
- Training-time regularization enhances OOD detection by incorporating techniques like Outlier Exposure and Contrastive Learning during model training.
- Each method has trade-offs: threshold-based methods are simpler, while training-time regularization often improves robustness at higher computational cost.

::::::::::::::::::::::::::::::::::::::::::::::::::