---
title: "Distribution shift and OOD detection"
teaching: 0
exercises: 0
---

:::::::::::::::::::::::::::::::::::::: questions 

- What is distribution shift and what are its implications in machine learning models?
- How can AI systems be deployed while ensuring that they do not break when encountered with new data?
- What are some of the latest methods in detecting out-of-distribution data during inference?

::::::::::::::::::::::::::::::::::::::::::::::::

::::::::::::::::::::::::::::::::::::: objectives

- Understand the concept of distribution shift in the context of machine learning.
- Identify the factors that can lead to distribution shift.
- Explain the implications of distribution shift on the performance and reliability of machine learning models.
- Discuss strategies for detecting and mitigating distribution shift in machine learning applications.
- Identify best practices for robust model deployment and monitoring.
- Examine the importance of detecting out-of-distribution data in machine learning applications.
- Explore the limitations of traditional methods for detecting out-of-distribution data.
- Investigate state-of-the-art techniques for out-of-distribution detection.

::::::::::::::::::::::::::::::::::::::::::::::::

::::::::::::::::::::::::::::::::::::: keypoints 

- TODO

::::::::::::::::::::::::::::::::::::::::::::::::

Lesson content is being drafted here: https://docs.google.com/document/d/1BpNwJK4jRhnNJCzZSFaueMOB7jKJJHupZ92bLmQlqYQ/edit?usp=sharing

## Distribution shift and its implications

## Overview of out-of-distribution (OOD) detection methods

## Glossary
* ID/OOD: In-distribution, out-of-distribution. Generally, the OOD instances can be defined as instances (x, y) sampled from an underlying distribution other than the training distribution P(Xtrain, Ytrain), where Xtrain and Ytrain are the training corpus and training label set, respectively.
* OOD instances with semantic shift: OOD instances with semantic shift refer to instances that do not belong to y_train. More specifically, instances with semantic shift may come from unknown categories or irrelevant tasks. 
* OOD instances with non-semantic shift: OOD instances with non-semantic shift refer to the instances that belong to Ytrain but are sampled from a distribution other than Xtrain, e.g., a different corpus.
* Closed-world assumption: an assumption that the training and test data are sampled from the same distribution. However, training data can rarely capture the entire distribution. In real-world scenarios, out-of-distribution (OOD) instances, which come from categories that are not known to the model, can often be present in inference phases

## References / Learn more
* Zhou et al., Contrastive Out-of-Distribution Detection for Pretrained Transformers, https://aclanthology.org/2021.emnlp-main.84.pdf 
* Uppaal et al., Is Fine-tuning Needed? Pre-trained Language Models Are Near Perfect for Out-of-Domain Detection; https://aclanthology.org/2023.acl-long.717.pdf 