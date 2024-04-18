---
title: "Scientific validity in the modeling process"
teaching: 0
exercises: 0
---

:::::::::::::::::::::::::::::::::::::: questions 

- What impact does overfitting and underfitting have on model performance?
- What is data leakage? 

::::::::::::::::::::::::::::::::::::::::::::::::

::::::::::::::::::::::::::::::::::::: objectives

- Implement at least two types of machine learning models in Python.
- Describe the risks of, identify, and understand mitigation steps for overfitting and underfitting.
- Understand why data leakage is harmful to scientific validity and how it can appear in machine learning pipelines.

::::::::::::::::::::::::::::::::::::::::::::::::

::::::::::::::::::::::::::::::::::::: keypoints 

- Overfitting is characterized by worse performance on the test set than on the train set and can be fixed by switching to a simpler model architecture or by adding regularization.
- Underfitting is characterized by poor performance on both the training and test datasets. It can be fixed by collecting more training data, switching to a more complex model architecture, or improving feature quality.
- Data leakage occurs when the model has access to the test data during training and results in overconfidence in the model's performance. 

::::::::::::::::::::::::::::::::::::::::::::::::
