---
title: "Fairness"
teaching: 0
exercises: 0
---

:::::::::::::::::::::::::::::::::::::: questions 

- How do we define fairness and bias in machine learning outcomes?
- How can we improve the fairness of machine learning models? 

::::::::::::::::::::::::::::::::::::::::::::::::

::::::::::::::::::::::::::::::::::::: objectives
- Reason about model performance through standard evaluation metrics.
- Understand and distinguish between various notions of fairness in machine learning.
- Describe and implement two different ways of modifying the machine learning modeling process to improve the fairness of a model.

::::::::::::::::::::::::::::::::::::::::::::::::

:::::::::::::::::::::::::::::::::::::: challenge

### Matching fairness terminology with definitions

Match the following types of formal fairness with their definitions.
(A) Individual fairness,
(B) Equalized odds,
(C) Demographic parity, and 
(D) Group-level calibration

1. The model is equally accurate across all demographic groups. 
2. Different demographic groups have the same true positive rates and false positive rates. 
3. Similar people are treated similarly.
4. People from different demographic groups receive each outcome at the same rate.
::::::::::::::::::::::::::::::::::::::::::::::::::

:::::::::::::: solution

### Solution

A - 3, B - 2, C - 4, D - 1

:::::::::::::::::::::::::


::::::::::::::::::::::::::::::::::::: keypoints 

- It's important to consider many dimensions of model performance: a single accuracy score is not sufficient.
- There is no single definition of "fair machine learning": different notions of fairness are appropriate in different contexts.
- It is usually not possible to satisfy all possible notions of fairness.
- The fairness of a model can be improved by using techniques like data reweighting and model postprocessing.

::::::::::::::::::::::::::::::::::::::::::::::::