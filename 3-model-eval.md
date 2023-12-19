---
title: "Model evaluation and fairness"
teaching: 0
exercises: 0
---

:::::::::::::::::::::::::::::::::::::: questions 

- How do we define fairness and bias in machine learning outcomes?
- What types of bias and unfairness can occur in generative AI?
- How can we improve the fairness of machine learning models? 

::::::::::::::::::::::::::::::::::::::::::::::::

::::::::::::::::::::::::::::::::::::: objectives
- Reason about model performance through standard evaluation metrics.
- Understand and distinguish between various notions of fairness in machine learning.
- Describe and implement two different ways of modifying the machine learning modeling process to improve the fairness of a model.

::::::::::::::::::::::::::::::::::::::::::::::::

## Accuracy metrics

Stakeholders often want to know the accuracy of a machine learning model -- what percent of predictions are correct? Accuracy can be decomposed into further metrics: e.g., in a binary prediction setting, recall (the fraction of positive samples that are classified correctly) and precision (the fraction of samples classified as positive that actually are positive) are commonly-used metrics. 

**TO DO** put table (confusion matrix) here with metrics and discuss what different entries mean

:::::::::::::::::::::::::::::::::::::: challenge

### What accuracy metric to use?

Different accuracy metrics may be more relevant in different situations. Discuss with a partner or small groups whether precision, recall, or some combination of the two is most relevant in the following prediction tasks:

1. Deciding what patients are high risk for a disease and who should get additional low-cost screening.
2. Deciding what patients are high risk for a disease and should start taking medication to lower the disease risk. The medication is expensive and can have unpleasant side effects. 
3. **TODO**

::::::::::::::::::::::::::::::::::::::::::::::::::

:::::::::::::: solution

### Solution

1. It is best if all patients who need the screening get it, and there is little downside for doing screenings unnecessarily because the screening costs are low. Thus, a high recall score is optimal.

2. Given the costs and side effects of the medicine, we do not want patients not at risk for the disease to take the medication. So, a high precision score is ideal.

3. **TODO**

:::::::::::::::::::::::::


## Fairness metrics 

What does it mean for a machine learning model to be fair? There is no single definition of fairness, adn it stems beyond data, model internals, and model output to how a model is deployed in practice. But the aggregate model outputs can be used to gain an overall understanding of how models behave with respect to different demographic groups -- an approach called group fairness. 

In general, if there are no differences between groups, achieving fairness is easy. But, in practice, in many social settings wehre prediction tools are used, there are differences between groups, e.g., due to historical and current discrimination. 

For instance, in a loan prediction setting in the United States, the average white applicant may be better positioned to repay a loan than the average Black applicant due to differences in generational wealth, education opportunities, and other factors stemming from anti-Black racism. If, say, 50% of white applicants are granted a loan, with a precision of 90% and a recall of 70% -- in other words, 90% of white people granted loans end up repaying them, and 70% of all people who would have repayed the loan, if given the opportunity, get the loan. Consider the following scenarios:

* (Demographic parity) We give loans to 50% of Black applicants in a way that maximizes overall accurcy
* (Equalized odds) We give loans to X% of Black applicants, where X is chosen to maximize accuracy subject to keeping precision equal to 90%. 
* (Group level calibration) We give loans to X% of Black applicants, where X is chosen to maximize accuracy while keeping recall equal to 70%. 

There are *many* notions of statistical group fainress, but most boil down to one of the three above options: demographic parity, equalized  odds, and group-level calibration.

**TODO** need example here, case study

**TODO** need discussion of individual fainress (especially if we keep the challenge below)

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


## Fairness in generative AI


## Improving fairness of models


::::::::::::::::::::::::::::::::::::: keypoints 

- It's important to consider many dimensions of model performance: a single accuracy score is not sufficient.
- There is no single definition of "fair machine learning": different notions of fairness are appropriate in different contexts.
- Representational harms and stereotypes can be perpetuated by generative AI.
- The fairness of a model can be improved by using techniques like data reweighting and model postprocessing.

::::::::::::::::::::::::::::::::::::::::::::::::