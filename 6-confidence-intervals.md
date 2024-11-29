---
title: "Estimating model uncertainty"
teaching: 15
exercises: 0
---

:::::::::::::::::::::::::::::::::::::: questions 

- What is model uncertainty, and how can it be categorized?  
- How do uncertainty estimation methods intersect with OOD detection methods?  
- What are the computational challenges of estimating model uncertainty?  
- When is uncertainty estimation useful, and what are its limitations?  
- Why is OOD detection often preferred over traditional uncertainty estimation techniques in modern applications?  

::::::::::::::::::::::::::::::::::::::::::::::::

::::::::::::::::::::::::::::::::::::: objectives

- Define and distinguish between aleatoric and epistemic uncertainty in machine learning models.  
- Explore common techniques for estimating aleatoric and epistemic uncertainty.  
- Understand why OOD detection has become a widely adopted approach in many real-world applications.  
- Compare and contrast the goals and computational costs of uncertainty estimation and OOD detection.  
- Summarize when and where different uncertainty estimation methods are most useful.  

::::::::::::::::::::::::::::::::::::::::::::::::


### Estimating model uncertainty

Understanding how confident a model is in its predictions is a valuable tool for building trustworthy AI systems, especially in high-stakes settings like healthcare or autonomous vehicles. Model uncertainty estimation focuses on quantifying the model's confidence and is often used to identify predictions that require further review or caution.

Model uncertainty can be divided into two categories:

- **Aleatoric uncertainty**: Inherent noise in the data (e.g., overlapping classes) that cannot be reduced, even with more data.
- **Epistemic uncertainty**: Gaps in the model's knowledge about the data distribution, which can be reduced by using more data or improved models.

#### Common techniques for estimating aleatoric uncertainty

Aleatoric uncertainty arises from the data itself. Methods to estimate it include:

- **Predictive variance in regression models**: Outputs the variance of the predicted value, reflecting the noise in the data. For instance, in a regression task predicting house prices, predictive variance highlights how much randomness exists in the relationship between input features (like square footage) and price.  
- **Heteroscedastic models**: Use specialized loss functions that allow the model to predict the noise level in the data directly. These models are particularly critical in fields like **robotics**, where sensor noise varies significantly depending on environmental conditions. For example, a robot navigating in bright daylight versus dim lighting conditions may experience vastly different levels of noise in its sensor inputs, and heteroscedastic models can help account for this variability.  
- **Data augmentation and perturbation analysis**: Assess variability in predictions by adding noise to the input data and observing how much the model’s outputs change. A highly sensitive change in predictions may indicate underlying noise or instability in the data. For instance, in image classification, augmenting training data with synthetic noise can help the model better handle real-world imperfections like motion blur or occlusions.

#### Common techniques for estimating epistemic uncertainty

Epistemic uncertainty arises from the model's lack of knowledge about certain regions of the data space. Techniques to estimate it include:

- **Monte Carlo dropout**: In this method, dropout (a regularization technique that randomly disables some neurons) is applied during inference, and multiple forward passes are performed for the same input. The variability in the outputs across these passes gives an estimate of uncertainty. Intuitively, each forward pass simulates a slightly different version of the model, akin to an ensemble. If the model consistently predicts similar outputs despite dropout, it is confident; if predictions vary widely, the model is uncertain about that input.
- **Bayesian neural networks**: These networks incorporate probabilistic layers to model uncertainty directly in the weights of the network. Instead of assigning a single deterministic weight to each connection, Bayesian neural networks assign distributions to these weights, reflecting the uncertainty about their true values. During inference, these distributions are sampled multiple times to generate predictions, which naturally include uncertainty estimates. While Bayesian neural networks are theoretically rigorous and align well with the goal of epistemic uncertainty estimation, they are computationally expensive and challenging to scale for large datasets or deep architectures. This is because calculating or approximating posterior distributions over all parameters becomes intractable as model size grows. To address this, methods like variational inference or Monte Carlo sampling are often used, but these approximations can introduce inaccuracies, making Bayesian approaches less practical for many modern applications. Despite these challenges, Bayesian neural networks remain valuable for research contexts where precise uncertainty quantification is needed or in domains where computational resources are less of a concern.
- **Ensemble models**: These involve training multiple models on the same data, each starting with different initializations or random seeds. The ensemble's predictions are aggregated, and the variance in their outputs reflects uncertainty. This approach works well because different models often capture different aspects of the data. For example, if all models agree, the prediction is confident; if they disagree, there is uncertainty. Ensembles are effective but computationally expensive, as they require training and evaluating multiple models.
- **Out-of-distribution detection**: Identifies inputs that fall significantly outside the training distribution, flagging areas where the model's predictions are unreliable. Many OOD methods produce continuous scores, such as Mahalanobis distance or energy-based scores, which measure how novel or dissimilar an input is from the training data. These scores can be interpreted as a form of epistemic uncertainty, providing insight into how unfamiliar an input is. However, OOD detection focuses on distinguishing ID from OOD inputs rather than offering confidence estimates for predictions on ID inputs.

### Method selection summary table

:::::::::::::::::::::::::::::::::::::: callout
#### Understanding size categories in table

To help guide method selection, here are rough definitions for **model size**, **data size**, and **compute requirements** used in the table:

**Model size**

- **Small**: Fewer than 10M parameters (e.g., logistic regression, LeNet).
- **Medium**: 10M–100M parameters (e.g., ResNet-50, BERT-base).
- **Large**: More than 100M parameters (e.g., GPT-3, Vision Transformers).

**Data size**

- **Small**: Fewer than 10,000 samples (e.g., materials science datasets).
- **Medium**: 10,000–1M samples (e.g., ImageNet).
- **Large**: More than 1M samples (e.g., Common Crawl, LAION-5B).

**Compute time** (approximate)

- **Low**: Suitable for standard CPU or single GPU, training/inference in minutes to an hour.
- **Medium**: Requires a modern GPU, training/inference in hours to a day.
- **High**: Requires multiple GPUs/TPUs or distributed setups, training/inference in days to weeks.
::::::::::::::::::::::::::::::::::::::

| Method                  | Type of uncertainty | Key strengths                               | Key limitations                                    | Model size restrictions         | Data size restrictions          | Compute time (approx.)          |
|-------------------------|---------------------|--------------------------------------------|--------------------------------------------------|---------------------------------|---------------------------------|----------------------------------|
| Predictive variance     | Aleatoric          | Simple, intuitive for regression tasks     | Limited to regression problems; doesn’t address epistemic uncertainty | Small to medium                 | Small to medium                 | Very low (single pass)           |
| Heteroscedastic models  | Aleatoric          | Models variable noise across inputs        | Requires specialized architectures or loss functions | Medium to large (task-dependent) | Medium                          | Medium (depends on task complexity) |
| Monte Carlo dropout     | Epistemic          | Easy to implement in existing neural networks | Computationally expensive due to multiple forward passes | Medium                         | Medium to large                 | High (scales with forward passes) |
| Bayesian neural nets    | Epistemic          | Rigorous probabilistic foundation          | Computationally prohibitive for large models/datasets | Small to medium (challenging to scale) | Small to medium               | Very high (depends on sampling)  |
| Ensemble models         | Epistemic          | Effective and robust; captures diverse uncertainties | Resource-intensive; requires training multiple models | Medium to large                 | Medium to large                 | Very high (training multiple models) |
| OOD detection           | Epistemic          | Efficient, scalable, excels at rejecting anomalous inputs | Limited to identifying OOD inputs, not fine-grained uncertainty | Medium to large                 | Small to large                  | Low to medium (scales efficiently) |


#### Why is OOD detection widely adopted?

Among epistemic uncertainty methods, OOD detection has become a widely adopted approach in real-world applications due to its ability to efficiently identify inputs that fall outside the training data distribution, where predictions are inherently unreliable. Many OOD detection techniques produce continuous scores that quantify the novelty or dissimilarity of inputs, which can be interpreted as a form of uncertainty. This makes OOD detection not only effective at rejecting anomalous inputs but also useful for prioritizing inputs based on their predicted risk.

For example, in autonomous vehicles, OOD detection can help flag unexpected scenarios (e.g., unusual objects on the road) in near real-time, enabling safer decision-making. Similarly, in NLP, OOD methods are used to identify queries or statements that deviate from a model's training corpus, such as out-of-context questions in a chatbot system. In the next couple of episodes, we'll see how to implement various OOD strategies.

#### Summary

While uncertainty estimation provides a broad framework for understanding model confidence, different methods are suited for specific types of uncertainty and use cases. OOD detection stands out as the most practical approach for handling epistemic uncertainty in modern applications, thanks to its efficiency and ability to reject anomalous inputs. Together, these methods form a complementary toolkit for building trustworthy AI systems.
