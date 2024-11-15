---
title: "Estimating model uncertainty: overview"
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
- **Epistemic uncertainty**: Gaps in the model’s knowledge about the data distribution, which can be reduced by using more data or improved models.

#### Common techniques for estimating aleatoric uncertainty

Aleatoric uncertainty arises from the data itself. Methods to estimate it include:
- **Predictive variance in regression models**: Outputs the variance of the predicted value, reflecting the noise in the data. For instance, in a regression task predicting house prices, predictive variance highlights how much randomness exists in the relationship between input features (like square footage) and price.  
- **Heteroscedastic models**: Use specialized loss functions that allow the model to predict the noise level in the data directly. These models account for varying uncertainty levels across different inputs, such as distinguishing between noisier and cleaner data points in an image classification task.  
- **Data augmentation and perturbation analysis**: Assess variability in predictions by adding noise to the input data and observing how much the model’s outputs change. A highly sensitive change in predictions may indicate underlying noise or instability in the data.  

#### Common techniques for estimating epistemic uncertainty

Epistemic uncertainty arises from the model's lack of knowledge about certain regions of the data space. Techniques to estimate it include:
- **Monte Carlo dropout**: In this method, dropout (a regularization technique that randomly disables some neurons) is applied during inference, and multiple forward passes are performed for the same input. The variability in the outputs across these passes gives an estimate of uncertainty. Intuitively, each forward pass simulates a slightly different version of the model, akin to an ensemble. If the model consistently predicts similar outputs despite dropout, it is confident; if predictions vary widely, the model is uncertain about that input.

- **Bayesian neural networks**: These networks incorporate probabilistic layers to model uncertainty directly in the weights of the network. Instead of having a single deterministic weight for each connection, the weights are distributions, reflecting the uncertainty about the model's parameters. During inference, these distributions are sampled to generate predictions, which naturally include uncertainty estimates. While theoretically rigorous, Bayesian neural networks are computationally intensive and challenging to implement, making them less common in practice.

- **Ensemble models**: These involve training multiple models on the same data, each starting with different initializations or random seeds. The ensemble’s predictions are aggregated, and the variance in their outputs reflects uncertainty. This approach works well because different models often capture different aspects of the data. For example, if all models agree, the prediction is confident; if they disagree, there is uncertainty. Ensembles are effective but computationally expensive, as they require training and evaluating multiple models.

- **Out-of-distribution detection**: Identifies inputs that fall significantly outside the training distribution, flagging areas where the model’s predictions are unreliable. For instance, OOD methods may detect that a cat classifier is being applied to an image of a car, an input it has never seen before. Unlike other methods, OOD detection doesn’t provide fine-grained confidence scores but excels at rejecting anomalous inputs.

#### Summary table

| **Method**                | **Type of uncertainty**  | **Key strengths**                                             | **Key limitations**                                   |
|---------------------------|-------------------------|-------------------------------------------------------------|----------------------------------------------------------|
| **Predictive variance**    | Aleatoric              | Simple, intuitive for regression tasks                     | Limited to regression problems; doesn’t address epistemic uncertainty |
| **Heteroscedastic models** | Aleatoric              | Models variable noise across inputs                        | Requires specialized architectures or loss functions      |
| **Monte Carlo dropout**    | Epistemic             | Easy to implement in existing neural networks              | Computationally expensive due to multiple forward passes  |
| **Bayesian neural nets**   | Epistemic             | Rigorous probabilistic foundation                          | Computationally prohibitive for large models/datasets     |
| **Ensemble models**        | Epistemic             | Effective and robust; captures diverse uncertainties       | Resource-intensive; requires training multiple models     |
| **OOD detection**          | Epistemic             | Efficient, scalable, excels at rejecting anomalous inputs  | Limited to identifying OOD inputs, not fine-grained uncertainty |

#### Why is OOD detection widely adopted?

Among epistemic uncertainty methods, **OOD detection** has become a widely adopted approach in real-world applications due to its ability to efficiently identify inputs that fall outside the training data distribution, where predictions are inherently unreliable. Compared to methods like Monte Carlo dropout or Bayesian neural networks, which require multiple forward passes or computationally expensive probabilistic frameworks, many OOD detection techniques are lightweight and scalable. 

For example, in autonomous vehicles, OOD detection can help flag unexpected scenarios (e.g., unusual objects on the road) in near real-time, enabling safer decision-making. Similarly, in NLP, OOD methods are used to identify queries or statements that deviate from a model’s training corpus, such as out-of-context questions in a chatbot system. 

While OOD detection excels at flagging anomalous inputs, it does not provide fine-grained uncertainty estimates for in-distribution data, making it best suited for tasks where the primary concern is identifying outliers or novel inputs.

#### Summary

While uncertainty estimation provides a broad framework for understanding model confidence, different methods are suited for specific types of uncertainty and use cases. OOD detection stands out as the most practical approach for handling epistemic uncertainty in modern applications, thanks to its efficiency and ability to reject anomalous inputs. Together, these methods form a complementary toolkit for building trustworthy AI systems.
