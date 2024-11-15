---
title: "Estimating model uncertainty"
teaching: 20
exercises: 0
---

:::::::::::::::::::::::::::::::::::::: questions 

- What is model uncertainty, and how can it be categorized?  
- How do uncertainty estimation methods differ from OOD detection methods?  
- What are the computational challenges of estimating model uncertainty?  
- When is uncertainty estimation useful, and what are its limitations?  
- Why is OOD detection often preferred over uncertainty estimation in modern applications?  

::::::::::::::::::::::::::::::::::::::::::::::::

::::::::::::::::::::::::::::::::::::: objectives

- Define and distinguish between aleatoric and epistemic uncertainty in machine learning models.  
- Understand the methods used to estimate model uncertainty, including Monte Carlo dropout, Bayesian neural networks, and model ensembles.  
- Compare and contrast the goals and computational costs of uncertainty estimation and OOD detection.  
- Explore the strengths and limitations of uncertainty estimation methods in real-world applications.  
- Recognize scenarios where uncertainty estimation may still be valuable despite its computational cost.
  
::::::::::::::::::::::::::::::::::::::::::::::::


### Estimating model uncertainty
We currently provide a high-level overview of uncertainty estimation. Depending on demand, we may expand this section in the future to include specific techniques and their practical applications.

Understanding how confident a model is in its predictions is a valuable tool for building trustworthy AI systems, especially in high-stakes settings like healthcare or autonomous vehicles. Model uncertainty estimation focuses on quantifying the model's confidence and is often used to identify predictions that require further review or caution.

Model uncertainty can be divided into two categories:

- **Aleatoric uncertainty**: Inherent noise in the data (e.g., overlapping classes) that cannot be reduced, even with more data.
- **Epistemic uncertainty**: Gaps in the model’s knowledge about the data distribution, which can be reduced by using more data or improved models.

Common techniques for uncertainty estimation include **Monte Carlo dropout**, **Bayesian neural networks**, and **model ensembles**. While these methods provide valuable insights, they are often computationally expensive. For instance:

- Monte Carlo dropout requires performing multiple forward passes through the model for each prediction.
- Ensembles require training and running multiple models, effectively multiplying the computational cost by the size of the ensemble.
- Bayesian approaches, while theoretically sound, are computationally prohibitive for large datasets or complex models, making them challenging to scale.

#### How does this compare to OOD detection?

While uncertainty estimation highlights **low-confidence predictions**, **out-of-distribution (OOD) detection** focuses on identifying inputs that differ significantly from the training data. These inputs are more likely to produce unreliable predictions because the model has never encountered similar data. In practice, OOD methods are often more computationally efficient and scalable compared to uncertainty estimation, as they typically involve a single evaluation pass rather than multiple computations.

As a result, OOD detection is becoming the preferred approach in many cases, particularly for tasks requiring robust detection of anomalous inputs.

#### Weaknesses and use cases

Uncertainty estimation methods have several limitations:

- They require access to the model's internal structure (white-box methods), making them less applicable for black-box systems.
- Their computational cost makes them impractical for applications requiring real-time predictions or working with large datasets.
- They may struggle to distinguish between uncertainty due to **noisy data** and inputs that are truly **out of distribution**.
- Despite their complexity, these methods often lack interpretability for stakeholders.

Uncertainty estimation is still valuable in scenarios where detailed confidence levels are needed, such as calibrating predictions, improving model robustness, or combining it with OOD detection for better decision-making. However, for many modern use cases, OOD methods provide a more efficient and scalable alternative.

