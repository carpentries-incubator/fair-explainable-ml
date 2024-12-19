---
title: "Estimating model uncertainty"
teaching: 40
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
## How confident is my model? Will it generalize to new data?
Understanding how confident a model is in its predictions is a valuable tool for building trustworthy AI systems, especially in high-stakes settings like healthcare or autonomous vehicles. Model uncertainty estimation focuses on quantifying the model's confidence and is often used to identify predictions that require further review or caution.

## Sources of uncertainty 
At its core, model uncertainty starts with the **data** itself, as all models learn to form embeddings (feature representations) of the data. Uncertainty in the data—whether from inherent randomness or insufficient coverage—propagates through the model's embeddings, leading to uncertainty in the outputs. 

#### 1) Aleatoric (Random) uncertainty
Aleotoric or random uncertainty is the inherent noise in the data that cannot be reduced, even with more data (observations OR missing features). 

  - Inconsistent readings from faulty sensors (e.g., modern image sensors exhibit "thermal noise" or "shot noise", where pixel values randomly fluctuate even under constant lighting)
  - Random crackling/static in recordings
  - Human errors in data entry
  - Any aspect of the data that is unpredictable
    
#### Methods for addressing aleatoric uncertainty
Since aleatoric/random uncertainty is generally considered inherent (unless you upgrade sensors or remove whatever is causing the random generating process), methods to address it focus on measuring the degree of noise or uncertainty.

- **Predictive variance in linear regression**: The ability to derive error bars or prediction intervals in traditional regression comes from the assumption that the errors (residuals) follow a normal distribution and are homoskedastic (errors stay relatively constant across different values of predictors).
  - In contrast, deep learning models are highly non-linear and have millions (or billions) of parameters. The mapping between inputs and outputs is not a simple linear equation but rather a complex, multi-layer function. In addition, deep learning can overfit common classes and underfit rarer clases. Because of these factors, errors are rarely normally distributed and homoskedastic in deep learning applications.
- **Heteroskedastic models**: Use specialized loss functions that allow the model to predict the noise level in the data directly. These models are particularly critical in fields like *robotics*, where sensor noise varies significantly depending on environmental conditions. It is possible to build this functionality into both linear models and modern deep learning models. However, these methods may require some calibration, as ground truth measurements of noise usually aren't available.
  - Example application: Managing hospital reporting inconsistencies.  
  - Reference: Kendall, A., & Gal, Y. (2017). "[What uncertainties do we need in Bayesian deep learning for computer vision?](https://arxiv.org/abs/1703.04977)".
- **Data augmentation and perturbation analysis**: Assess variability in predictions by adding noise to the input data and observing how much the model’s outputs change. A highly sensitive change in predictions may indicate underlying noise or instability in the data. For instance, in image classification, augmenting training data with synthetic noise can help the model better handle real-world imperfections stemming from sensor artifacts. 
  - Example application: Handling motion blur in tumor detection.  
  - Reference: Shorten, C., & Khoshgoftaar, T. M. (2019). "[A survey on image data augmentation for deep learning.](https://journalofbigdata.springeropen.com/articles/10.1186/s40537-019-0197-0)"  

#### 2) Subjectivity and ill-defined problems 

- Overlapping classes, ambiguous labels due to subjective interpretations
- Ambiguous or conflicting text inputs.

#### Methods for addressing subjectivity and ill-defined problems 
- **Reframe problme**: If the overlap or subjectivity stems from an ill-posed problem, reframing the task can help. Example: Instead of classifying "happy" vs. "neutral" expressions (which overlap), predict the intensity of happiness on a scale of 0–1. For medical images, shift from hard "benign vs. malignant" classifications to predicting risk scores.
- **Consensus-based labeling (inter-annotator agreement)**: Aggregate labels from multiple annotators to reduce subjectivity and quantify ambiguity. Use metrics like Cohen's kappa or Fleiss' kappa to measure agreement between annotators. Example: In medical imaging (e.g., tumor detection), combining expert radiologists’ opinions can reduce subjective bias in labeling.
- **Probabilistic labeling or soft targets**: Instead of using hard labels (e.g., 0 or 1), assign probabilistic labels to account for ambiguity in the data. Example: If 70% of annotators labeled an image as "happy" and 30% as "neutral," you can label it as [0.7, 0.3] instead of forcing a binary decision.

    
#### 3. Epistemic uncertainty 
**Epistemic** (ep·i·ste·mic) is an adjective that means, "*relating to knowledge or to the degree of its validation.*" 

Epistemic uncertainty refers to gaps in the model's knowledge about the data distribution, which can be reduced by using more data or improved models. Aleatoric uncertainy can arise due to:

- **Out-of-distribution (OOD) data**:
  - Tabular: Classifying user behavior from a new region not included in training data. Predicting hospital demand during a rare pandemic with limited historical data. Applying model trained on one location to another.
  - Image: Recognizing a new species in wildlife monitoring. Detecting a rare/unseen obstacle to automate driving. A model trained on high-resolution images but tested on low-resolution inputs.
  - Text: Queries about topics completely outside the model's domain (e.g., financial queries in a healthcare chatbot).  Interpreting slang or idiomatic expressions unseen during training.

- **Sparse or insufficient data in feature space**:
  - Tabular: High-dimensional data with many missing or sparsely sampled features (e.g., genomic datasets).
  - Image: Limited labeled examples for rare diseases in medical imaging datasets.
  - Text: Rare domain-specific terminology.

#### Methods for addressing epistemic uncertainty

Epistemic uncertainty arises from the model's lack of knowledge about certain regions of the data space. Techniques to address this uncertainty include:

- **Collect more data**: Easier said than done! Focus on gathering data from underrepresented scenarios or regions of the feature space, particularly areas where the model exhibits high uncertainty (e.g., rare medical conditions, edge cases in autonomous driving). This directly reduces epistemic uncertainty by expanding the model's knowledge base.
  - **Active learning**: Use model uncertainty estimates to prioritize uncertain or ambiguous samples for annotation, enabling more targeted data collection.
- **Ensemble models**: These involve training multiple models on the same data, each starting with different initializations or random seeds. The ensemble's predictions are aggregated, and the variance in their outputs reflects uncertainty. This approach works well because different models often capture different aspects of the data. For example, if all models agree, the prediction is confident; if they disagree, there is uncertainty. Ensembles are effective but computationally expensive, as they require training and evaluating multiple models.
- **Bayesian neural networks**: These networks incorporate probabilistic layers to model uncertainty directly in the weights of the network. Instead of assigning a single deterministic weight to each connection, Bayesian neural networks assign distributions to these weights, reflecting the uncertainty about their true values. During inference, these distributions are sampled multiple times to generate predictions, which naturally include uncertainty estimates. While Bayesian neural networks are theoretically rigorous and align well with the goal of epistemic uncertainty estimation, they are computationally expensive and challenging to scale for large datasets or deep architectures. This is because calculating or approximating posterior distributions over all parameters becomes intractable as model size grows. To address this, methods like variational inference or Monte Carlo sampling are often used, but these approximations can introduce inaccuracies, making Bayesian approaches less practical for many modern applications. Despite these challenges, Bayesian neural networks remain valuable for research contexts where precise uncertainty quantification is needed or in domains where computational resources are less of a concern.
  - Example application: Detecting rare tumor types in radiology.  
  - Reference: Blundell, C., et al. (2015). "[Weight uncertainty in neural networks.](https://arxiv.org/abs/1505.05424)"
- **Out-of-distribution detection**: Identifies inputs that fall significantly outside the training distribution, flagging areas where the model's predictions are unreliable. Many OOD methods produce continuous scores, such as Mahalanobis distance or energy-based scores, which measure how novel or dissimilar an input is from the training data. These scores can be interpreted as a form of epistemic uncertainty, providing insight into how unfamiliar an input is. However, OOD detection focuses on distinguishing ID from OOD inputs rather than offering confidence estimates for predictions on ID inputs.
  - Example application: Flagging out-of-scope queries in chatbot systems.  
  - Reference: Hendrycks, D., & Gimpel, K. (2017). "[A baseline for detecting misclassified and out-of-distribution examples in neural networks.](https://arxiv.org/abs/1610.02136)"  


#### Why is OOD detection widely adopted?

Among epistemic uncertainty methods, OOD detection has become a widely adopted approach in real-world applications due to its ability to efficiently identify inputs that fall outside the training data distribution, where predictions are inherently unreliable. Many OOD detection techniques produce continuous scores that quantify the novelty or dissimilarity of inputs, which can be interpreted as a form of uncertainty. This makes OOD detection not only effective at rejecting anomalous inputs but also useful for prioritizing inputs based on their predicted risk.

For example, in autonomous vehicles, OOD detection can help flag unexpected scenarios (e.g., unusual objects on the road) in near real-time, enabling safer decision-making. Similarly, in NLP, OOD methods are used to identify queries or statements that deviate from a model's training corpus, such as out-of-context questions in a chatbot system. In the next couple of episodes, we'll see how to implement various OOD strategies.

:::: challenge

## Identify aleatoric and epistemic uncertainty

For each scenario below, identify the sources of **aleatoric** and **epistemic** uncertainty. Provide specific examples based on the context of the application.

1. **Tabular data example:** Hospital resource allocation during seasonal flu outbreaks and pandemics.  
2. **Image data example:** Tumor detection in radiology images.  
3. **Text data example:** Chatbot intent recognition.  

::::

::::: solution

1. **Hospital resource allocation**

- **Aleatoric uncertainty**: Variability in seasonal flu demand; inconsistent local reporting.  
- **Epistemic uncertainty**: Limited data for rare pandemics; incomplete understanding of emerging health crises.

2. **Tumor detection in radiology images**

- **Aleatoric uncertainty**: Imaging artifacts such as noise or motion blur.  
- **Epistemic uncertainty**: Limited labeled data for rare tumor types; novel imaging modalities.

3. **Chatbot intent recognition**

- **Aleatoric uncertainty**: Noise in user queries such as typos or speech-to-text errors.  
- **Epistemic uncertainty**: Lack of training data for queries from out-of-scope domains; ambiguity due to unclear or multi-intent queries.

:::::


#### Summary
Uncertainty estimation is a critical component of building reliable and trustworthy machine learning models, especially in high-stakes applications. By understanding the distinction between aleatoric uncertainty (inherent data noise) and epistemic uncertainty (gaps in the model's knowledge), practitioners can adopt tailored strategies to improve model robustness and interpretability.

- Aleatoric uncertainty is irreducible noise in the data itself. Addressing this requires models that can predict variability, such as heteroscedastic loss functions, or strategies like data augmentation to make models more resilient to imperfections.
- Epistemic uncertainty arises from the model's incomplete understanding of the data distribution. It can be mitigated through methods like Monte Carlo dropout, Bayesian neural networks, ensemble models, and Out-of-Distribution (OOD) detection. Among these methods, OOD detection has become a cornerstone for handling epistemic uncertainty in practical applications. Its ability to flag anomalous or out-of-distribution inputs makes it an essential tool for ensuring model predictions are reliable in real-world scenarios.
  - In many cases, collecting more data and employing active learning can directly address the root causes of epistemic uncertainty.

When choosing a method, it’s important to consider the trade-offs in computational cost, model complexity, and the type of uncertainty being addressed. Together, these techniques form a powerful toolbox, enabling models to better navigate uncertainty and maintain trustworthiness in dynamic environments. By combining these approaches strategically, practitioners can ensure that their systems are not only accurate but also robust, interpretable, and adaptable to the challenges of real-world data.
