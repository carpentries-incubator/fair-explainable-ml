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

## Estimating model uncertainty
Understanding how confident a model is in its predictions is a valuable tool for building trustworthy AI systems, especially in high-stakes settings like healthcare or autonomous vehicles. Model uncertainty estimation focuses on quantifying the model's confidence and is often used to identify predictions that require further review or caution.

### Sources of uncertainty 
At its core, uncertainty starts with the **data** itself, as all models learn to form embeddings (feature representations) of the data. Uncertainty in the data—whether from inherent randomness or insufficient coverage—propagates through the model's embeddings, leading to uncertainty in the outputs. 

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
- **Data augmentation and perturbation analysis**: Assess variability in predictions by adding noise to the input data and observing how much the model’s outputs change. A highly sensitive change in predictions may indicate underlying noise or instability in the data. For instance, in image classification, augmenting training data with synthetic noise can help the model better handle real-world imperfections stemming from sensor artifacts. 

#### 2) Subjectivity and ill-defined problems 

- Overlapping classes, ambiguous labels due to subjective interpretations
- Ambiguous or conflicting text inputs.

#### Methods for addressing subjectivity and ill-defined problems 
- **Reframe problme**: If the overlap or subjectivity stems from an ill-posed problem, reframing the task can help. Example: Instead of classifying "happy" vs. "neutral" expressions (which overlap), predict the intensity of happiness on a scale of 0–1. For medical images, shift from hard "benign vs. malignant" classifications to predicting risk scores.
- **Consensus-based labeling (inter-annotator agreement)**: Aggregate labels from multiple annotators to reduce subjectivity and quantify ambiguity. Use metrics like Cohen's kappa or Fleiss' kappa to measure agreement between annotators. Example: In medical imaging (e.g., tumor detection), combining expert radiologists’ opinions can reduce subjective bias in labeling.
- **Probabilistic labeling or soft targets**: Instead of using hard labels (e.g., 0 or 1), assign probabilistic labels to account for ambiguity in the data. Example: If 70% of annotators labeled an image as "happy" and 30% as "neutral," you can label it as [0.7, 0.3] instead of forcing a binary decision.

    
### 2. Epistemic uncertainty 
**Epistemic** (ep·i·ste·mic) is an adjective that means, "*relating to knowledge or to the degree of its validation.*" 

Epistemic uncertainty refers to gaps in the model's knowledge about the data distribution, which can be reduced by using more data or improved models. Aleatoric uncertainy can arise due to:

- **Systematic resolution differences**:
  - Image: A model trained on high-resolution images but tested on low-resolution inputs (e.g., wildlife drones capturing lower-resolution data than the training dataset).
  - Text: OCR systems misclassifying text scanned at lower resolution than the training examples.

- **Out-of-distribution (OOD) data**:
  - Tabular: Classifying user behavior from a new region not included in training data. Predicting hospital demand during a rare pandemic with limited historical data.
  - Image: Recognizing a new species in wildlife monitoring. Detecting a rare/unseen obstacle to automate driving.
  - Text: Queries about topics completely outside the model's domain (e.g., financial queries in a healthcare chatbot).  Interpreting slang or idiomatic expressions unseen during training.

- **Sparse or insufficient data in feature space**:
  - Tabular: High-dimensional data with many missing or sparsely sampled features (e.g., genomic datasets).
  - Image: Limited labeled examples for rare diseases in medical imaging datasets.
  - Text: Few-shot learning scenarios for domain-specific terminology.

#### Methods for addressing epistemic uncertainty

Epistemic uncertainty arises from the model's lack of knowledge about certain regions of the data space. Techniques to address this uncertainty include:

- **Ensemble models**: These involve training multiple models on the same data, each starting with different initializations or random seeds. The ensemble's predictions are aggregated, and the variance in their outputs reflects uncertainty. This approach works well because different models often capture different aspects of the data. For example, if all models agree, the prediction is confident; if they disagree, there is uncertainty. Ensembles are effective but computationally expensive, as they require training and evaluating multiple models.
- **Bayesian neural networks**: These networks incorporate probabilistic layers to model uncertainty directly in the weights of the network. Instead of assigning a single deterministic weight to each connection, Bayesian neural networks assign distributions to these weights, reflecting the uncertainty about their true values. During inference, these distributions are sampled multiple times to generate predictions, which naturally include uncertainty estimates. While Bayesian neural networks are theoretically rigorous and align well with the goal of epistemic uncertainty estimation, they are computationally expensive and challenging to scale for large datasets or deep architectures. This is because calculating or approximating posterior distributions over all parameters becomes intractable as model size grows. To address this, methods like variational inference or Monte Carlo sampling are often used, but these approximations can introduce inaccuracies, making Bayesian approaches less practical for many modern applications. Despite these challenges, Bayesian neural networks remain valuable for research contexts where precise uncertainty quantification is needed or in domains where computational resources are less of a concern.
- **Out-of-distribution detection**: Identifies inputs that fall significantly outside the training distribution, flagging areas where the model's predictions are unreliable. Many OOD methods produce continuous scores, such as Mahalanobis distance or energy-based scores, which measure how novel or dissimilar an input is from the training data. These scores can be interpreted as a form of epistemic uncertainty, providing insight into how unfamiliar an input is. However, OOD detection focuses on distinguishing ID from OOD inputs rather than offering confidence estimates for predictions on ID inputs.
- **Collect more data**: Easier said than done! Focus on gathering data from underrepresented scenarios or regions of the feature space, particularly areas where the model exhibits high uncertainty (e.g., rare medical conditions, edge cases in autonomous driving). This directly reduces epistemic uncertainty by expanding the model's knowledge base.
  - **Active learning**: Use model uncertainty estimates to prioritize uncertain or ambiguous samples for annotation, enabling more targeted data collection.

#### Why is OOD detection widely adopted?

Among epistemic uncertainty methods, OOD detection has become a widely adopted approach in real-world applications due to its ability to efficiently identify inputs that fall outside the training data distribution, where predictions are inherently unreliable. Many OOD detection techniques produce continuous scores that quantify the novelty or dissimilarity of inputs, which can be interpreted as a form of uncertainty. This makes OOD detection not only effective at rejecting anomalous inputs but also useful for prioritizing inputs based on their predicted risk.

For example, in autonomous vehicles, OOD detection can help flag unexpected scenarios (e.g., unusual objects on the road) in near real-time, enabling safer decision-making. Similarly, in NLP, OOD methods are used to identify queries or statements that deviate from a model's training corpus, such as out-of-context questions in a chatbot system. In the next couple of episodes, we'll see how to implement various OOD strategies.

### Exercises: Analyzing uncertainty in real-world applications

:::: challenge

#### Tabular Data Example: Hospital Resource Allocation

You are tasked with designing a machine learning system to predict hospital resource demands during seasonal flu outbreaks and rare pandemics. Discuss:

1. Types of uncertainty (aleatoric vs. epistemic) present in this application.
2. Methods to estimate and address each type of uncertainty.
3. Trade-offs between methods in terms of complexity and applicability.

::::

::::: solution

For hospital resource allocation, both aleatoric and epistemic uncertainties play a critical role. Aleatoric uncertainty arises from inherent variability in hospital resource demand, such as seasonal flu spikes or local reporting inconsistencies. Epistemic uncertainty stems from limited or incomplete data, particularly for rare events like pandemics.

#### General Approaches:
- **Aleatoric Uncertainty**:
  - Use models that capture the variability in the data through predictive variance or heteroscedastic loss.
  - Employ robust models or data augmentation to handle noise in the data.
- **Epistemic Uncertainty**:
  - Use ensemble methods or Bayesian neural networks to estimate uncertainty in underrepresented scenarios.
  - Apply OOD detection methods to identify anomalies or novel data points outside the training distribution.

#### Specific Examples:
- **Aleatoric Uncertainty**:
  - Predictive variance in regression models can highlight inherent randomness in hospital bed demand due to unpredictable events.  
    - Reference: [Taylor et al., 2021](https://bmcmedinformdecismak.biomedcentral.com/articles/10.1186/s12911-022-01787-9).
  - Heteroscedastic models can account for variability in reporting systems or fluctuating patient arrival rates.  
    - Reference: [Rajkomar et al., 2018](https://www.nature.com/articles/s41746-018-0029-1.pdf).

- **Epistemic Uncertainty**:
  - Tree ensembles can capture uncertainty in rare conditions, such as during pandemics.  
    - Reference: [Shahid et al., 2020](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1010602).
  - OOD detection methods identify unexpected spikes in demand or errors in occupancy reporting.  
    - Reference: [Pang et al., 2021](https://arxiv.org/abs/2110.11334).

:::::


:::: challenge

#### Image Data Example: Tumor Detection in Radiology Images

Design a system to detect tumors in radiology images. Discuss:

1. Sources of aleatoric and epistemic uncertainty in this application.
2. Potential methods to estimate and address these uncertainties.
3. Evaluate how well these methods scale for large datasets and real-world deployment.

::::

::::: solution

For tumor detection, aleatoric uncertainty arises from variability in image quality, such as noise, motion blur, or resolution artifacts. Epistemic uncertainty comes from limited labeled data for rare tumor types or novel imaging modalities.

#### General Approaches:
- **Aleatoric Uncertainty**:
  - Use heteroscedastic loss to model data-dependent noise.
  - Apply data augmentation to improve robustness to imaging artifacts.
- **Epistemic Uncertainty**:
  - Use Monte Carlo dropout or Bayesian models to estimate uncertainty in rare cases.
  - Apply OOD detection to flag previously unseen tumor types or conditions.

#### Specific Examples:
- **Aleatoric Uncertainty**:
  - CNNs with heteroscedastic loss can account for imaging artifacts like low resolution or motion blur.  
    - Reference: [Kendall & Gal, 2017](https://arxiv.org/abs/1703.04977).
  - Data augmentation strategies, such as adding synthetic noise, improve robustness.  
    - Reference: [Shorten & Khoshgoftaar, 2019](https://journalofbigdata.springeropen.com/articles/10.1186/s40537-019-0197-0).

- **Epistemic Uncertainty**:
  - Monte Carlo dropout generates uncertainty maps for tumor boundaries.  
    - Reference: [Leibig et al., 2017](https://www.nature.com/articles/s41598-017-17876-z.pdf).
  - OOD detection flags anomalous imaging data, such as unseen tumor types.  
    - Reference: [Hendrycks & Gimpel, 2017](https://arxiv.org/abs/1610.02136).

:::::

:::: challenge

#### Text Data Example: Chatbot Intent Recognition

Develop a system for chatbot intent recognition. Discuss:

1. Aleatoric and epistemic uncertainties in the application.
2. Methods to mitigate these uncertainties.
3. Trade-offs between implementing simple uncertainty measures versus advanced techniques.

::::

::::: solution

For chatbot intent recognition, aleatoric uncertainty comes from ambiguous or noisy user queries, while epistemic uncertainty arises from queries in domains not represented in the training data.

#### General Approaches:
- **Aleatoric Uncertainty**:
  - Use predictive variance in simpler models like logistic regression to detect ambiguities.
  - Heteroscedastic models, particularly in transformers, can handle token-level noise.
- **Epistemic Uncertainty**:
  - Bayesian methods or Monte Carlo dropout can quantify uncertainty for rare or unseen topics.
  - OOD detection can identify out-of-scope queries.

#### Specific Examples:
- **Aleatoric Uncertainty**:
  - Logistic regression captures variance in mixed-intent queries.  
    - Reference: [Hazra et al., 2020](https://direct.mit.edu/tacl/article/doi/10.1162/tacl_a_00483/111592/Uncertainty-Estimation-and-Reduction-of-Pre).
  - Transformers with heteroscedastic models capture token-level uncertainty in noisy inputs.  
    - Reference: [Malinin & Gales, 2018](https://arxiv.org/pdf/1802.10501v3).

- **Epistemic Uncertainty**:
  - Bayesian transformers quantify uncertainty in rare topics.  
    - Reference: [Fort et al., 2020](https://openreview.net/pdf?id=CSXa8LJMttt).
  - OOD detection identifies out-of-scope or idiomatic queries.  
    - Reference: [Lin & Xu, 2019](https://aclanthology.org/2020.coling-main.125.pdf).

:::::


#### Summary

Uncertainty estimation is a critical component of building reliable and trustworthy machine learning models, especially in high-stakes applications. By understanding the distinction between aleatoric uncertainty (inherent data noise) and epistemic uncertainty (gaps in the model's knowledge), practitioners can adopt tailored strategies to improve model robustness and interpretability.

- Aleatoric uncertainty is irreducible noise in the data itself. Addressing this requires models that can predict variability, such as heteroscedastic loss functions, or strategies like data augmentation to make models more resilient to imperfections.
- Epistemic uncertainty arises from the model's incomplete understanding of the data distribution. It can be mitigated through methods like Monte Carlo dropout, Bayesian neural networks, ensemble models, and Out-of-Distribution (OOD) detection. Among these methods, OOD detection has become a cornerstone for handling epistemic uncertainty in practical applications. Its ability to flag anomalous or out-of-distribution inputs makes it an essential tool for ensuring model predictions are reliable in real-world scenarios.
  - In many cases, collecting more data and employing active learning can directly address the root causes of epistemic uncertainty.

When choosing a method, it’s important to consider the trade-offs in computational cost, model complexity, and the type of uncertainty being addressed. Together, these techniques form a powerful toolbox, enabling models to better navigate uncertainty and maintain trustworthiness in dynamic environments. By combining these approaches strategically, practitioners can ensure that their systems are not only accurate but also robust, interpretable, and adaptable to the challenges of real-world data.
