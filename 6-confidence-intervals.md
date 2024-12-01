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

## Estimating model uncertainty

Understanding how confident a model is in its predictions is a valuable tool for building trustworthy AI systems, especially in high-stakes settings like healthcare or autonomous vehicles. Model uncertainty estimation focuses on quantifying the model's confidence and is often used to identify predictions that require further review or caution.

Model uncertainty can be divided into two categories:

### 1. Aleatoric (Random) uncertainty
**Aleatoric** is a synonym for "random":
  a·le·a·to·ry
  /ˈālēəˌtôrē/
  adjective
  adjective: aleatoric
  depending on the throw of a dice or on chance; random.

Aleatoric uncertainty is the inherent noise in the data that cannot be reduced, even with more data (observations OR missing features). Aleatoric uncertainy can arise due to:
  - Inconsistent readings from faulty sensors
  - Background noise in audio, multiple overlapping signals, recording quality
  - Random fluctuations in image resolution or lightning conditions (NOT systemic or cyclic)
  - Overlapping classes, ambiguous labels due to subjective interpretations
  - Human errors in data entry, random missing values
    
#### Methods for addressing aleatoric uncertainty

Aleatoric uncertainty arises from the data itself. Methods to estimate it include:

- **Predictive variance in regression models**: Outputs the variance of the predicted value, reflecting the noise in the data. For instance, in a regression task predicting house prices, predictive variance highlights how much randomness exists in the relationship between input features (like square footage) and price.  
- **Heteroscedastic models**: Use specialized loss functions that allow the model to predict the noise level in the data directly. These models are particularly critical in fields like *robotics*, where sensor noise varies significantly depending on environmental conditions. For example, a robot navigating in bright daylight versus dim lighting conditions may experience vastly different levels of noise in its sensor inputs, and heteroscedastic models can help account for this variability.  
- **Data augmentation and perturbation analysis**: Assess variability in predictions by adding noise to the input data and observing how much the model’s outputs change. A highly sensitive change in predictions may indicate underlying noise or instability in the data. For instance, in image classification, augmenting training data with synthetic noise can help the model better handle real-world imperfections like motion blur or occlusions.


### 2. Epistemic uncertainty 

**Epistemic** is defined as:
  ep·i·ste·mic
  /ˌepəˈstēmik,ˌepəˈstemik/
  adjectivePhilosophy
  relating to knowledge or to the degree of its validation.

Epistemic uncertainty refers to gaps in the model's knowledge about the data distribution, which can be reduced by using more data or improved models. Aleatoric uncertainy can arise due to:
### 2. Epistemic Uncertainty

Epistemic uncertainty refers to gaps in the model's knowledge about the data distribution, which can be reduced by using more data or improved models. Epistemic uncertainty can arise due to:
  - **Rare or underrepresented scenarios**:
    - Tabular: Predicting hospital demand during a rare pandemic with limited historical data.
    - Image: Detecting tumors in rare imaging modalities (e.g., PET scans).
    - Text: Answering questions about niche technical domains in a chatbot system.

  - **Systematic resolution differences**:
    - Image: A model trained on high-resolution images but tested on low-resolution inputs (e.g., wildlife drones capturing lower-resolution data than the training dataset).
    - Text: OCR systems misclassifying text scanned at lower resolution than the training examples.

  - **Novel or unseen data points**:
    - Tabular: Classifying user behavior from a new region not included in training data.
    - Image: Recognizing a new species in wildlife monitoring.
    - Text: Interpreting slang or idiomatic expressions unseen during training.

  - **Out-of-distribution (OOD) data**:
    - Tabular: Unexpected shifts in sensor readings from equipment malfunctions.
    - Image: Adversarial images with imperceptible changes designed to confuse the model.
    - Text: Queries about topics completely outside the model's domain (e.g., financial queries in a healthcare chatbot).

  - **Sparse or insufficient data in feature space**:
    - Tabular: High-dimensional data with many missing or sparsely sampled features (e.g., genomic datasets).
    - Image: Limited labeled examples for rare diseases in medical imaging datasets.
    - Text: Few-shot learning scenarios for domain-specific terminology.

#### Methods for addressing epistemic uncertainty

Epistemic uncertainty arises from the model's lack of knowledge about certain regions of the data space. Techniques to estimate it include:

- **Monte Carlo dropout**: In this method, dropout (a regularization technique that randomly disables some neurons) is applied during inference, and multiple forward passes are performed for the same input. The variability in the outputs across these passes gives an estimate of uncertainty. Intuitively, each forward pass simulates a slightly different version of the model, akin to an ensemble. If the model consistently predicts similar outputs despite dropout, it is confident; if predictions vary widely, the model is uncertain about that input.
- **Bayesian neural networks**: These networks incorporate probabilistic layers to model uncertainty directly in the weights of the network. Instead of assigning a single deterministic weight to each connection, Bayesian neural networks assign distributions to these weights, reflecting the uncertainty about their true values. During inference, these distributions are sampled multiple times to generate predictions, which naturally include uncertainty estimates. While Bayesian neural networks are theoretically rigorous and align well with the goal of epistemic uncertainty estimation, they are computationally expensive and challenging to scale for large datasets or deep architectures. This is because calculating or approximating posterior distributions over all parameters becomes intractable as model size grows. To address this, methods like variational inference or Monte Carlo sampling are often used, but these approximations can introduce inaccuracies, making Bayesian approaches less practical for many modern applications. Despite these challenges, Bayesian neural networks remain valuable for research contexts where precise uncertainty quantification is needed or in domains where computational resources are less of a concern.
- **Ensemble models**: These involve training multiple models on the same data, each starting with different initializations or random seeds. The ensemble's predictions are aggregated, and the variance in their outputs reflects uncertainty. This approach works well because different models often capture different aspects of the data. For example, if all models agree, the prediction is confident; if they disagree, there is uncertainty. Ensembles are effective but computationally expensive, as they require training and evaluating multiple models.
- **Out-of-distribution detection**: Identifies inputs that fall significantly outside the training distribution, flagging areas where the model's predictions are unreliable. Many OOD methods produce continuous scores, such as Mahalanobis distance or energy-based scores, which measure how novel or dissimilar an input is from the training data. These scores can be interpreted as a form of epistemic uncertainty, providing insight into how unfamiliar an input is. However, OOD detection focuses on distinguishing ID from OOD inputs rather than offering confidence estimates for predictions on ID inputs.
- **Collect more data**: Focus on gathering data from underrepresented scenarios or regions of the feature space, particularly areas where the model exhibits high uncertainty (e.g., rare medical conditions, edge cases in autonomous driving). This directly reduces epistemic uncertainty by expanding the model's knowledge base.
- **Active learning**: Use model uncertainty estimates to prioritize uncertain or ambiguous samples for annotation, enabling more targeted data collection.

#### Methods for addressing epistemic uncertainty (table)

| Method                  | Key strengths                               | Key limitations                                    | Suitable model sizes           | Suitable data sizes            | Compute time (approx.)          |
|-------------------------|--------------------------------------------|--------------------------------------------------|---------------------------------|--------------------------------|----------------------------------|
| Bayesian neural nets    | Rigorous probabilistic foundation          | Computationally prohibitive for large models/datasets due to repeated approximation of posterior distributions | Small to medium                 | Small to medium                 | Very high (requires posterior approximation with multiple forward passes) |
| Ensemble models         | Effective and robust; captures diverse uncertainties | Resource-intensive; requires training multiple models | Small to large (scales with ensemble size) | Small to large                  | Very high (training multiple models) |
| Monte Carlo dropout     | Easy to implement in existing neural networks | Computationally expensive due to multiple forward passes | Small to large                  | Small to large                  | High (scales with forward passes) |
| OOD detection           | Efficient, scalable, excels at rejecting anomalous inputs | Comparisons to OOD classes can be infinite, making perfect thresholds hard to define; struggles with subtle in-distribution shifts | Small to large                  | Small to large                  | Low to medium (scales efficiently) |

:::::::::::::::::::::::::::::::::::::: callout
#### Understanding size categories in table

To help guide method selection, here are rough definitions for **model size**, **data size**, and **compute requirements** used in the table:

**Model size**

- **Small**: Fewer than 10M parameters (e.g., logistic regression, LeNet).
- **Medium**: 10M–100M parameters (e.g., ResNet-50, BERT-base).
- **Large**: More than 100M parameters (e.g., GPT-3, Vision Transformers).

**Data size**

- **Small**: Fewer than 10,000 samples.
- **Medium**: 10,000–1M samples (e.g., ImageNet).
- **Large**: More than 1M samples (e.g., Common Crawl, LAION-5B).

**Compute time** (approximate)

- **Low**: Suitable for standard CPU or single GPU, training/inference in minutes to an hour.
- **Medium**: Requires a modern GPU, training/inference in hours to a day.
- **High**: Requires multiple GPUs/TPUs or distributed setups, training/inference in days to weeks.
  
::::::::::::::::::::::::::::::::::::::

#### Why is OOD detection widely adopted?

Among epistemic uncertainty methods, OOD detection has become a widely adopted approach in real-world applications due to its ability to efficiently identify inputs that fall outside the training data distribution, where predictions are inherently unreliable. Many OOD detection techniques produce continuous scores that quantify the novelty or dissimilarity of inputs, which can be interpreted as a form of uncertainty. This makes OOD detection not only effective at rejecting anomalous inputs but also useful for prioritizing inputs based on their predicted risk.

For example, in autonomous vehicles, OOD detection can help flag unexpected scenarios (e.g., unusual objects on the road) in near real-time, enabling safer decision-making. Similarly, in NLP, OOD methods are used to identify queries or statements that deviate from a model's training corpus, such as out-of-context questions in a chatbot system. In the next couple of episodes, we'll see how to implement various OOD strategies.

### Exercises: Analyzing Uncertainty in Real-World Applications

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

Develop a system for chatbot intent recognition. Address:
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

While uncertainty estimation provides a broad framework for understanding model confidence, different methods are suited for specific types of uncertainty and use cases. OOD detection stands out as the most practical approach for handling epistemic uncertainty in modern applications, thanks to its efficiency and ability to reject anomalous inputs. Together, these methods form a complementary toolkit for building trustworthy AI systems.
