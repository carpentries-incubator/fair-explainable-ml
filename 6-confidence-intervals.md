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

#### 1. Aleatoric (Random) uncertainty
**Aleatoric** is a synonym for "random":
  a·le·a·to·ry
  /ˈālēəˌtôrē/
  adjective
  adjective: aleatoric
  depending on the throw of a dice or on chance; random.

Aleatoric uncertainty is the inherent noise in the data that cannot be reduced, even with more data. Aleatoric uncertainy can arise due to:
  - Inconsistent readings from faulty sensors
  - Background noise in audio, multiple overlapping signals, recording quality
  - Resolution of image, lighting conditions
  - Overlapping classes, ambiguous labels due to subjective interpretations
  - Human errors in data entry, missing values
    
##### Methods for addressing aleatoric uncertainty

Aleatoric uncertainty arises from the data itself. Methods to estimate it include:

- **Predictive variance in regression models**: Outputs the variance of the predicted value, reflecting the noise in the data. For instance, in a regression task predicting house prices, predictive variance highlights how much randomness exists in the relationship between input features (like square footage) and price.  
- **Heteroscedastic models**: Use specialized loss functions that allow the model to predict the noise level in the data directly. These models are particularly critical in fields like *robotics*, where sensor noise varies significantly depending on environmental conditions. For example, a robot navigating in bright daylight versus dim lighting conditions may experience vastly different levels of noise in its sensor inputs, and heteroscedastic models can help account for this variability.  
- **Data augmentation and perturbation analysis**: Assess variability in predictions by adding noise to the input data and observing how much the model’s outputs change. A highly sensitive change in predictions may indicate underlying noise or instability in the data. For instance, in image classification, augmenting training data with synthetic noise can help the model better handle real-world imperfections like motion blur or occlusions.


#### 2. Epistemic uncertainty 

**Epistemic** is defined as:
  ep·i·ste·mic
  /ˌepəˈstēmik,ˌepəˈstemik/
  adjectivePhilosophy
  relating to knowledge or to the degree of its validation.

Epistemic uncertainty refers to gaps in the model's knowledge about the data distribution, which can be reduced by using more data or improved models.

##### Methods for addressing epistemic uncertainty

Epistemic uncertainty arises from the model's lack of knowledge about certain regions of the data space. Techniques to estimate it include:

- **Monte Carlo dropout**: In this method, dropout (a regularization technique that randomly disables some neurons) is applied during inference, and multiple forward passes are performed for the same input. The variability in the outputs across these passes gives an estimate of uncertainty. Intuitively, each forward pass simulates a slightly different version of the model, akin to an ensemble. If the model consistently predicts similar outputs despite dropout, it is confident; if predictions vary widely, the model is uncertain about that input.
- **Bayesian neural networks**: These networks incorporate probabilistic layers to model uncertainty directly in the weights of the network. Instead of assigning a single deterministic weight to each connection, Bayesian neural networks assign distributions to these weights, reflecting the uncertainty about their true values. During inference, these distributions are sampled multiple times to generate predictions, which naturally include uncertainty estimates. While Bayesian neural networks are theoretically rigorous and align well with the goal of epistemic uncertainty estimation, they are computationally expensive and challenging to scale for large datasets or deep architectures. This is because calculating or approximating posterior distributions over all parameters becomes intractable as model size grows. To address this, methods like variational inference or Monte Carlo sampling are often used, but these approximations can introduce inaccuracies, making Bayesian approaches less practical for many modern applications. Despite these challenges, Bayesian neural networks remain valuable for research contexts where precise uncertainty quantification is needed or in domains where computational resources are less of a concern.
- **Ensemble models**: These involve training multiple models on the same data, each starting with different initializations or random seeds. The ensemble's predictions are aggregated, and the variance in their outputs reflects uncertainty. This approach works well because different models often capture different aspects of the data. For example, if all models agree, the prediction is confident; if they disagree, there is uncertainty. Ensembles are effective but computationally expensive, as they require training and evaluating multiple models.
- **Out-of-distribution detection**: Identifies inputs that fall significantly outside the training distribution, flagging areas where the model's predictions are unreliable. Many OOD methods produce continuous scores, such as Mahalanobis distance or energy-based scores, which measure how novel or dissimilar an input is from the training data. These scores can be interpreted as a form of epistemic uncertainty, providing insight into how unfamiliar an input is. However, OOD detection focuses on distinguishing ID from OOD inputs rather than offering confidence estimates for predictions on ID inputs.

##### Methods for addressing epistemic uncertainty (table)

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

:::::::::::::::::::::::::::::: callout
### Tabular Data Example

#### Application: Hospital resource allocation

**Aleatoric Uncertainty:**

- **Linear models:** Predictive variance captures inherent randomness in hospital occupancy patterns due to seasonal variability or unpredictable local events. See [Taylor et al., 2021](https://bmcmedinformdecismak.biomedcentral.com/articles/10.1186/s12911-022-01787-9)  for regression-based modeling of hospital bed demand during flu seasons.
- **Deep learning:** Heteroscedastic models account for input-dependent noise, such as variability in hospital reporting systems or random fluctuations in patient arrival rates during holidays or flu season. See [Rajkomar et al., 2018](https://www.nature.com/articles/s41746-018-0029-1.pdf)  for applications of deep learning in hospital resource prediction.

**Epistemic Uncertainty:**

- **Tree ensembles:** Capture uncertainty for underrepresented or novel conditions, such as predicting hospital demand during rare pandemics. For an example, see [Shahid et al., 2020](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1010602), which uses ensemble models for pandemic demand forecasting.
- **OOD detection:** Identifies anomalies in resource usage data, such as unexpected spikes in equipment demands or misreported occupancy. See [Pang et al., 2021](https://arxiv.org/abs/2110.11334) for a comprehensive survey on OOD detection in real-world tabular data.

::::::::::::::::::::::::::::::

:::::::::::::::::::::::::::::: callout
### Image Data Example

#### Application: Tumor detection in radiology images

**Aleatoric Uncertainty:**

- **CNNs:** Heteroscedastic loss accounts for noise from imaging artifacts like low resolution or motion blur. For an example of modeling aleatoric uncertainty in medical imaging, see [Kendall & Gal, 2017](https://arxiv.org/abs/1703.04977).
- **Data augmentation:** Synthetic noise during training improves robustness to real-world imperfections in medical images. See [Shorten & Khoshgoftaar, 2019](https://journalofbigdata.springeropen.com/articles/10.1186/s40537-019-0197-0) for a survey on data augmentation strategies in deep learning.

**Epistemic Uncertainty:**

- **Monte Carlo dropout:** Samples multiple outputs to generate uncertainty maps for tumor boundaries in rare conditions. See [Leibig et al., 2017](https://www.nature.com/articles/s41598-017-17876-z.pdf) for an application of MC dropout to retinal disease detection.
- **OOD detection:** Flags anomalous radiology images, such as previously unseen tumor types or imaging modalities. See [Hendrycks & Gimpel, 2017](https://arxiv.org/abs/1610.02136) for the foundational OOD detection method applied to medical imaging.
::::::::::::::::::::::::::::::

:::::::::::::::::::::::::::::: callout
### Text Data Example

#### Application: Chatbot intent recognition

**Aleatoric Uncertainty:**

- **Logistic regression:** Predictive variance highlights ambiguities in user queries with mixed intents. See [Hazra et al., 2020](https://direct.mit.edu/tacl/article/doi/10.1162/tacl_a_00483/111592/Uncertainty-Estimation-and-Reduction-of-Pre) for a discussion on uncertainty estimation in NLP tasks.
- **Transformers:** Heteroscedastic models capture token-level uncertainty in noisy or ambiguous language inputs. See [Malinin & Gales, 2018](https://arxiv.org/pdf/1802.10501v3) for heteroscedastic neural network modeling in NLP.

**Epistemic Uncertainty:**

- **Bayesian transformers:** Quantify uncertainty in rare or unseen query topics (e.g., domain-specific technical questions). See [Fort et al., 2020](https://openreview.net/pdf?id=CSXa8LJMttt) for a discussion on uncertainty in transformers.
- **OOD detection:** Detects out-of-scope queries or previously unseen intents, such as rare idiomatic expressions or slang. See [Lin & Xu, 2019](https://aclanthology.org/2020.coling-main.125.pdf) for OOD detection in NLP systems.
::::::::::::::::::::::::::::::

#### Why is OOD detection widely adopted?

Among epistemic uncertainty methods, OOD detection has become a widely adopted approach in real-world applications due to its ability to efficiently identify inputs that fall outside the training data distribution, where predictions are inherently unreliable. Many OOD detection techniques produce continuous scores that quantify the novelty or dissimilarity of inputs, which can be interpreted as a form of uncertainty. This makes OOD detection not only effective at rejecting anomalous inputs but also useful for prioritizing inputs based on their predicted risk.

For example, in autonomous vehicles, OOD detection can help flag unexpected scenarios (e.g., unusual objects on the road) in near real-time, enabling safer decision-making. Similarly, in NLP, OOD methods are used to identify queries or statements that deviate from a model's training corpus, such as out-of-context questions in a chatbot system. In the next couple of episodes, we'll see how to implement various OOD strategies.

#### Summary

While uncertainty estimation provides a broad framework for understanding model confidence, different methods are suited for specific types of uncertainty and use cases. OOD detection stands out as the most practical approach for handling epistemic uncertainty in modern applications, thanks to its efficiency and ability to reject anomalous inputs. Together, these methods form a complementary toolkit for building trustworthy AI systems.
