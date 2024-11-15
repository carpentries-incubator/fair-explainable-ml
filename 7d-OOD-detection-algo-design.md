---
title: "OOD detection: training-time regularization"
teaching: 0
exercises: 0
---
:::::::::::::::::::::::::::::::::::::::: questions
- What are the key considerations when designing algorithms for OOD detection?
- How can OOD detection be incorporated into the loss functions of models?
- What are the challenges and best practices for training models with OOD detection capabilities?
::::::::::::::::::::::::::::::::::::::::::::::::::
::::::::::::::::::::::::::::::::::::::: objectives
- Understand the critical design considerations for creating effective OOD detection algorithms.
- Learn how to integrate OOD detection into the loss functions of machine learning models.
- Identify the challenges in training models with OOD detection and explore best practices to overcome these challenges.
::::::::::::::::::::::::::::::::::::::::::::::::::

# Training-time regularization for OOD detection
Training-time regularization methods improve OOD detection by incorporating penalties into the training process. These penalties encourage the model to handle OOD data effectively, either by:

- Penalizing high confidence on OOD samples,
- Optimizing feature representations to separate ID and OOD data,
- Or enhancing robustness to adversarial or ambiguous inputs.

The following methods apply these penalties in different ways: outlier exposure, contrastive learning, confidence penalties, and adversarial training.

#### 2a) Outlier exposure
Outlier Exposure (OE) penalizes high confidence on OOD samples by introducing auxiliary datasets during training. This method teaches the model to differentiate OOD data from ID data.

**How it works**:

- Use a curated auxiliary dataset of OOD samples that differ from the training distribution.
- Augment the training loss function to penalize high confidence on these auxiliary samples.
- Resulting models are less likely to misclassify OOD inputs as ID.

| **Advantages**                     | **Limitations**                                                        |
|------------------------------------|------------------------------------------------------------------------|
| Simple to implement when auxiliary datasets are available. | Requires access to high-quality, diverse OOD datasets during training. |
| Improves OOD detection performance without significant computational cost. | Performance may degrade for OOD samples dissimilar to the auxiliary dataset. |

#### 2b) Contrastive learning
Contrastive learning optimizes feature representations by applying penalties that control the similarity of embeddings. Positive pairs (similar samples) are brought closer together, while negative pairs (dissimilar samples) are pushed apart. This results in a feature space where OOD data is less likely to overlap with ID data.

**How it works**:

- Define a contrastive loss that minimizes the distance between embeddings of similar samples (e.g., belonging to the same class).
- Simultaneously maximize the distance between embeddings of dissimilar samples (e.g., ID vs. synthetic or auxiliary OOD samples).
- Often uses data augmentation or self-supervised techniques to generate "positive" and "negative" pairs.

| **Advantages**                     | **Limitations**                                                        |
|------------------------------------|------------------------------------------------------------------------|
| Does not require labeled auxiliary OOD data, as augmentations or unsupervised data can be used. | Computationally expensive, especially with large datasets.              |
| Improves the quality of learned representations, benefiting other tasks. | Requires careful tuning of the contrastive loss and data augmentation strategy. |

#### 2c) Other regularization-based techniques
Other methods incorporate penalties directly into the training process to improve robustness to OOD data:

- **Confidence penalties**: Penalize overconfidence in predictions, especially on ambiguous samples.
- **Adversarial training**: Generate adversarial examples (slightly perturbed ID samples) to penalize high confidence on these perturbed examples, improving robustness.

| **Advantages**                     | **Limitations**                                                        |
|------------------------------------|------------------------------------------------------------------------|
| Enhances OOD detection performance by integrating it into the training process. | Requires careful design of the training procedure and loss function.   |
| Leads to better generalization for both ID and OOD scenarios.           | Computationally intensive and may need access to additional datasets.  |

#### Summary of Training-Time Regularization Methods

| **Method**                        | **Penalty Applied**                                                             | **Advantages**                                   | **Limitations**                                                                 |
|-----------------------------------|--------------------------------------------------------------------------------|-------------------------------------------------|---------------------------------------------------------------------------------|
| Outlier Exposure                  |  High confidence on auxiliary OOD data.                              | Simple to implement, improves performance.      | Requires high-quality auxiliary datasets, may not generalize to unseen OOD data.|
| Contrastive Learning              |  Embedding similarity for dissimilar samples (and vice versa) | Improves feature space quality, versatile.      | Computationally expensive, requires careful tuning.                             |
| Confidence Penalties              |  Overconfidence on ambiguous inputs.                                 | Improves robustness, generalizes well.          | Requires careful design, computationally intensive.                             |
| Adversarial Training              |  High confidence on adversarial examples.                            | Enhances robustness to perturbed inputs.        | Computationally intensive, challenging to implement.                            |

:::::::::::::::::::::::::::::::::::::::: keypoints

- Out-of-distribution (OOD) data significantly differs from training data and can lead to unreliable model predictions.
- Threshold-based methods use model outputs or distances in feature space to detect OOD instances by defining a score threshold.
- Training-time regularization enhances OOD detection by incorporating techniques like Outlier Exposure and Contrastive Learning during model training.
- Each method has trade-offs: threshold-based methods are simpler, while training-time regularization often improves robustness at higher computational cost.

::::::::::::::::::::::::::::::::::::::::::::::::::
