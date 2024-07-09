---
title: "Documenting and releasing a model"
teaching: 0
exercises: 0
---

:::::::::::::::::::::::::::::::::::::: questions 

- Why is model sharing important in the context of reproducibility and responsible use?
- What are the challenges, risks, and ethical considerations related to sharing models?
- How can model-sharing best practices be applied using tools like model cards and the Hugging Face platform?
- What is distribution shift and what are its implications in machine learning models?

::::::::::::::::::::::::::::::::::::::::::::::::

::::::::::::::::::::::::::::::::::::: objectives

- Understand the importance of model sharing and best practices to ensure reproducibility and responsible use of models.
- Understand the challenges, risks, and ethical concerns associated with model sharing.
- Apply model-sharing best practices through using model cards and the Hugging Face platform.

::::::::::::::::::::::::::::::::::::::::::::::::

::::::::::::::::::::::::::::::::::::: keypoints 

- Model cards are the standard technique for communicating information about how machine learning systems were trained and how they should and should not be used.
- Models can be shared and reused via the Hugging Face platform.

::::::::::::::::::::::::::::::::::::::::::::::::

:::::::::::::::::::::::::::::::::::::: challenge

### Why should we share trained models?
Discuss in small groups and report out: *Why do you believe it is or isn’t important to share ML models? How has model-sharing contributed to your experiences or projects?*

::::::::::::::::::::::::::::::::::::::::::::::::::

:::::::::::::: solution

### Solution

* **Accelerating research**: Sharing models allows researchers and practitioners to build upon existing work, accelerating the pace of innovation in the field.
* **Knowledge exchange**: Model sharing promotes knowledge exchange and collaboration within the machine learning community, fostering a culture of open science.
* **Reproducibility**: Sharing models, along with associated code and data, enhances reproducibility, enabling others to validate and verify the results reported in research papers.
* **Benchmarking**: Shared models serve as benchmarks for comparing new models and algorithms, facilitating the evaluation and improvement of state-of-the-art techniques.
* **Education / Accessibility to state-of-the-art architectures**: Shared models provide valuable resources for educational purposes, allowing students and learners to explore and experiment with advanced machine learning techniques.
* **Repurpose (transfer learning and finetuning)**: Some models (i.e., foundation models) can be repurposed for a wide variety of tasks. This is especially useful when working with limited data.
Data scarcity
* **Resource efficiency**: Instead of training a model from the ground up, practitioners can use existing models as a starting point, saving time, computational resources, and energy.

:::::::::::::::::::::::::

:::::::::::::::::::::::::::::::::::::: challenge

### What pieces must be well-documented to ensure reproducible and responsible model sharing?
Discuss in small groups and report out: *Why do you believe it is or isn’t important to share ML models? How has model-sharing contributed to your experiences or projects?*

::::::::::::::::::::::::::::::::::::::::::::::::::

:::::::::::::: solution

### Solution

* Environment setup
* Training data
  * How the data was collected
  * Who owns the data: data license and usage terms
  * Basic descriptive statistics: number of samples, features, classes, etc.
  * Note any class imbalance or general bias issues
  * Description of data distribution to help prevent out-of-distribution failures.
* Preprocessing steps. 
  * Data splitting
  * Standardization method
  * Feature selection
  * Outlier detection and other filters
* Model architecture, hyperparameters and, training procedure (e.g., dropout or early stopping)
* Model weights
* Evaluation metrics. Results and performance. The more tasks/datasets you can evaluate on, the better.
* Ethical considerations:  Include investigations of bias/fairness when applicable (i.e., if your model involves human data or affects decision-making involving humans)
* Contact info
* Acknowledgments
* Examples and demos (highly recommended)
:::::::::::::::::::::::::


:::::::::::::::::::::::::::::::::::::: challenge

### Challenges and risks of model sharing
Discuss in small groups and report out: *What are some potential challenges, risks, or ethical concerns associated with model sharing and reproducing ML workflows?*

::::::::::::::::::::::::::::::::::::::::::::::::::

:::::::::::::: solution

### Solution
* **Privacy concerns**: Sharing models that were trained on sensitive or private data raises privacy concerns. The potential disclosure of personal information through the model poses a risk to individuals and can lead to unintended consequences.
* **Informed consent**: If models involve user data, ensuring informed consent is crucial. Sharing models trained on user-generated content without clear consent may violate privacy norms and regulations.
* **Data bias and fairness**: Models trained on biased datasets may perpetuate or exacerbate existing biases. Reproducing workflows without addressing bias in the data may result in unfair outcomes, particularly in applications like hiring or criminal justice.
* **Intellectual property**: Models may be developed within organizations with proprietary data and methodologies. Sharing such models without proper consent or authorization may lead to intellectual property disputes and legal consequences.
* **Model robustness and generalization**: Reproduced models may not generalize well to new datasets or real-world scenarios. Failure to account for the limitations of the original model can result in reduced performance and reliability in diverse settings.
* **Lack of reproducibility**: Incomplete documentation, missing details, or changes in dependencies over time can hinder the reproducibility of ML workflows. This lack of reproducibility can impede scientific progress and validation of research findings.
* **Unintended use and misuse**: Shared models may be used in unintended ways, leading to ethical concerns. Developers should consider the potential consequences of misuse, particularly in applications with societal impact, such as healthcare or law enforcement.
* **Responsible AI considerations**: Ethical considerations, such as fairness, accountability, and transparency, should be addressed during model sharing. Failing to consider these aspects can result in models that inadvertently discriminate or lack interpretability. Models used for decision-making, especially in critical areas like healthcare or finance, should be ethically deployed. Transparent documentation and disclosure of how decisions are made are essential for responsible AI adoption.
:::::::::::::::::::::::::

