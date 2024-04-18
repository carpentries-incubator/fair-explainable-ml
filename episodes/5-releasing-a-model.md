---
title: "Documenting and Releasing a Model"
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
- Understand distribution shift and its implications.

::::::::::::::::::::::::::::::::::::::::::::::::

::::::::::::::::::::::::::::::::::::: keypoints 

- Model cards are the standard technique for communicating information about how machine learning systems were trained and how they should and should not be used.
- Models can be shared and reused via the Hugging Face platform.
- Distribution shift is common. It can be caused by temporal shifts (i.e., using old training data) or by applying a model to new populations.
- Distribution shift can be addressed by TODO

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
