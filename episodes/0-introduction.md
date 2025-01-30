---
title: "Overview"
teaching: 30
exercises: 1
---
 
:::::::::::::::::::::::::::::::::::::: questions 

- What do we mean by "Trustworthy AI"? 
- How is this workshop structured, and what content does it cover?

::::::::::::::::::::::::::::::::::::::::::::::::
 
::::::::::::::::::::::::::::::::::::: objectives

- Define trustworthy AI and its various components.
- Be prepared to dive into the rest of the workshop.

::::::::::::::::::::::::::::::::::::::::::::::::

## What is trustworthy AI? 

:::::::::::::::::::::::::::::::::::::: discussion

Take a moment to brainstorm what keywords/concepts come to mind when we mention "Trustworthy AI". 
Share your thoughts with the class.

::::::::::::::::::::::::::::::::::::::::::::::::::


Artificial intelligence (AI) and machine learning (ML) are being used widely to improve upon human capabilities (either in speed/convenience/cost or accuracy) in a variety of domains: medicine, social media, news, marketing, policing, and more. 
It is important that the decisions made by AI/ML models uphold values that we, as a society, care about. 

Trustworthy AI is a large and growing sub-field of AI that aims to ensure that AI models are trained and deployed in ways that are ethical and responsible.

## The AI Bill of Rights
In October 2022, the Biden administration released a [Blueprint for an AI Bill of Rights](https://www.whitehouse.gov/ostp/ai-bill-of-rights/), a non-binding document that outlines how automated systems and AI should behave in order to protect Americans' rights.

The blueprint is centered around five principles:

*  Safe and Effective Systems -- AI systems should work as expected, and should not cause harm
*  Algorithmic Discrimination Protections -- AI systems should not discriminate or produce inequitable outcomes
*  Data Privacy -- data collection should be limited to what is necessary for the system functionality, and you should have control over how and if your data is used
*  Notice and Explanation -- it should be transparent when an AI system is being used, and there should be an explanation of how particular decisions are reached 
*  Human Alternatives, Consideration, and Fallback -- you should be able to opt out of engaging with AI systems, and a human should be available to remedy any issues



## This workshop

This workshop centers around four principles that are important to trustworthy AI: *scientific validity*, *fairness*, *transparency*, *safety & uncertainty*, and *accountability*. We summarize each principle here.

### Scientific validity
In order to be trustworthy, a model and its predictions need to be founded on good science. A model is not going to perform well if is not trained on the correct data, if it fits the underlying data poorly, or if it cannot recognize its own limitations. Scientific validity is closely linked to the AI Bill of Rights principle of "safe and effective systems". 

In this workshop, we cover the following topics relating to scientific validity:

* Defining the problem (Preparing to Train a Model episode)
* Training and evaluating a model, especially selecting an accuracy metric, avoiding over/underfitting, and preventing data leakage (Model Evaluation and Fairness episode)

### Fairness
As stated in the AI Bill of Rights, AI systems should not be discriminatory or produce inequitable outcomes. In the Model Evaluation and Fairness episode we discuss various definitions of fairness in the context of AI, and overview how model developers try to make their models more fair. 

### Transparency
Transparency -- i.e., insight into *how* a model makes its decisions -- is important for trustworthy AI, as we want models that make the right decisions *for the right reasons*. Transparency can be achieved via *explanations* or by using inherently *interpretable* models. We discuss transparency in the follow episodes:

* Interpretability vs Explainability
* Explainability Methods Overview
* Explainability Methods: Deep Dive, Linear Probe, and GradCAM episodes
  
### Safety & uncertainty awareness
AI models should be able to quantify their uncertainty and recognize when they encounter novel or unreliable inputs. If a model makes confident predictions on data that it has never seen before (e.g., out-of-distribution data), it can lead to critical failures in high-stakes applications like healthcare or autonomous systems.

In this workshop, we cover the following topics relating to safety and uncertainty awareness:

* Estimating model uncertainty—understanding when models should be uncertain and how to measure it (Estimating Model Uncertainty episode)
* Out-of-distribution detection—distinguishing between known and unknown data distributions to improve reliability (OOD Detection episodes)
* Comparing uncertainty estimation and OOD detection approaches, including:
  * Output-based methods (softmax confidence, energy-based models)
  * Distance-based methods (Mahalanobis distance, k-NN)
  * Contrastive learning for improving generalization

By incorporating uncertainty estimation and OOD detection, we emphasize the importance of AI models *knowing what they don’t know* and making safer decisions.

### Accountability
Accountability is important for trustworthy AI because, inevitably, models will make mistakes or cause harm. Accountability is multi-faceted and largely non-technical, which is not to say unimportant, but just that it falls partially out of scope of this technical workshop.

We discuss two facets of accountability, model documentation and model sharing, in the Documenting and Releasing a Model episode. 

For those who are interested, we recommend these papers to learn more about different aspects of AI accountability:

1. [Accountability of AI Under the Law: The Role of Explanation](https://arxiv.org/pdf/1711.01134) by Finale Doshi-Velez and colleagues. This paper discusses how explanations can be used in a legal context to determine accountability for harms caused by AI. 
2. [Closing the AI accountability gap: defining an end-to-end framework for internal algorithmic auditing](https://dl.acm.org/doi/abs/10.1145/3351095.3372873) by Deborah Raji and colleagues proposes a framework for auditing algorithms. A key contribution of this paper is defining an auditing procedure over the whole model development and implementation pipeline, rather than narrowly focusing on the modeling stages. 
3. [AI auditing: The Broken Bus on the Road to AI Accountability](https://ieeexplore.ieee.org/abstract/document/10516659) by Abeba Birhane and colleagues challenges previous work on AI accountability, arguing that most existing AI auditing systems are not effective. They propose necessary traits for effective AI audits, based on a review of existing practices. 

### Topics we do not cover
Trustworthy AI is a large, and growing, area of study. As of September 24, 2024, **there are about 18,000 articles on Google Scholar that mention Trustworthy AI and were published in the first 9 months of 2024**. 

There are different Trustworthy AI methods for different types of models -- e.g., decisions trees or linear models that are commonly used with tabular data, neural networks that are used with image data, or large multi-modal foundation models. In this workshop, we focus primarily on neural networks for the specific techniques we show in the technical implementations. That being said, much of the conceptual content is relevant to any model type. 

Many of the topics we do not cover are sub-topics of the broad categories -- e.g., fairness, explainability, or OOD detection -- of the workshop and are important for specific use cases, but less relevant for a general audience. But, there are a few major areas of research that we don't have time to touch on. We summarize a few of them here:

#### Data Privacy
In the US's Blueprint for an AI Bill of Rights, one principle is data privacy, meaning that people should be aware how their data is being used, companies should not collect more data than they need, and people should be able to consent and/or opt out of data collection and usage. 

A lack of data privacy poses several risks: first, whenever data is collected, it can be subject to data breaches. This risk is unavoidable, but collecting only the data that is truly necessary mitigates this risk, as does implementing safeguards to how data is stored and and accessed. Second, when data is used to train ML models, that data can sometimes be identifying by attackers. For instance, large language models like ChatGPT are known to release private data that was part of the training corpus when prompted in clever ways (see this [blog post](https://not-just-memorization.github.io/extracting-training-data-from-chatgpt.html) for more information).  
Membership inference attacks, where an attacker determines whether a particular individual's data was in the training corpus, are another vulnerability. These attacks may reveal things about a person directly (e.g., if the training dataset consisted of only people with a particular medical condition), or can be used to setup downstream attacks to gain more information. 

There are several areas of active research relating to data privacy.

* [Differential privacy](https://link.springer.com/chapter/10.1007/978-3-540-79228-4_1) is a statistical technique that protects the privacy of individual data points. Models can be trained using differential privacy to provably prevent future attacks, but this currently comes at a high cost to accuracy. 
* [Federated learning](https://ieeexplore.ieee.org/abstract/document/9599369) trains models using decentralized data from a variety of sources. Since the data is not shared centrally, there is less risk of data breaches or unauthorized data usage.

#### Generative AI risks
We touch on fairness issues with generative AI in the Model Evaluation and Fairness episode. But generative AI poses other risks, too, many of which are just starting to be researched and understood given how new widely-available generative AI is. We discuss one such risk, disinformation, briefly here: 

* Disinformation: A major risk of generative AI is the creation of misleading or fake and malicious content, often known as [deep fakes](https://timreview.ca/article/1282). Deep fakes pose risks to individuals (e.g., creating content that harms an individual's reputation) and society (e.g., fake news articles or pictures that look real). 

:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::: instructor

Inline instructor notes can help inform instructors of timing challenges
associated with the lessons. They appear in the "Instructor View"

::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::


