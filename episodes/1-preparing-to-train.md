---
title: "Preparing to train a model"
teaching: 0
exercises: 0
---

:::::::::::::::::::::::::::::::::::::: questions 

- For what prediction tasks is machine learning an appropriate tool?
- How can inappropriate target variable choice lead to suboptimal outcomes in a machine learning pipeline?
- What forms of "bias" can occur in machine learning, and where do these biases come from?

::::::::::::::::::::::::::::::::::::::::::::::::

::::::::::::::::::::::::::::::::::::: objectives

- Judge what tasks are appropriate for machine learning
- Understand why the choice of prediction task / target variable is important.
- Describe how bias can appear in training data and algorithms.

::::::::::::::::::::::::::::::::::::::::::::::::

## Choosing appropriate tasks

Machine learning is a rapidly advancing, powerful technology that is helping to drive innovation. Before embarking on a machine learning project, we need to consider the task carefully. Many machine learning efforts are not solving problems that need to be solved. Or, the problem may be valid, but the machine learning approach makes incorrect assumptions and fails to solve the problem effectively. Worse, many applications of machine learning are not for the public good. 

We will start by considering the [NIH Guiding Principles for Ethical Research](https://www.nih.gov/health-information/nih-clinical-research-trials-you/guiding-principles-ethical-research), which provide a useful set of considerations for any project.

:::::::::::::::::::::::::::::::::::::: challenge

Take a look at the [NIH Guiding Principles for Ethical Research](https://www.nih.gov/health-information/nih-clinical-research-trials-you/guiding-principles-ethical-research).

What are the main principles? 

::::::::::::::::::::::::::::::::::::::::::::::::::

:::::::::::::: solution

A summary of the principles is listed below:

* Social and clinical value: Does the social or clinical value of developing and implementing the model outweigh the risk and burden of the people involved?
* Scientific validity: Once created, will the model provide valid, meaningful outputs?
* Fair subject selection: Are the people who contribute and benefit from the model selected fairly, and not through vulnerability, privilege, or other unrelated factors?
* Favorable risk-benefit ratio: Do the potential benefits of of developing and implementing the model outweigh the risks?
* Independent review: Has the project been reviewed by someone independent of the project, and has an Institutional Review Board (IRB) been approached where appropriate?
* Informed consent: Are participants whose data contributes to development and implementation of the model, as well as downstream recipients of the model, kept informed?
* Respect for potential and enrolled subjects: Is the privacy of participants respected and are steps taken to continuously monitor the effect of the model on downstream participants?

:::::::::::::::::::::::::

AI tasks are often most controversial when they involve human subjects, and especially visual representations of people. We'll discuss two case studies that use people's faces as a prediction tool, and discuss whether these uses of AI are appropriate.

### Case study 1: Physiognomy
In 2019, Nature Medicine [published a paper](https://www.nature.com/articles/s41591-018-0279-0.epdf) that describes a model that can identify genetic disorders from a photograph of a patient’s face. The abstract of the paper is copied below:

>  Syndromic genetic conditions, in aggregate, affect 8% of the population. Many syndromes have recognizable facial features that are highly informative to clinical geneticists. Recent studies show that facial analysis technologies measured up to the capabilities of expert clinicians in syndrome identification. However, these technologies identified only a few disease phenotypes, limiting their role in clinical settings, where hundreds of diagnoses must be considered. Here we present a facial image analysis framework, DeepGestalt, using computer vision and deep-learning algorithms, that quantifies similarities to hundreds of syndromes.
>
>    DeepGestalt outperformed clinicians in three initial experiments, two with the goal of distinguishing subjects with a target syndrome from other syndromes, and one of separating different genetic sub-types in Noonan syndrome. On the final experiment reflecting a real clinical setting problem, DeepGestalt achieved 91% top-10 accuracy in identifying the correct syndrome on 502 different images. The model was trained on a dataset of over 17,000 images representing more than 200 syndromes, curated through a community-driven phenotyping platform. DeepGestalt potentially adds considerable value to phenotypic evaluations in clinical genetics, genetic testing, research and precision medicine.

* What is the proposed value of the algorithm?
* What are the potential risks?
* Are you supportive of this kind of research?
* What safeguards, if any, would you want to be used when developing and using this algorithm?

Media reports about this paper were largely positive, e.g., [reporting that clinicians are excited about the new technology](https://www.genengnews.com/insights/a-i-gets-in-the-face-of-rare-genetic-diseases/). 

### Case study 2: 

There is a long history of physiognomy, the “science” of trying to read someone’s character from their face. With the advent of machine learning, this discredited area of research has made a comeback. There have been numerous studies attempting to guess characteristics such as trustworthness, criminality, and political and sexual orientation.

In 2018, for example, researchers suggested that neural networks could be used to detect sexual orientation from facial images. The abstract is copied below:

>   We show that faces contain much more information about sexual orientation than can be perceived and interpreted by the human brain. We used deep neural networks to extract features from 35,326 facial images. These features were entered into a logistic regression aimed at classifying sexual orientation. Given a single facial image, a classifier could correctly distinguish between gay and heterosexual men in 81% of cases, and in 74% of cases for women. Human judges achieved much lower accuracy: 61% for men and 54% for women. The accuracy of the algorithm increased to 91% and 83%, respectively, given five facial images per person.
>
>   Facial features employed by the classifier included both fixed (e.g., nose shape) and transient facial features (e.g., grooming style). Consistent with the prenatal hormone theory of sexual orientation, gay men and women tended to have gender-atypical facial morphology, expression, and grooming styles. Prediction models aimed at gender alone allowed for detecting gay males with 57% accuracy and gay females with 58% accuracy. Those findings advance our understanding of the origins of sexual orientation and the limits of human perception. Additionally, given that companies and governments are increasingly using computer vision algorithms to detect people’s intimate traits, our findings expose a threat to the privacy and safety of gay men and women.

* What is the proposed value of the algorithm?
* What are the potential risks?
* Are you supportive of this kind of research?
* What distinguishes this use of AI from the use of AI described in Case Study 1?

Media reports of this algorithm were largely negative, with a [Scientific American article](https://www.scientificamerican.com/blog/observations/can-we-read-a-persons-character-from-facial-images/) highlighting the connections to physiognomy and raising concern over government use of these algorithms: 

>   This is precisely the kind of “scientific” claim that can motivate repressive governments to apply AI algorithms to images of their citizens. And what is it to stop them from “reading” intelligence, political orientation and criminal inclinations from these images?


## Choosing the outcome variable

Sometimes, choosing the outcome variable is easy: for instance, when building a model to predict how warm it will be out tomorrow, the temperature can be the outcome variable because it's measurable (i.e., you know what temperature it was yesterday and today) and your predictions won't cause a feedback loop (e.g., given a set of past weather data, the weather next Monday won't change based on what your model predicts tomorrow's temperature to be).

By contrast, sometimes it's not possible to measure the target prediction subject directly, and sometimes predictions can cause feedback loops.

### Case Study: Proxy variables

Consider the scenario described in the challenge below.

:::::::::::::::::::::::::::::::::::::: challenge

Suppose that you work for a hospital and are asked to build a model to predict which patients are high-risk and need extra care to prevent negative health outcomes. 

Discuss the following with a partner or small group:
1. What is the goal target variable? 
2. What are challenges in measuring the target variable in the training data (i.e., former patients)?
3. Are there other variables that are easier to measure, but can approximate the target variable, that could serve as proxies?
3. How do social inequities interplay with the value of the target variable versus the value of the proxies?

::::::::::::::::::::::::::::::::::::::::::::::::::

The "challenge" scenario is not hypothetical:
A well-known [study by Obermeyer et al.](https://escholarship.org/content/qt6h92v832/qt6h92v832.pdf) analyzed an algorithm that hospitals used to assign patients risk scores for various conditions. The algorithm had access to various patient data, such as demographics (e.g., age and sex), the number of chronic conditions, insurance type, diagnoses, and medical costs. The algorithm did not have access to the patient's race.
The patient risk score determined the level of care the patient should receive, with higher-risk patients receiving additional care. 

Ideally, the target variable would be health needs, but this can be challenging to measure: how do you compare the severity of two different conditions? Do you count chronic and acute conditions equally? In the system described by Obermeyer et al., the hospital decided to use **health-care costs** as a proxy for health needs, perhaps reasoning that this data is at least standardized across patients and doctors.

However, Obermeyer et al. reveal that the algorithm is biased against Black patients. 
That is, if there are two individuals -- one white and one Black -- with equal health,
the algorithm tends to assign a higher risk score to the white patient, thus giving them
access to higher care quality. 
The authors blame the choice of proxy variable for the racial disparities. 

The authors go on to describe how, due to how health-care access is structured in the US, richer patients have more healthcare expenses, even if they are equally (un)healthy to a lower-income patient. The richer patients are also more likely to be white. 

Consider the following:

* How could the algorithm developers have caught this problem earlier? 
* Is this a technical mistake or a process-based mistake? Why?

### Case study: Feedback loop
Consider social media, like Instagram or TikTok's "for you page" or Facebook or Twitter's newsfeed. 
The algorithms that determine what to show are complex (and proprietary!) but a large part of the algorithms' objective is engagement: the number of clicks, views, or re-posts. 
For instance, this focus on engagement can create an "echo chamber" where individual users solely see content that aligns with their political ideology, thereby maximizing the positive engagement with each post. But the impact of social media feedback loops spreads beyond politics: [researchers have explored](https://arxiv.org/pdf/2305.11316) how similar feedback loops exist for mental health conditions such as eating disorders. If someone finds themselves in this area of social media, it's likely because they have, or have risk factors for, an eating disorder, and seeing pro-eating disorder content can drive engagement, but ultimately be very bad for mental health.

Consider the following questions:

* Why do social media companies optimize for engagement?
* What would be an alternative optimization target? How would the outcomes differ, both for users and for the companies' profits?

## Understanding bias
The term "bias" is overloaded, and can have the following definitions:

* (Statistical) bias: the tendency of an algorithm to produce one solution over another, even though some alternatives may be just as good, or better. Statistical bias can have multiple sources, which we'll get into below. 
* (Social) bias: outcomes are unfair to one or more social groups. Social bias can be the result of statistical bias (i.e., an algorithm giving preferential treatment to one social group over others), but can also occur outside of a machine learning context. 

### Sources of statistical bias

#### Algorithmic bias
Algorithmic bias is the tendency of an algorithm to favor one solution over another. Algorithmic bias is not always bad, and may sometimes be encoded for by algorithm developers. For instance, linear regression with L0-regularization displays algorithmic bias towards sparse classifiers (i.e., classifiers where most weights are 0). This bias may be desirable in settings where human interpretability is important.

But algorithmic bias can also occur unintentionally: for instance, if there is data bias (described below), this may lead algorithm developers to select an algorithm that is ill-suited to underrepresented groups. Then, even if the data bias is rectified, sticking with the original algorithm choice may not fix biased outcomes.


#### Data bias: 
Data bias is when the available training data is not accurate or representative of the target population. Data bias is extremely common (it's often hard to collect perfectly-representative, and perfectly-accurate data), and care arise in multiple ways:

* Measurement error - if a tool is not well calibrated, measurements taken by that tool won't be accurate. Likewise, human biases can lead to measurement error, for instance, if people systematically over-report their height on dating apps, or if doctors do not believe patient's self-reports of their pain levels.
* Response bias - for instance, when conducting a survey about customer satisfaction, customers who had very positive or very negative experiences may be more likely to respond. 
* Representation bias - the data is not well representative of the whole population. For instance, doing clinical trials primarily on white men means that women and other races are not well represented in data.

Through the rest of this lesson, if we use the term "bias" without any additional context, we will be referring to social bias that stems from statistical bias.

:::::::::::::::::::::::::::::::::::::: challenge

### Case Study

With a partner or small group, choose one of the three case study options. Read or watch individually, then discuss as a group how bias manifested in the training data, and what strategies could correct for it.

After the discussion, share with the whole workshop what you discussed.

1. [Predictive policing](https://www.technologyreview.com/2019/02/13/137444/predictive-policing-algorithms-ai-crime-dirty-data/)
2. [Facial recognition](http://gendershades.org/) (video, 5 min.)
3. [Amazon hiring tool](https://www.aclu.org/news/womens-rights/why-amazons-automated-hiring-tool-discriminated-against)

::::::::::::::::::::::::::::::::::::::::::::::::::


::::::::::::::::::::::::::::::::::::: keypoints 

- Some tasks are not appropriate for machine learning due to ethical concerns. 
- Machine learning tasks should have a valid prediction target that maps clearly to the real-world goal.
- Training data can be biased due to societal inequities, errors in the data collection process, and lack of attention to careful sampling practices.
- "Bias" also refers to statistical bias, and certain algorithms can be biased towards some solutions.

::::::::::::::::::::::::::::::::::::::::::::::::
