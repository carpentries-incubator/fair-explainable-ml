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

### Case study 1: Identifying genetic disorders


### Case study 2: "Reading" a person's face




## Choosing the outcome variable


:::::::::::::::::::::::::::::::::::::: challenge

### Case Study

A well-known [study by Obermeyer et al.](https://escholarship.org/content/qt6h92v832/qt6h92v832.pdf) analyzed an algorithm that hospitals used to assign patients risk scores for various conditions. The algorithm had access to various patient data, such as demographics (e.g., age and sex), the number of chronic conditions, insurance type, diagnoses, and medical costs. The algorithm did not have access to the patient's race.
The patient risk score determined the level of care the patient should receive, with higher-risk patients receiving additional care. 

Obermeyer et al. reveal that the algorithm is biased against Black patients. 
That is, if there are two individuals -- one white and one Black -- with equal health,
the algorithm tends to assign a higher risk score to the white patient, thus giving them
access to higher care quality. 
The target variable used to construct the algorithm is health care cost, and the authors blame this choice of target variable for the algorithm's racial bias.

Discuss the following with a partner or small group:

1. Why might using health care cost as a target variable lead to racially biased outcomes? (Hint: this algorithm was deployed in the United States.)
2. What would be a more appropriate target variable to use instead? 
3. Why do you think the algorithm designers chose to use health care cost as the target variable? 
4. How could the algorithm's racial bias have been caught before deploying it widely? E.g., what processes could the algorithm developers have implemented?

::::::::::::::::::::::::::::::::::::::::::::::::::

:::::::::::::: solution

### Solution
1. Racial disparities in the US lead to Black Americans receiving less care than equally-sick white Americans. 
2. Data focusing on health, not the cost of care, may be more appropriate. For instance, the paper suggests using the number of chronic conditions (or number of chronic conditions that flare up in a given year) as an alternate label.
3. The algorithm designers probably assumed that cost would be a good proxy for health. (Indeed, health needs and health cost *are* correlated.) Another factor may have been the quality of data -- health care billing is standardized and will be present for all patient visits.
4. There's no single right answer -- the algorithm developers could have experimented with using different target variables and compared the outcomes for different racial groups, or they could have engaged with non-technical experts who may be more attuned to the impacts of race disparities in the US health system. 

:::::::::::::::::::::::::



## Understanding bias



:::::::::::::::::::::::::::::::::::::: challenge

### Case Study

With a partner or small group, choose one of the three case study options. Read individually, then discuss as a group how bias manifested in the training data, and what strategies could correct for it.

After the discussion, share with the whole workshop what you discussed.

1. [Predictive policing](https://www.technologyreview.com/2019/02/13/137444/predictive-policing-algorithms-ai-crime-dirty-data/)
2. Computer vision
3. [Amazon hiring tool](https://www.aclu.org/news/womens-rights/why-amazons-automated-hiring-tool-discriminated-against)

::::::::::::::::::::::::::::::::::::::::::::::::::



::::::::::::::::::::::::::::::::::::: keypoints 

- Some tasks are not appropriate for machine learning due to ethical concerns. 
- Machine learning tasks should have a valid prediction target that maps clearly to the real-world goal.
- Training data can be biased due to societal inequities, errors in the data collection process, and lack of attention to careful sampling practices.
- "Bias" also refers to statistical bias, and certain algorithms can be biased towards some solutions.

::::::::::::::::::::::::::::::::::::::::::::::::
