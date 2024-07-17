---
title: "Interpretablility versus explainability"
teaching: 0
exercises: 2
---

:::::::::::::::::::::::::::::::::::::: questions 

- What are popular machine learning models? 
- What are model intepretability and model explainability? Why are they important?
- Which should you choose: interpretable models or explainable models?

::::::::::::::::::::::::::::::::::::::::::::::::

::::::::::::::::::::::::::::::::::::: objectives

- Showcase machine learning models that are widely used in practice.
- Understand and distinguish between explainable machine learning models and interpretable machine learning models.
- Describe two reasons when deciding which model to choose.

::::::::::::::::::::::::::::::::::::::::::::::::

::::::::::::::::::::::::::::::::::::: keypoints 

- **Model Explainability vs. Model Interpretability:**
   - **Interpretability:** Refers to the degree to which a human can understand the cause of a decision made by a model. It is essential for verifying the correctness of the model, ensuring compliance with regulations, and enabling effective debugging.
   - **Explainability:** Refers to the extent to which the internal mechanics of a machine learning model can be explained in human terms. It is crucial for understanding how models make decisions, ensuring transparency, and building trust with stakeholders.
   

- **Choosing Between Explainable and Interpretable Models:**
   - **When Transparency is Critical:** Opt for interpretable models (e.g., linear regression, decision trees) when it is essential to have a clear understanding of how decisions are made, such as in healthcare or finance.
   - **When Performance is a Priority:** Choose explainable models (e.g., neural networks, gradient boosting machines) when predictive accuracy is the primary concern, and you can use explanation methods to understand model behavior.

::::::::::::::::::::::::::::::::::::::::::::::::

:::::::::::::::::::::::::::::::::::::: exercise

### Exercise 1: Model Selection for Predicting COVID-19 Progression, a study by [Giotta et al.](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC9602523/)

**Objective:** 

To predict bad outcomes (death or transfer to an intensive care unit) from COVID-19 patients using hematological, biochemical, and inflammatory biomarkers.

**Motivation:** 

In the early days of the COVID-19 pandemic, healthcare professionals around the world faced unprecedented challenges. Predicting the progression of the disease and identifying patients at high risk of severe outcomes became crucial for effective treatment and resource allocation. One such study, published on the National Center for Biotechnology Information (NCBI) website, investigated the characteristics of patients who either succumbed to the disease or required intensive care compared to those who recovered.

This study highlighted the critical role of various biomarkers, such as hematological, biochemical, and inflammatory markers, in understanding disease progression. However, simply identifying these markers was not enough. Clinicians needed tools that could not only predict outcomes with high accuracy but also provide clear, understandable reasons for their predictions. 


**Dataset Specification:**
Hematological biomarkers included white blood cells, neutrophils count, lymphocytes count, monocytes count, eosinophils count, platelet count, cluster of differentiation (CD)4, CD8 percentages, and hemoglobin. Biochemical markers were albumin, alanine aminotransferase, aspartate aminotransferase, total bilirubin, creatinine, creatinine kinase, lactate dehydrogenase (LDH), cardiac troponin I, myoglobin, and creatine kinase-MB. The coagulation markers were prothrombin time, activated partial thromboplastin time (APTT), and D-dimer. The inflammatory biomarkers were C-reactive protein (CRP), serum ferritin, procalcitonin (PCT), erythrocyte sedimentation rate, and interleukin and tumor necrosis factor-alpha (TNFα) levels.

**Some statistics from the dataset:**

**Table 1**: Main characteristics of the patients included in the study at baseline and results of comparison of percentage between outcome using chi-square or Fisher exact test.

|                              | Death or Transferred to Intensive Care Unit (n = 32) | Discharged Alive (n = 113) | p-Value |
|------------------------------|------------------------------------------------------|----------------------------|---------|
|                              | N    | %         | N  | %     |         |
| **Sex**                      |      |           |    |       |         |
| Male                         | 18   | 56.25%    | 61 | 53.98%| 1.00    |
| Female                       | 14   | 43.75%    | 52 | 46.02%| 1.00    |
| **Symptoms**                 |      |           |    |       |         |
| Dyspnea                      | 12   | 37.50%    | 52 | 46.02%| 0.999   |
| Cough                        | 5    | 15.63%    | 35 | 30.97%| 1.00    |
| Fatigue                      | 7    | 21.88%    | 30 | 26.55%| 1.00    |
| Headache                     | 2    | 6.25%     | 12 | 10.62%| 1.00    |
| Confusion                    | 1    | 3.13%     | 9  | 7.96% | 1.00    |
| Nausea                       | 1    | 3.13%     | 8  | 7.08% | 1.00    |
| Sick                         | 1    | 3.13%     | 6  | 5.31% | 1.00    |
| Pharyngitis                  | 1    | 3.13%     | 6  | 5.31% | 1.00    |
| Nasal congestion             | 1    | 3.13%     | 3  | 2.65% | 0.999   |
| Arthralgia                   | 0    | 0.00%     | 3  | 2.65% | 1.00    |
| Myalgia                      | 1    | 3.13%     | 2  | 1.77% | 0.997   |
| Arrhythmia                   | 3    | 9.38%     | 12 | 10.62%| 1.00    |
| **Comorbidity**              |      |           |    |       |         |
| Hypertension                 | 12   | 37.50%    | 71 | 62.83%| 0.356   |
| Cardiovascular disease       | 12   | 37.50%    | 43 | 38.05%| 1.00    |
| Diabetes                     | 11   | 34.38%    | 35 | 30.97%| 1.00    |
| Cerebrovascular disease      | 9    | 28.13%    | 19 | 16.81%| 0.896   |
| Chronic kidney disease       | 8    | 25.00%    | 14 | 12.39%| 0.585   |
| COPD                         | 5    | 15.63%    | 14 | 12.39%| 0.999   |
| Tumors                       | 5    | 15.63%    | 11 | 9.73% | 0.986   |
| Hepatitis B                  | 0    | 0.00%     | 6  | 5.31% | 0.974   |
| Immunopathological disease   | 1    | 3.13%     | 5  | 4.42% | 1.00    |

**Table 2**: Comparison of clinical characteristics and laboratory findings between patients who died or were transferred to ICU and those who were discharged alive.

|                               | Patients Deaths or Transferred to ICU (n = 32) | Patients Alive (n = 113) | p-Value |
|-------------------------------|------------------------------------------------|--------------------------|---------|
|                               | Median | Q1   | Q3   | Median | Q1   | Q3   |         |
| **Age (years)**               | 78.0   | 67.0 | 85.75| 70.0   | 57.0 | 82.0 | 0.011   |
| **Temperature (°C)**          | 36.5   | 36.0 | 36.9 | 36.4   | 36.2 | 36.7 | 0.715   |
| **Respiratory rate (rpm)**    | 20.0   | 18.0 | 20.0 | 18.0   | 18.0 | 20.0 | 0.110   |
| **Cardiac frequency (rpm)**   | 79.0   | 70.0 | 90.0 | 75.0   | 70.0 | 85.0 | 0.157   |
| **Systolic blood pressure (mmHg)** | 137.5 | 116.0 | 150.0 | 130.0 | 110.0 | 150.0 | 0.647   |
| **Diastolic blood pressure (mmHg)** | 77.5  | 65.0 | 83.0 | 75.0  | 70.0 | 90.0 | 0.943   |
| **Temperature at admission (°C)** | 36.0 | 35.7 | 36.4 | 36.0 | 35.7 | 36.5 | 0.717   |
| **Percentage of O2 saturation** | 90.0 | 87.0 | 95.0 | 92.0  | 88.0 | 95.0 | 0.415   |
| **FiO2 (%)**                  | 100.0  | 96.0 | 100.0| 100.0  | 100.0| 100.0| 0.056   |
| **Neutrophil count (*10^3/µL)** | 7.98  | 4.75 | 10.5 | 5.38  | 3.77 | 7.7  | 0.006   |
| **Lymphocyte count (*10^3/µL)** | 1.34  | 0.85 | 1.98 | 1.43  | 1.04 | 2.1  | 0.201   |
| **Platelet count (*10^3/µL)** | 202.00 | 147.5| 272.25 | 220.0 | 153.0 | 293.0 | 0.157 |
| **Hemoglobin level (g/dL)**   | 12.7   | 11.8 | 14.5 | 13.5   | 12.0 | 15.0 | 0.017   |
| **Procalcitonin levels (ng/mL)** | 0.11 | 0.07 | 0.27 | 0.07 | 0.05 | 0.1 | 0.063   |
| **CRP (mg/dL)**               | 8.06   | 2.9  | 16.1 | 3.0    | 0.7  | 8.9  | 0.002   |
| **LDH (mg/dL)**               | 307.0  | 258.5| 386.0| 280.0  | 207.0| 385.0| 0.094   |
| **Albumin (mg/dL)**           | 27.0   | 24.5 | 32.5 | 34.0   | 28.0 | 37.0 | 0.000   |
| **ALT (mg/dL)**               | 23.0   | 12.0 | 47.5 | 20.0   | 11.0 | 37.0 | 0.291   |
| **AST (mg/dL)**               | 30.0   | 22.0 | 52.5 | 26.0   | 20.0 | 39.0 | 0.108   |
| **ALP (mg/dL)**               | 70.0   | 53.5 | 88.0 | 71.0   | 54.0 | 94.0 | 0.554   |
| **Direct bilirubin (mg/dL)**  | 0.15   | 0.1  | 0.27 | 0.1    | 0.0  | 0.2  | 0.036   |
| **Indirect bilirubin (mg/dL)** | 0.15  | 0.012| 0.002| 0.1    | 0.0  | 0.2  | 0.376   |
| **Total bilirubin (mg/dL)**   | 0.3    | 0.2  | 0.6  | 0.2    | 0.1  | 0.3  | 0.108   |
| **Creatinine (mg/dL)**        | 1.03   | 0.6  | 1.637| 0.82   | 0.63 | 1.09 | 0.125   |
| **CPK (mg/dL)**               | 79.0   | 47.0 | 194.0| 67.0   | 42.0 | 137.0| 0.330   |
| **Sodium (mg/dL)**            | 140.0  | 137.0| 142.5| 138.0  | 135.0| 140.0| 0.057   |
| **Potassium (mg/dL)**         | 4.4    | 4.0  | 5.0  | 4.0    | 3.6  | 4.3  | 0.029   |
| **INR**                       | 1.1    | 1.0  | 1.2  | 1.0    | 1.0  | 1.1  | 0.049   |
| **IL-6 (pg/mL)**              | 88.8   | 13.7 | 119.7| 9.9    | 7.8  | 40.9 | 0.009   |
| **IgM (AU/mL)**               | 3.4    | 0.0  | 8.1  | 2.6    | 0.0  | 13.3 | 0.323   |
| **IgG (AU/mL)**               | 12.0   | 5.7  | 13.4 | 10.1   | 4.8  | 13.3 | 0.502   |
| **Length of stay (days)**     | 11.0   | 5.75 | 17.0 | 9.0    | 5.0  | 15.0 | 0.837   |

**Real-World Impact:**

During the pandemic, numerous studies and models were developed to aid in predicting COVID-19 outcomes. The study from this paper serves as an excellent example of how detailed patient data can inform model development. By designing a suitable machine learning model, researchers and healthcare providers can not only achieve high predictive accuracy but also ensure that their findings are actionable and trustworthy.

**Discussion Questions:**

1. **Compare the Advantages:**
   - What are the advantages of using explainable models such as decision trees in predicting COVID-19 outcomes?
   - What are the advantages of using black box models such as neural networks in this scenario?

2. **Assess the Drawbacks:**
   - What are the potential drawbacks of using explainable models like decision trees?
   - What are the potential drawbacks of using black box models in healthcare settings?

3. **Decision-Making Criteria:**
   - In what situations might you prioritize an explainable model over a black box model, and why?
   - Are there scenarios where the higher accuracy of black box models justifies their use despite their lack of transparency?

4. **Practical Application:**
   - Design a simple decision tree based on the provided biomarkers to predict bad outcomes.
   - Evaluate how the decision tree can aid healthcare providers in making informed decisions.

::::::::::::::::::::::::::::::::::::::::::::::::

::::::::::::::::::::::::::::::::::::: solution
### Solution
1. **Compare the Advantages:**
   - **Explainable Models:** Allow healthcare professionals to understand and trust the model's decisions, providing clear insights into which biomarkers contribute most to predicting bad outcomes. This transparency is crucial in critical fields such as healthcare, where understanding the decision-making process can inform treatment plans and improve patient outcomes.
   - **Black Box Models:** Often provide higher predictive accuracy, which can be crucial for identifying patterns in complex datasets. They can capture non-linear relationships and interactions that simpler models might miss.

2. **Assess the Drawbacks:**
   - **Explainable Models:** May not capture complex relationships in the data as effectively as black box models, potentially leading to lower predictive accuracy in some cases.
   - **Black Box Models:** Can be difficult to interpret, which hinders trust and adoption by medical professionals. Without understanding the model's reasoning, it becomes challenging to validate its correctness, ensure regulatory compliance, and effectively debug or refine the model.

3. **Decision-Making Criteria:**
   - **Prioritizing Explainable Models:** When transparency, trust, and regulatory compliance are critical, such as in healthcare settings where understanding and validating decisions is essential.
   - **Using Black Box Models:** When the need for high predictive accuracy outweighs the need for transparency, and when supplementary methods for interpreting the model's output can be employed.

4. **Practical Application:**
   - **Design a Decision Tree:** Using the given biomarkers, create a simple decision tree. Identify key split points (e.g., high CRP levels, elevated LDH) and illustrate how these markers can be used to predict bad outcomes. Tools like scikit-learn or any decision tree visualization tool can be used.
   - **Example Decision Tree:** Here is a [Decision Tree](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC9602523/figure/ijerph-19-13016-f003/) found by Giotta et al.

::::::::::::::::::::::::::::::::::::::::::::::::

:::::::::::::::::::::::::::::::::::::: exercise
### Exercise2: COVID-19 Diagnosis Using Chest X-Rays, a study by [Ucar and Korkmaz](https://www.sciencedirect.com/science/article/pii/S0306987720307702)

**Objective:** Diagnose COVID-19 through chest X-rays.

**Motivation:**

The COVID-19 pandemic has had an unprecedented impact on global health, affecting millions of people worldwide. One of the critical challenges in managing this pandemic is the rapid and accurate diagnosis of infected individuals. Traditional methods, such as the Reverse Transcription Polymerase Chain Reaction (RT-PCR) test, although widely used, have several drawbacks. These tests are time-consuming, require specialized equipment and personnel, and often suffer from low detection rates, necessitating multiple tests to confirm a diagnosis.

In this context, radiological imaging, particularly chest X-rays, has emerged as a valuable tool for COVID-19 diagnosis. Early studies have shown that COVID-19 causes specific abnormalities in chest X-rays, such as ground-glass opacities, which can be used as indicators of the disease. However, interpreting these images requires expertise and time, both of which are in short supply during a pandemic.

To address these challenges, researchers have turned to machine learning techniques...

**Dataset Specification:** [Chest X-ray images](https://ars.els-cdn.com/content/image/1-s2.0-S0306987720307702-gr5.jpg)

**Real-World Impact:**

The COVID-19 pandemic highlighted the urgent need for rapid and accurate diagnostic tools. Traditional methods like RT-PCR tests, while effective, are often time-consuming and have variable detection rates. Using chest X-rays for diagnosis offers a quicker and more accessible alternative. By analyzing chest X-rays, healthcare providers can swiftly identify COVID-19 cases, enabling timely treatment and isolation measures. Developing a machine learning method that can quickly and accurately analyze chest X-rays can significantly enhance the speed and efficiency of the healthcare response, especially in areas with limited access to RT-PCR testing.


**Discussion Questions:**

1. **Compare the Advantages:**
   - What are the advantages of using deep neural networks in diagnosing COVID-19 from chest X-rays?
   - What are the advantages of traditional methods, such as genomic data analysis, for COVID-19 diagnosis?

2. **Assess the Drawbacks:**
   - What are the potential drawbacks of using deep neural networks for COVID-19 diagnosis from chest X-rays?
   - How do these drawbacks compare to those of traditional methods?

3. **Decision-Making Criteria:**
   - In what situations might you prioritize using deep neural networks over traditional methods, and why?
   - Are there scenarios where the rapid availability of X-ray results justifies the use of deep neural networks despite potential drawbacks?

4. **Practical Application:**
   - Design a simple deep neural network architecture for diagnosing COVID-19 from chest X-rays.
   - Evaluate how this deep learning model can aid healthcare providers in making informed decisions quickly.

::::::::::::::::::::::::::::::::::::::::::::::::

::::::::::::::::::::::::::::::::::::: solution
### Solution
1. **Compare the Advantages:**
   - **Deep Neural Networks:** Provide high accuracy (e.g., 98%) in diagnosing COVID-19 from chest X-rays, offering a quick and non-invasive diagnostic tool. They can handle large amounts of image data and identify complex patterns that might be missed by human eyes.
   - **Traditional Methods:** Provide detailed and specific diagnostic information by analyzing genomic data and biomarkers, which can be crucial for understanding the virus's behavior and patient response.

2. **Assess the Drawbacks:**
   - **Deep Neural Networks:** Require large labeled datasets for training, which may not always be available. The models can be seen as "black boxes", making it challenging to interpret their decisions without additional explainability methods.
   - **Traditional Methods:** Time-consuming and may have lower detection accuracy. They often require specialized equipment and personnel, leading to delays in diagnosis.

3. **Decision-Making Criteria:**
   - **Prioritizing Deep Neural Networks:** When rapid diagnosis is critical, and chest X-rays are readily available. Useful in large-scale screening scenarios where speed is more critical than the detailed understanding provided by genomic data.
   - **Using Traditional Methods:** When detailed and specific information about the virus is needed for treatment planning, and when the availability of genomic data and biomarkers is not a bottleneck.

4. **Practical Application:**
   - **Design a Neural Network:** Create a simple convolutional neural network (CNN) architecture using tools like TensorFlow or PyTorch. Use a dataset of labeled chest X-ray images to train and validate the model.
   - **Example Model:** Here is a [model proposed by Ucar and Korkmaz](https://ars.els-cdn.com/content/image/1-s2.0-S0306987720307702-ga1.jpg)

     - **Evaluate the Model:** Train the model on your dataset and evaluate its performance. Discuss how this model can help healthcare providers make quick and accurate diagnoses.

::::::::::::::::::::::::::::::::::::::::::::::::


