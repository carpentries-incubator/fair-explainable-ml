---
title: "OOD Detection: Overview, Output-Based Methods"
teaching: 0
exercises: 0
---
:::::::::::::::::::::::::::::::::::::::: questions

- What are out-of-distribution (OOD) data and why is detecting them important in machine learning models?
- How do output-based methods like softmax, energy-based, and distance-based methods work for OOD detection?
- What are the limitations of output-based OOD detection methods?
::::::::::::::::::::::::::::::::::::::::::::::::::
::::::::::::::::::::::::::::::::::::::: objectives

- Understand the concept of out-of-distribution data and its significance in building trustworthy machine learning models.
- Learn about different output-based methods for OOD detection, including softmax and energy-based methods
- Identify the strengths and limitations of output-based OOD detection techniques.
::::::::::::::::::::::::::::::::::::::::::::::::::
# Introduction to Out-of-Distribution (OOD) Data
## What is OOD data?
Out-of-distribution (OOD) data refers to data that significantly differs from the training data on which a machine learning model was built. The difference can arise from either:

- Semantic shift: OOD sample is drawn from a class that was not present during training
- Covariate shift: OOD sample is drawn from a different domain; input feature distribution is drastically different than training data

When an ML model encounters OOD data, its performance can degrade significantly because the model is not equipped to handle these unfamiliar instances.

**TODO**: Add closed/open-world image similar to Sharon Li's tutorial at 4:28: https://www.youtube.com/watch?v=hgLC9_9ZCJI

## Why does OOD data matter?
Model reliability: Models trained on a specific distribution might make incorrect predictions on OOD data, leading to unreliable outputs. In critical applications (e.g., healthcare, autonomous driving), encountering OOD data without proper handling can have severe consequences.

### Ex1: Tesla crashes into jet
In April 2022, a [Tesla Model Y crashed into a $3.5 million private jet](https://www.newsweek.com/video-tesla-smart-summon-mode-ramming-3m-jet-viewed-34m-times-1700310 ) at an aviation trade show in Spokane, Washington, while operating on the "Smart Summon" feature. The feature allows Tesla vehicles to autonomously navigate parking lots to their owners, but in this case, it resulted in a significant mishap.
- The Tesla was summoned by its owner using the Tesla app, which requires holding down a button to keep the car moving. The car continued to move forward even after making contact with the jet, pushing the expensive aircraft and causing notable damage.
- The crash highlighted several issues with Tesla's Smart Summon feature, particularly its object detection capabilities. The system failed to recognize and appropriately react to the presence of the jet, a problem that has been observed in other scenarios where the car's sensors struggle with objects that are lifted off the ground or have unusual shapes.


### Ex2: IBM Watson for Oncology
IBM Watson for Oncology faced several issues due to OOD data. The system was primarily trained on data from Memorial Sloan Kettering Cancer Center (MSK), which did not generalize well to other healthcare settings. This led to the following problems:
1. Unsafe Recommendations: Watson for Oncology provided treatment recommendations that were not safe or aligned with standard care guidelines in many cases outside of MSK. This happened because the training data was not representative of the diverse medical practices and patient populations in different regions
2. Bias in Training Data: The system's recommendations were biased towards the practices at MSK, failing to account for different treatment protocols and patient needs elsewhere. This bias is a classic example of an OOD issue, where the model encounters data (patients and treatments) during deployment that significantly differ from its training data

### Ex3: Doctors using GPT3
#### Misdiagnosis and Inaccurate Medical Advice
In various studies and real-world applications, GPT-3 has been shown to generate inaccurate medical advice when faced with OOD data. This can be attributed to the fact that the training data, while extensive, does not cover all possible medical scenarios and nuances, leading to hallucinations or incorrect responses when encountering unfamiliar input.

A [study published by researchers at Stanford](https://hai.stanford.edu/news/generating-medical-errors-genai-and-erroneous-medical-references) found that GPT-3, even when using retrieval-augmented generation, provided unsupported medical advice in about 30% of its statements. For example, it suggested the use of a specific dosage for a defibrillator based on monophasic technology, while the cited source only discussed biphasic technology, which operates differently.

#### Fake Medical Literature References
Another critical OOD issue is the generation of fake or non-existent medical references by LLMs. When LLMs are prompted to provide citations for their responses, they sometimes generate references that sound plausible but do not actually exist. This can be particularly problematic in academic and medical contexts where accurate sourcing is crucial.

In [evaluations of GPT-3's ability to generate medical literature references](https://hai.stanford.edu/news/generating-medical-errors-genai-and-erroneous-medical-references) , it was found that a significant portion of the references were either entirely fabricated or did not support the claims being made. This was especially true for complex medical inquiries that the model had not seen in its training data.
# Detecting and Handling OOD Data
Given the problems posed by OOD data, a reliable model should identify such instances, and then either:
1. Reject them during inference
2. Hand them off to a model trained on a more similar distribution (an in-distribution)

How can we determine whether a given instance is OOD or ID? Over the past several years, there have been a wide assortment of new methods developed to tackle this task. In this episode, we will cover a few of the most common approaches and discuss advantages/disadvantages of each.

## Threshold-based methods
Threshold-based methods are one of the simplest and most intuitive approaches for detecting out-of-distribution (OOD) data. The central idea is to define a threshold on a certain score or confidence measure, beyond which the data point is considered out-of-distribution. Typically, these scores are derived from the model's output probabilities or other statistical measures of uncertainty. Common approaches include:

### Output-based
- Softmax Scores: The softmax output of a neural network represents the predicted probabilities for each class. A common threshold-based method involves setting a confidence threshold, and if the maximum softmax score of an instance falls below this threshold, it is flagged as OOD.
- Energy: Energy measures the uncertainty in the predicted probability distribution. High Energy indicates high uncertainty. By setting a threshold on the Energy value, instances with Energy above the threshold can be classified as OOD.

### Distance-based
- Distance: This method calculates the distance of an instance from the distribution of training data features. If the distance is beyond a certain threshold, the instance is considered OOD.
  
# Example 1: Softmax scores
```python
n_epochs = 10
verbose = False
alpha=0.2

```
```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from keras.datasets import fashion_mnist

# Load Fashion MNIST dataset
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

# Define classes for simplicity
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# Prepare OOD data - Sandals (5)
ood_data = test_images[test_labels == 5]
ood_labels = test_labels[test_labels == 5]
print(f'ood_data.shape={ood_data.shape}')

# Filter data for T-shirts (0) and Trousers (1) as in-distribution
train_filter = np.isin(train_labels, [0, 1])
test_filter = np.isin(test_labels, [0, 1])

train_data = train_images[train_filter]
train_labels = train_labels[train_filter]
print(f'train_data.shape={train_data.shape}')

test_data = test_images[test_filter]
test_labels = test_labels[test_filter]
print(f'test_data.shape={test_data.shape}')

# Display examples of in-distribution and OOD data
plt.figure(figsize=(10, 4))
for i in range(5):
    plt.subplot(2, 5, i + 1)
    plt.imshow(train_data[i], cmap='gray')
    plt.title("In-Dist")
    plt.axis('off')
for i in range(5):
    plt.subplot(2, 5, i + 6)
    plt.imshow(ood_data[i], cmap='gray')
    plt.title("OOD")
    plt.axis('off')
plt.show()

```
```python
train_labels[0:5] # shirts are 0s, pants are 1s
```
```python
# Flatten images for PCA and logistic regression
train_data_flat = train_data.reshape((train_data.shape[0], -1))
test_data_flat = test_data.reshape((test_data.shape[0], -1))
ood_data_flat = ood_data.reshape((ood_data.shape[0], -1))

print(f'train_data_flat.shape={train_data_flat.shape}')
print(f'test_data_flat.shape={test_data_flat.shape}')
print(f'ood_data_flat.shape={ood_data_flat.shape}')
```
## Visualizing OOD and ID data

### PCA
PCA visualization can provide insights into how well a model is separating ID and OOD data. If the OOD data overlaps significantly with ID data in the PCA space, it might indicate that the model could struggle to correctly identify OOD samples.
```python
# Perform PCA to visualize the first two principal components
from sklearn.decomposition import PCA

pca = PCA(n_components=2)
train_data_pca = pca.fit_transform(train_data_flat)
test_data_pca = pca.transform(test_data_flat)
ood_data_pca = pca.transform(ood_data_flat)

# Plotting PCA components
plt.figure(figsize=(10, 6))
scatter1 = plt.scatter(train_data_pca[train_labels == 0, 0], train_data_pca[train_labels == 0, 1], c='blue', label='T-shirts (ID)', alpha=0.5)
scatter2 = plt.scatter(train_data_pca[train_labels == 1, 0], train_data_pca[train_labels == 1, 1], c='red', label='Trousers (ID)', alpha=0.5)
scatter3 = plt.scatter(ood_data_pca[:, 0], ood_data_pca[:, 1], c='green', label='Sandals (OOD)', edgecolor='k')

# Create a single legend for all classes
plt.legend(handles=[scatter1, scatter2, scatter3], loc="upper right")
plt.xlabel('First Principal Component')
plt.ylabel('Second Principal Component')
plt.title('PCA of In-Distribution and OOD Data')
plt.show()
```
### UMAP (or similar)

However, PCA also has some limitations that might make other techniques, such as Uniform Manifold Approximation and Projection (UMAP), more suitable in certain scenarios:

1. **Focus on Linear Relationships**: PCA is a linear dimensionality reduction technique. It assumes that the directions of maximum variance in the data can be captured by linear combinations of the original features. This can be a limitation when the data has complex, non-linear relationships, as PCA may not capture the true structure of the data.

2. **Overlooked Non-Linear Structures**: OOD data might differ from ID data in non-linear ways that PCA cannot capture. For instance, if OOD data lies in a non-linear manifold that differs from the ID data manifold, PCA may not effectively separate them.

3. **Global Structure vs. Local Structure**: PCA emphasizes capturing the global variance in the data, which might not be ideal if the distinctions between ID and OOD data are subtle and local. In such cases, PCA might not highlight the differences effectively.

4. **Alternative Methods Like UMAP**:
    - **UMAP**: Uniform Manifold Approximation and Projection (UMAP) is a non-linear dimensionality reduction technique that preserves both the global structure and the local neighborhood relationships in the data. UMAP is often better at maintaining the continuity of data points that lie on non-linear manifolds.
    - **Better Clustering**: UMAP tends to provide more meaningful visualizations for clustering tasks, making it easier to identify distinct clusters of ID and OOD data.
    - **Preservation of Local Distances**: UMAP preserves local distances more effectively than PCA, which can be crucial for distinguishing between ID and OOD data that are close in the original high-dimensional space but separate in the underlying manifold.

5. **Visualization Clarity**: UMAP often provides clearer and more interpretable visualizations for complex datasets. It can reveal patterns and structures that PCA might miss, making it a valuable tool for analyzing ID and OOD distributions.


```python
!pip install umap-learn
```
```python
import umap

# Perform UMAP to visualize the data
umap_reducer = umap.UMAP(n_components=2, random_state=42)
combined_data = np.vstack([train_data_flat, ood_data_flat])
combined_labels = np.hstack([train_labels, np.full(ood_data_flat.shape[0], 2)])  # Use 2 for OOD class

umap_results = umap_reducer.fit_transform(combined_data)

# Split the results back into in-distribution and OOD data
umap_in_dist = umap_results[:len(train_data_flat)]
umap_ood = umap_results[len(train_data_flat):]
```
```python
# Plotting UMAP components
plt.figure(figsize=(10, 6))

# Plot in-distribution data
scatter1 = plt.scatter(umap_in_dist[train_labels == 0, 0], umap_in_dist[train_labels == 0, 1], c='blue', label='T-shirts (ID)', alpha=alpha)
scatter2 = plt.scatter(umap_in_dist[train_labels == 1, 0], umap_in_dist[train_labels == 1, 1], c='red', label='Trousers (ID)', alpha=alpha)

# Plot OOD data
scatter3 = plt.scatter(umap_ood[:, 0], umap_ood[:, 1], c='green', label='Sandals (OOD)', edgecolor='k', alpha=alpha)

# Create a single legend for all classes
plt.legend(handles=[scatter1, scatter2, scatter3], loc="upper right")
plt.xlabel('First UMAP Component')
plt.ylabel('Second UMAP Component')
plt.title('UMAP of In-Distribution and OOD Data')
plt.show()
```
The warning message indicates that UMAP has overridden the n_jobs parameter to 1 due to the random_state being set. This behavior ensures reproducibility by using a single job. If you want to avoid the warning and still use parallelism, you can remove the random_state parameter. However, removing random_state will mean that the results might not be reproducible.
## Model-Dependent Visualization
The choice of visualization technique can align with the nature of the model being used. Here's how you might frame this idea:

* Linear Models and PCA: If you're using a linear model, PCA can be more appropriate for visualizing in-distribution (ID) and out-of-distribution (OOD) data because both PCA and linear models operate under linear assumptions. PCA will effectively capture the main variance in the data as seen by the linear model, making it easier to understand the decision boundaries and how OOD data deviates from the ID data within those boundaries.

* Non-Linear Models and UMAP: For non-linear models, techniques like UMAP are more suitable because they preserve both local and global structures in a non-linear fashion, similar to how non-linear models capture complex relationships in the data. UMAP can provide a more accurate visualization of the data manifold, highlighting distinctions between ID and OOD data that may not be apparent with PCA.
```python
# Train a logistic regression classifier
model = LogisticRegression(max_iter=10, solver='lbfgs', multi_class='multinomial').fit(train_data_flat, train_labels)

# # Evaluate the model on in-distribution data
# in_dist_preds = model.predict(test_data_flat)
# in_dist_accuracy = accuracy_score(test_labels, in_dist_preds)
# print(f'In-Distribution Accuracy: {in_dist_accuracy:.2f}')
```
```python
# Predict probabilities using the model on OOD data (Sandals)
ood_probs = model.predict_proba(ood_data_flat)
avg_ood_prob = np.mean(ood_probs, 0)
print(f"Avg. probability of sandal being T-shirt: {avg_ood_prob[0]:.4f}")
print(f"Avg. probability of sandal being trousers: {avg_ood_prob[1]:.4f}")

id_probs = model.predict_proba(train_data_flat)
id_probs_shirts = id_probs[train_labels==0,:]
id_probs_pants = id_probs[train_labels==1,:]
```
```python
import matplotlib.pyplot as plt

# Creating the figure and subplots
fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=False)
bins=60
# Plotting the histogram of probabilities for OOD data (Sandals)
axes[0].hist(ood_probs[:, 0], bins=bins, alpha=0.5, label='T-shirt probability')
axes[0].set_xlabel('Probability')
axes[0].set_ylabel('Frequency')
axes[0].set_title('OOD Data (Sandals)')
axes[0].legend()

# Plotting the histogram of probabilities for ID data (T-shirt)
axes[1].hist(id_probs_shirts[:, 0], bins=bins, alpha=0.5, label='T-shirt probability', color='orange')
axes[1].set_xlabel('Probability')
axes[1].set_title('ID Data (T-shirt)')
axes[1].legend()

# Plotting the histogram of probabilities for ID data (Pants)
axes[2].hist(id_probs_pants[:, 1], bins=bins, alpha=0.5, label='Pants probability', color='green')
axes[2].set_xlabel('Probability')
axes[2].set_title('ID Data (Pants)')
axes[2].legend()

# Adjusting layout
plt.tight_layout()

# Displaying the plot
plt.show()

```
```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

# Assuming id_probs_shirts and id_probs_pants are subsets of id_probs
id_probs_shirts = id_probs[train_labels == 0]
id_probs_pants = id_probs[train_labels == 1]

# Create figure
plt.figure(figsize=(10, 6))

# Define bins
alpha = 0.3

# Plot PDF for ID T-shirt (T-shirt probability)
density_id_shirts = gaussian_kde(id_probs_shirts[:, 0])
x_id_shirts = np.linspace(0, 1, 1000)
plt.plot(x_id_shirts, density_id_shirts(x_id_shirts), label='ID T-shirt (T-shirt probability)', color='orange', alpha=alpha)

# Plot PDF for ID Pants (Pants probability)
density_id_pants = gaussian_kde(id_probs_pants[:, 1])
x_id_pants = np.linspace(0, 1, 1000)
plt.plot(x_id_pants, density_id_pants(x_id_pants), label='ID Pants (Pants probability)', color='green', alpha=alpha)

# Plot PDF for OOD (T-shirt probability)
density_ood = gaussian_kde(ood_probs[:, 0])
x_ood = np.linspace(0, 1, 1000)
plt.plot(x_ood, density_ood(x_ood), label='OOD (T-shirt probability)', color='blue', alpha=alpha)

# Adding labels and title
plt.xlabel('Probability')
plt.ylabel('Density')
plt.title('Probability Density Distributions for OOD and ID Data')
plt.legend()

# Displaying the plot
plt.show()

```
```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Assuming ood_probs, id_probs, and train_labels are defined
# Threshold values
upper_threshold = 0.9999

# Classifying OOD examples (sandals)
ood_classifications = np.where(ood_probs[:, 1] >= upper_threshold, 1,  # classified as pants
                               np.where(ood_probs[:, 0] >= upper_threshold, 0,  # classified as shirts
                                        -1))  # classified as OOD
ood_classifications

id_probs
# Classifying ID examples (T-shirts and pants)
id_classifications = np.where(id_probs[:, 1] >= upper_threshold, 1,  # classified as pants
                              np.where(id_probs[:, 0] >= upper_threshold, 0,  # classified as shirts
                                       -1))  # classified as OOD

id_classifications

# Combine OOD and ID classifications and true labels
all_predictions = np.concatenate([ood_classifications, id_classifications])
all_true_labels = np.concatenate([-1 * np.ones(ood_classifications.shape), train_labels])

# Confusion matrix
cm = confusion_matrix(all_true_labels, all_predictions, labels=[0, 1, -1])

# Plotting the confusion matrix
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Shirt", "Pants", "OOD"])
disp.plot(cmap=plt.cm.Blues)
plt.title('Confusion Matrix for OOD and ID Classification')
plt.show()

```
With a very conservative threshold, we can make sure very few OOD samples are incorrectly classified as ID. However, the flip side is that conservative thresholds tend to incorrectly classify many ID samples as being OOD. In this case, we incorrectly assume almost 20% of shirts are OOD samples.

Quick exercise: What threhsold is required to ensure that no OOD samples are incorrectly considered as IID? What percentage of ID samples are mistaken as OOD at this threshold? Answer:  0.9999, (3826+2414)/(3826+2414+2174+3586)=52%
## Iterative Threshold Determination (setup as exercise?)

In practice, selecting an appropriate threshold is an iterative process that balances the trade-off between correctly identifying in-distribution (ID) data and accurately flagging out-of-distribution (OOD) data. Here's how you can iteratively determine the threshold:

* Define Evaluation Metrics: Decide on the performance metrics you want to optimize, such as accuracy, precision, recall, or the F1 score for both ID and OOD detection.

* Evaluate Over a Range of Thresholds: Test different threshold values and evaluate the performance on a validation set containing both ID and OOD data.

* Select the Optimal Threshold: Choose the threshold that provides the best balance according to your chosen metrics.


**Possible exercise (might not be our first priority since these methods aren't as popular anymore)**: Use the below code to determine what threshold should be set to ensure precision = 100%. What threshold is required for recall to be 100%? What threshold gives the highest F1 score?

Note: F1 scores can be calculated per class, and then averaged in different ways (macro, micro, or weighted) when dealing with multiclass or multilabel classification problems. Here are the key types of averaging methods:

* Macro-Averaging: Calculates the F1 score for each class independently and then takes the average of these scores. This treats all classes equally, regardless of their support (number of true instances for each class).

* Micro-Averaging: Aggregates the contributions of all classes to compute the average F1 score. This is typically used for imbalanced datasets as it gives more weight to classes with more instances.

* Weighted-Averaging: Calculates the F1 score for each class independently and then takes the average, weighted by the number of true instances for each class. This accounts for class imbalance by giving more weight to classes with more instances.
```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, confusion_matrix, ConfusionMatrixDisplay

# Define thresholds to evaluate
thresholds = np.linspace(.5, 1, 50)

# Store evaluation metrics for each threshold
accuracies = []
precisions = []
recalls = []
f1_scores = []

for threshold in thresholds:

  # threshold = 1
  # Classify OOD examples
  ood_classifications = np.where(ood_probs[:, 1] >= threshold, 1,  # classified as pants
                                  np.where(ood_probs[:, 0] >= threshold, 0,  # classified as shirts
                                          -1))  # classified as OOD

  # Classify ID examples
  id_classifications = np.where(id_probs[:, 1] >= threshold, 1,  # classified as pants
                                np.where(id_probs[:, 0] >= threshold, 0,  # classified as shirts
                                          -1))  # classified as OOD

  # Combine OOD and ID classifications and true labels
  all_predictions = np.concatenate([ood_classifications, id_classifications])
  all_true_labels = np.concatenate([-1 * np.ones(ood_classifications.shape), train_labels])

  # Evaluate metrics
  precision, recall, f1, _ = precision_recall_fscore_support(all_true_labels, all_predictions, labels=[0, 1], average='macro') # discuss macro vs micro .
  accuracy = accuracy_score(all_true_labels, all_predictions)

  accuracies.append(accuracy)
  precisions.append(precision)
  recalls.append(recall)
  f1_scores.append(f1)

# Find the best thresholds for each metric
best_f1_index = np.argmax(f1_scores)
best_f1_threshold = thresholds[best_f1_index]

best_precision_index = np.argmax(precisions)
best_precision_threshold = thresholds[best_precision_index]

best_recall_index = np.argmax(recalls)
best_recall_threshold = thresholds[best_recall_index]

print(f"Best F1 threshold: {best_f1_threshold}, F1 Score: {f1_scores[best_f1_index]}")
print(f"Best Precision threshold: {best_precision_threshold}, Precision: {precisions[best_precision_index]}")
print(f"Best Recall threshold: {best_recall_threshold}, Recall: {recalls[best_recall_index]}")

# Plot metrics as functions of the threshold
plt.figure(figsize=(12, 8))
plt.plot(thresholds, precisions, label='Precision', color='g')
plt.plot(thresholds, recalls, label='Recall', color='b')
plt.plot(thresholds, f1_scores, label='F1 Score', color='r')

# Add best threshold indicators
plt.axvline(x=best_f1_threshold, color='r', linestyle='--', label=f'Best F1 Threshold: {best_f1_threshold:.2f}')
plt.axvline(x=best_precision_threshold, color='g', linestyle='--', label=f'Best Precision Threshold: {best_precision_threshold:.2f}')
plt.axvline(x=best_recall_threshold, color='b', linestyle='--', label=f'Best Recall Threshold: {best_recall_threshold:.2f}')

plt.xlabel('Threshold')
plt.ylabel('Metric Value')
plt.title('Evaluation Metrics as Functions of Threshold')
plt.legend()
plt.show()
```
# Example 2: Energy-Based OOD Detection

Liu et al., Energy-based Out-of-distribution Detection, NeurIPS 2020

* E(x, y) = energy value

* if x and y are "compatitble", lower energy

* Energy can be turned into probability through Gibbs distribution
    * looks at integral over all possible y's


* With energy scores, ID and OOD distributions become much more separable

* Another "output-based" method like softmax
# Conclusion


```python

```
