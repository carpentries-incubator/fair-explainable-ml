---
title: "OOD detection: softmax"
teaching: 0
exercises: 0
---
:::::::::::::::::::::::::::::::::::::::: questions

- What is softmax-based out-of-distribution (OOD) detection, and how does it work?
- What are the strengths and limitations of using softmax scores for OOD detection?
- How do threshold choices affect the performance of softmax-based OOD detection?
- How can we assess and improve softmax-based OOD detection through evaluation metrics and visualization?


::::::::::::::::::::::::::::::::::::::::::::::::::

::::::::::::::::::::::::::::::::::::::: objectives

- Understand how softmax scores can be leveraged for OOD detection.
- Explore the advantages and drawbacks of using softmax-based methods for OOD detection.
- Learn how to visualize and interpret softmax-based OOD detection performance using tools like PCA and probability density plots.
- Investigate the impact of thresholds on the trade-offs between detecting OOD and retaining in-distribution data.
- Build a foundation for understanding more advanced output-based OOD detection methods, such as energy-based detection.

::::::::::::::::::::::::::::::::::::::::::::::::::

## Leveraging softmax model outputs
Softmax-based methods are among the most widely used techniques for out-of-distribution (OOD) detection, leveraging the probabilistic outputs of a model to differentiate between in-distribution (ID) and OOD data. These methods are inherently tied to models employing a softmax activation function in their final layer, such as logistic regression or neural networks with a classification output layer. 

The softmax function normalizes the logits (i.e., sum of neuron input without passing through activation function) in the final layer, squeezing the output into a range between 0 and 1. This is useful for interpreting the model’s predictions as probabilities. Softmax probabilities are computed as:  

$$
P(y = k \mid x) = \frac{\exp(f_k(x))}{ \sum_{j} \exp(f_j(x))}
$$

In this lesson, we will train a logistic regression model to classify images from the Fashion MNIST dataset and explore how its softmax outputs can signal whether a given input belongs to the ID classes (e.g., T-shirts or pants) or is OOD (e.g., sandals). While softmax is most naturally applied in models with a logistic activation, alternative approaches, such as applying softmax-like operations post hoc to models with different architectures, are occasionally used. However, these alternatives are less common and may require additional considerations. By focusing on logistic regression, we aim to illustrate the fundamental principles of softmax-based OOD detection in a simple and interpretable context before extending these ideas to more complex architectures.

## Prepare the ID (train and test) and OOD data
In order to determine a threshold that can separate ID data from OOD data (or ensure new test data as ID), we need to sample data from both distributions. OOD data used should be representative of potential new classes (i.e., semanitic shift) that may be seen by your model, or distribution/covariate shifts observed in your application area. 

* ID = T-shirts/Blouses, Pants
* OOD = any other class. For Illustrative purposes, we'll focus on images of sandals as the OOD class.
```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from keras.datasets import fashion_mnist

def prep_ID_OOD_datasests(ID_class_labels, OOD_class_labels):
    """
    Prepares in-distribution (ID) and out-of-distribution (OOD) datasets 
    from the Fashion MNIST dataset.

    Parameters:
    - ID_class_labels: list or array-like, labels for the in-distribution classes.
                       Example: [0, 1] for T-shirts (0) and Trousers (1).
    - OOD_class_labels: list or array-like, labels for the out-of-distribution classes.
                        Example: [5] for Sandals.

    Returns:
    - train_data: np.array, training images for in-distribution classes.
    - test_data: np.array, test images for in-distribution classes.
    - ood_data: np.array, test images for out-of-distribution classes.
    - train_labels: np.array, labels corresponding to the training images.
    - test_labels: np.array, labels corresponding to the test images.
    - ood_labels: np.array, labels corresponding to the OOD test images.

    Notes:
    - The function filters images based on provided class labels for ID and OOD.
    - Outputs include images and their corresponding labels.
    """
    # Load Fashion MNIST dataset
    (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
    
    # Prepare OOD data: Sandals = 5
    ood_filter = np.isin(test_labels, OOD_class_labels)
    ood_data = test_images[ood_filter]
    ood_labels = test_labels[ood_filter]
    print(f'ood_data.shape={ood_data.shape}')
    
    # Filter data for T-shirts (0) and Trousers (1) as in-distribution
    train_filter = np.isin(train_labels, ID_class_labels)
    test_filter = np.isin(test_labels, ID_class_labels)
    
    train_data = train_images[train_filter]
    train_labels = train_labels[train_filter]
    print(f'train_data.shape={train_data.shape}')
    
    test_data = test_images[test_filter]
    test_labels = test_labels[test_filter]
    print(f'test_data.shape={test_data.shape}')

    return train_data, test_data, ood_data, train_labels, test_labels, ood_labels


def plot_data_sample(train_data, ood_data):
    """
    Plots a sample of in-distribution and OOD data.

    Parameters:
    - train_data: np.array, array of in-distribution data images
    - ood_data: np.array, array of out-of-distribution data images

    Returns:
    - fig: matplotlib.figure.Figure, the figure object containing the plots
    """
    fig = plt.figure(figsize=(10, 4))
    N_samples = 7
    for i in range(N_samples):
        plt.subplot(2, N_samples, i + 1)
        plt.imshow(train_data[i], cmap='gray')
        plt.title("In-Dist")
        plt.axis('off')
    for i in range(N_samples):
        plt.subplot(2, N_samples, i + N_samples+1)
        plt.imshow(ood_data[i], cmap='gray')
        plt.title("OOD")
        plt.axis('off')
    
    return fig
```
Load and prepare the ID data (train+test containing shirts and pants) and OOD data (sandals)

## Why not just add the OOD class to training dataset?
OOD data is, by definition, not part of the training distribution. It could encompass anything outside the known classes, which means you'd need to collect a representative dataset for "everything else" to train the OOD class. This is practically impossible because OOD data is often diverse and unbounded (e.g., new species, novel medical conditions, adversarial examples).

The key idea behind threshold-based methods is we want to vet our model against a small sample of potential risk-cases using known OOD data to determine an empirical threshold that *hopefully* extends to other OOD cases that may arise in real-world scenarios. 

That said, a common *next* step in OOD pipelines is to develop new models that handle the OOD data (e.g., adding new classes). The first step, however, is detecting the existence of such OOD data.
```python
# ID: T-shirts (0) and Trousers (1)
# OOD: Sandals (5)
train_data, test_data, ood_data, train_labels, test_labels, ood_labels = prep_ID_OOD_datasests(ID_class_labels=[0,1], OOD_class_labels=[5])
```
Plot sample
```python
fig = plot_data_sample(train_data, ood_data)
plt.show()
```
## Visualizing OOD and ID data with PCA

### PCA
PCA visualization can provide insights into how well a model is separating ID and OOD data. If the OOD data overlaps significantly with ID data in the PCA space, it might indicate that the model could struggle to correctly identify OOD samples.

**Focus on Linear Relationships**: PCA is a linear dimensionality reduction technique. It assumes that the directions of maximum variance in the data can be captured by linear combinations of the original features. This can be a limitation when the data has complex, non-linear relationships, as PCA may not capture the true structure of the data. However, if you're using a linear model (as we are here), PCA can be more appropriate for visualizing in-distribution (ID) and out-of-distribution (OOD) data because both PCA and linear models operate under linear assumptions. PCA will effectively capture the main variance in the data as seen by the linear model, making it easier to understand the decision boundaries and how OOD data deviates from the ID data within those boundaries.
```python
# Flatten images for PCA and logistic regression
train_data_flat = train_data.reshape((train_data.shape[0], -1))
test_data_flat = test_data.reshape((test_data.shape[0], -1))
ood_data_flat = ood_data.reshape((ood_data.shape[0], -1))

print(f'train_data_flat.shape={train_data_flat.shape}')
print(f'test_data_flat.shape={test_data_flat.shape}')
print(f'ood_data_flat.shape={ood_data_flat.shape}')
```
```python
# Perform PCA to visualize the first two principal components
from sklearn.decomposition import PCA

pca = PCA(n_components=2)
train_data_pca = pca.fit_transform(train_data_flat)
test_data_pca = pca.transform(test_data_flat)
ood_data_pca = pca.transform(ood_data_flat)

# Plotting PCA components
plt.figure(figsize=(10, 6))
scatter1 = plt.scatter(train_data_pca[train_labels == 0, 0], train_data_pca[train_labels == 0, 1], c='blue', label='T-shirt/top (ID)', alpha=0.3)
scatter2 = plt.scatter(train_data_pca[train_labels == 1, 0], train_data_pca[train_labels == 1, 1], c='red', label='Pants (ID)', alpha=0.3)
scatter3 = plt.scatter(ood_data_pca[:, 0], ood_data_pca[:, 1], c='green', label='Sandals (OOD)', edgecolor='k')

# Create a single legend for all classes
plt.legend(handles=[scatter1, scatter2, scatter3], loc="upper right")
plt.xlabel('First Principal Component')
plt.ylabel('Second Principal Component')
plt.title('PCA of In-Distribution and OOD Data')
plt.show()
```
From this plot, we see that sandals are more likely to be confused as T-shirts than pants. It also may be surprising to see that these data clouds overlap so much given their semantic differences. Why might this be?

* **Over-reliance on linear relationships**: Part of this has to do with the fact that we're only looking at linear relationships and treating each pixel as its own input feature, which is usually never a great idea when working with image data. In our next example, we'll switch to the more modern approach of CNNs.
* **Semantic gap != feature gap**: Another factor of note is that images that have a wide semantic gap may not necessarily translate to a wide gap in terms of the data's visual features (e.g., ankle boots and bags might both be small, have leather, and have zippers). Part of an effective OOD detection scheme involves thinking carefully about what sorts of data contanimations may be observed by the model, and assessing how similar these contaminations may be to your desired class labels.

## Train and evaluate model on ID data
```python
model = LogisticRegression(max_iter=10, solver='lbfgs', multi_class='multinomial').fit(train_data_flat, train_labels) # 'lbfgs' is an efficient solver that works well for small to medium-sized datasets.

```
Before we worry about the impact of OOD data, let's first verify that we have a reasonably accurate model for the ID data.
```python
# Evaluate the model on in-distribution data
ID_preds = model.predict(test_data_flat)
ID_accuracy = accuracy_score(test_labels, ID_preds)
print(f'In-Distribution Accuracy: {ID_accuracy:.2f}')
```
```python
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay

# Generate and display confusion matrix
cm = confusion_matrix(test_labels, ID_preds, labels=[0, 1])
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['T-shirt/top', 'Pants'])
disp.plot(cmap=plt.cm.Blues)
plt.show()
```
## How does our model view OOD data?

A basic question we can start with is to ask, on average, how are OOD samples classified? Are they more likely to be Tshirts or pants? For this kind of question, we can calculate the probability scores for the OOD data, and compare this to the ID data.
```python
# Predict probabilities using the model on OOD data (Sandals)
ood_probs = model.predict_proba(ood_data_flat)
avg_ood_prob = np.mean(ood_probs, 0)
print(f"Avg. probability of sandal being T-shirt: {avg_ood_prob[0]:.2f}")
print(f"Avg. probability of sandal being pants: {avg_ood_prob[1]:.2f}")

id_probs = model.predict_proba(test_data_flat) # a fairer comparison is to look at test set probabilities (just in case our model is overfitting)
id_probs_shirts = id_probs[test_labels==0,:]
id_probs_pants = id_probs[test_labels==1,:]
avg_tshirt_prob = np.mean(id_probs_shirts, 0)
avg_pants_prob = np.mean(id_probs_pants, 0)

print()
print(f"Avg. probability of T-shirt being T-shirt: {avg_tshirt_prob[0]:.2f}")
print(f"Avg. probability of pants being pants: {avg_pants_prob[1]:.2f}")
```
Based on the difference in averages here, it looks like softmax may provide at least a somewhat useful signal in separating ID and OOD data. Let's take a closer look by plotting histograms of all probability scores across our classes of interest (ID-Tshirt, ID-Pants, and OOD).
```python
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
axes[1].set_title('ID Data (T-shirt/top)')
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
Alternatively, for a better comparison across all three classes, we can use a probability density plot. This will allow for an easier comparison when the counts across classes lie on vastly different scales (i.e., max of 35 vs max of 5000).
```python
from scipy.stats import gaussian_kde

# Create figure
plt.figure(figsize=(10, 6))

# Define bins
alpha = 0.4

# Plot PDF for ID T-shirt (T-shirt probability)
density_id_shirts = gaussian_kde(id_probs_shirts[:, 0])
x_id_shirts = np.linspace(0, 1, 1000)
plt.plot(x_id_shirts, density_id_shirts(x_id_shirts), label='ID T-shirt (T-shirt probability)', color='orange', alpha=alpha)

# Plot PDF for ID Pants (Pants probability)
density_id_pants = gaussian_kde(id_probs_pants[:, 0])
x_id_pants = np.linspace(0, 1, 1000)
plt.plot(x_id_pants, density_id_pants(x_id_pants), label='ID Pants (T-shirt probability)', color='green', alpha=alpha)

# Plot PDF for OOD (T-shirt probability)
density_ood = gaussian_kde(ood_probs[:, 0])
x_ood = np.linspace(0, 1, 1000)
plt.plot(x_ood, density_ood(x_ood), label='OOD (T-shirt probability)', color='blue', alpha=alpha)

# Adding labels and title
plt.xlabel('Probability')
plt.ylabel('Density')
plt.title('Probability Density Distributions for OOD and ID Data')
plt.ylim([0,20])
plt.legend()

#plt.savefig('../images/OOD-detection_PSDs.png', dpi=300, bbox_inches='tight')

# Displaying the plot
plt.show()
```
Unfortunately, we observe a significant amount of overlap between OOD data and high T-shirt probability. Furthermore, the blue line doesn't seem to decrease much as you move from 0.9 to 1, suggesting that even a very high threshold is likely to lead to OOD contamination (while also tossing out a significant portion of ID data).

For pants, the problem is much less severe. It looks like a low threshold (on this T-shirt probability scale) can separate nearly all OOD samples from being pants.

## Setting a threshold
Let's put our observations to the test and produce a confusion matrix that includes ID-pants, ID-Tshirts, and OOD class labels. We'll start with a high threshold of 0.9 to see how that performs.
```python
def softmax_thresh_classifications(probs, threshold):
    """
    Classifies data points into categories based on softmax probabilities and a specified threshold.

    Parameters:
    - probs: np.array
        A 2D array of shape (n_samples, n_classes) containing the softmax probabilities for each sample 
        across all classes. Each row corresponds to a single sample, and each column corresponds to 
        the probability of a specific class.
    - threshold: float
        A probability threshold for classification. Samples are classified into a specific class if 
        their corresponding probability exceeds the threshold.

    Returns:
    - classifications: np.array
        A 1D array of shape (n_samples,) where:
        - 1 indicates the sample is classified as the second class (e.g., "pants").
        - 0 indicates the sample is classified as the first class (e.g., "shirts").
        - -1 indicates the sample is classified as out-of-distribution (OOD) because no class probability
          exceeds the threshold.

    Notes:
    - The function assumes binary classification with probabilities for two classes provided in the `probs` array.
    - If neither class probability exceeds the threshold, the sample is flagged as OOD with a classification of -1.
    - This approach is suitable for threshold-based OOD detection tasks where probabilities can serve as confidence scores.
    """
    classifications = np.where(probs[:, 1] >= threshold, 1,  # classified as pants
                               np.where(probs[:, 0] >= threshold, 0,  # classified as shirts
                                        -1))  # classified as OOD
    return classifications

```
```python
from sklearn.metrics import precision_recall_fscore_support

# Assuming ood_probs, id_probs, and train_labels are defined
# Threshold values
upper_threshold = 0.9

# Classifying OOD examples (sandals)
ood_classifications = softmax_thresh_classifications(ood_probs, upper_threshold)

# Classifying ID examples (T-shirts and pants)
id_classifications = softmax_thresh_classifications(id_probs, upper_threshold)

# Combine OOD and ID classifications and true labels
all_predictions = np.concatenate([ood_classifications, id_classifications])
all_true_labels = np.concatenate([-1 * np.ones(ood_classifications.shape), test_labels])

all_true_labels # Sandals (-1), T-shirts (0), Trousers (1)

```
```python
# Confusion matrix
cm = confusion_matrix(all_true_labels, all_predictions, labels=[0, 1, -1])

# Plotting the confusion matrix
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Shirt", "Pants", "OOD"])
disp.plot(cmap=plt.cm.Blues)
plt.title('Confusion Matrix for OOD and ID Classification')

plt.show()

# Looking at F1, precision, and recall
precision, recall, f1, _ = precision_recall_fscore_support(all_true_labels, all_predictions, labels=[0, 1], average='macro') # macro = average scores across classes

# ID: T-shirts (0) and Trousers (1)
print(f"F1: {f1}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
```
Even with a high threshold of 0.9, we end up with nearly a couple hundred OOD samples classified as ID. In addition, over 800 ID samples had to be tossed out due to uncertainty.

### Quick exercise
What threhsold is required to ensure that no OOD samples are incorrectly considered as IID? What percentage of ID samples are mistaken as OOD at this threshold?

With a very conservative threshold, we can make sure very few OOD samples are incorrectly classified as ID. However, the flip side is that conservative thresholds tend to incorrectly classify many ID samples as being OOD. In this case, we incorrectly assume almost 20% of shirts are OOD samples.

## Iterative threshold determination

In practice, selecting an appropriate threshold is an iterative process that balances the trade-off between correctly identifying in-distribution (ID) data and accurately flagging out-of-distribution (OOD) data. Here's how you can iteratively determine the threshold:

* **Define evaluation metrics**: While confusion matrices are an excellent tool when you're ready to more closely examine the data, we need a single metric that can summarize threshold performance so we can easily compare across threshold. Common metrics include accuracy, precision, recall, or the F1 score for both ID and OOD detection.

* **Evaluate over a range of thresholds**: Test different threshold values and evaluate the performance on a validation set containing both ID and OOD data.

* **Select the optimal threshold**: Choose the threshold that provides the best balance according to your chosen metrics.

Use the below code to determine what threshold should be set to ensure precision = 100%. What threshold is required for recall to be 100%? What threshold gives the highest F1 score?
```python
def eval_softmax_thresholds(thresholds, ood_probs, id_probs):
    """
    Evaluates the performance of softmax-based classification at various thresholds by calculating precision, 
    recall, and F1 scores for in-distribution (ID) and out-of-distribution (OOD) data.

    Parameters:
    - thresholds: list or np.array
        A list or array of threshold values to evaluate. Each threshold is applied to classify samples based 
        on their softmax probabilities.
    - ood_probs: np.array
        A 2D array of shape (n_ood_samples, n_classes) containing the softmax probabilities for OOD samples 
        across all classes.
    - id_probs: np.array
        A 2D array of shape (n_id_samples, n_classes) containing the softmax probabilities for ID samples 
        across all classes.

    Returns:
    - precisions: list
        A list of precision values computed for each threshold.
    - recalls: list
        A list of recall values computed for each threshold.
    - f1_scores: list
        A list of F1 scores computed for each threshold.

    Notes:
    - The function assumes binary classification for ID classes (e.g., T-shirts and pants) and uses -1 to 
      represent OOD classifications.
    - True labels for ID samples are taken from `test_labels` (defined externally).
    - True labels for OOD samples are set to -1, indicating their OOD nature.
    - Precision, recall, and F1 scores are calculated using macro-averaging, which treats each class equally 
      regardless of the number of samples.

    Example Usage:
    ```
    thresholds = np.linspace(0.5, 1.0, 50)
    precisions, recalls, f1_scores = eval_softmax_thresholds(thresholds, ood_probs, id_probs)
    ```
    """
    # Store evaluation metrics for each threshold
    precisions = []
    recalls = []
    f1_scores = []
    
    for threshold in thresholds:
        # Classifying OOD examples
        ood_classifications = softmax_thresh_classifications(ood_probs, threshold)
        
        # Classifying ID examples
        id_classifications = softmax_thresh_classifications(id_probs, threshold)
        
        # Combine OOD and ID classifications and true labels
        all_predictions = np.concatenate([ood_classifications, id_classifications])
        all_true_labels = np.concatenate([-1 * np.ones(ood_classifications.shape), test_labels])
        
        # Evaluate metrics
        precision, recall, f1, _ = precision_recall_fscore_support(all_true_labels, all_predictions, labels=[0, 1], average='macro')
        precisions.append(precision)
        recalls.append(recall)
        f1_scores.append(f1)
        
    return precisions, recalls, f1_scores

```
```python
# Define thresholds to evaluate
thresholds = np.linspace(.5, 1, 50)

# Evaluate on all thresholds
precisions, recalls, f1_scores = eval_softmax_thresholds(thresholds, ood_probs, id_probs)
```
```python
def plot_metrics_vs_thresholds(thresholds, f1_scores, precisions, recalls, OOD_signal):
    """
    Plots evaluation metrics (Precision, Recall, and F1 Score) as functions of threshold values for 
    softmax-based or energy-based OOD detection, and identifies the best threshold for each metric.

    Parameters:
    - thresholds: list or np.array
        A list or array of threshold values used for classification.
    - f1_scores: list or np.array
        A list or array of F1 scores computed at each threshold.
    - precisions: list or np.array
        A list or array of precision values computed at each threshold.
    - recalls: list or np.array
        A list or array of recall values computed at each threshold.
    - OOD_signal: str
        A descriptive label for the signal being used for OOD detection, such as "Softmax" or "Energy".

    Returns:
    - fig: matplotlib.figure.Figure
        The figure object containing the plot.
    - best_f1_threshold: float
        The threshold value corresponding to the highest F1 score.
    - best_precision_threshold: float
        The threshold value corresponding to the highest precision.
    - best_recall_threshold: float
        The threshold value corresponding to the highest recall.

    Notes:
    - The function identifies and highlights the best threshold for each metric (F1 Score, Precision, Recall).
    - It generates a line plot for each metric as a function of the threshold and marks the best thresholds 
      with vertical dashed lines.
    - This visualization is particularly useful for assessing the trade-offs between precision, recall, 
      and F1 score when selecting a classification threshold.

    Example Usage:
    ```
    fig, best_f1, best_precision, best_recall = plot_metrics_vs_thresholds(
        thresholds, f1_scores, precisions, recalls, OOD_signal='Softmax'
    )
    plt.show()
    ```
    """
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

    # Create a new figure
    fig, ax = plt.subplots(figsize=(8, 5))

    # Plot metrics as functions of the threshold
    ax.plot(thresholds, precisions, label='Precision', color='g')
    ax.plot(thresholds, recalls, label='Recall', color='b')
    ax.plot(thresholds, f1_scores, label='F1 Score', color='r')
    
    # Add best threshold indicators
    ax.axvline(x=best_f1_threshold, color='r', linestyle='--', label=f'Best F1 Threshold: {best_f1_threshold:.2f}')
    ax.axvline(x=best_precision_threshold, color='g', linestyle='--', label=f'Best Precision Threshold: {best_precision_threshold:.2f}')
    ax.axvline(x=best_recall_threshold, color='b', linestyle='--', label=f'Best Recall Threshold: {best_recall_threshold:.2f}')
    ax.set_xlabel(f'{OOD_signal} Threshold')
    ax.set_ylabel('Metric Value')
    ax.set_title('Evaluation Metrics as Functions of Threshold')
    ax.legend()

    return fig, best_f1_threshold, best_precision_threshold, best_recall_threshold

```
```python
fig, best_f1_threshold, best_precision_threshold, best_recall_threshold = plot_metrics_vs_thresholds(thresholds, f1_scores, precisions, recalls, 'Softmax')

```
```python
# Threshold values
upper_threshold = best_f1_threshold
# upper_threshold = best_precision_threshold

# Classifying OOD examples (sandals)
ood_classifications = softmax_thresh_classifications(ood_probs, upper_threshold)

# Classifying ID examples (T-shirts and pants)
id_classifications = softmax_thresh_classifications(id_probs, upper_threshold)

# Combine OOD and ID classifications and true labels
all_predictions = np.concatenate([ood_classifications, id_classifications])
all_true_labels = np.concatenate([-1 * np.ones(ood_classifications.shape), test_labels])

# Confusion matrix
cm = confusion_matrix(all_true_labels, all_predictions, labels=[0, 1, -1])

# Plotting the confusion matrix
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Shirt", "Pants", "OOD"])
disp.plot(cmap=plt.cm.Blues)
plt.title('Confusion Matrix for OOD and ID Classification')
plt.show()
```

#### Discuss
How might you use these tools to ensure that a model trained on health data from hospital A will reliably predict new test data from hospital B? 

:::::::::::::::::::::::::::::::::::::::: keypoints

- Softmax-based OOD detection uses the model's output probabilities to identify instances that do not belong to the training distribution.
- Threshold selection is critical and involves trade-offs between retaining in-distribution data and detecting OOD samples.
- Visualizations such as PCA and probability density plots help illustrate how OOD data overlaps with in-distribution data in feature space.
- While simple and widely used, softmax-based methods have limitations, including sensitivity to threshold choices and reduced reliability in high-dimensional settings.
- Understanding softmax-based OOD detection lays the groundwork for exploring more advanced techniques like energy-based detection.

::::::::::::::::::::::::::::::::::::::::::::::::::
