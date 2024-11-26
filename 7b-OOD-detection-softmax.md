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


# Example 1: Softmax scores
Softmax-based methods are among the most widely used techniques for out-of-distribution (OOD) detection, leveraging the probabilistic outputs of a model to differentiate between in-distribution (ID) and OOD data. These methods are inherently tied to models employing a softmax activation function in their final layer, such as logistic regression or neural networks with a classification output layer. Softmax produces a normalized probability distribution across the classes, which can then be thresholded to identify OOD instances.

In this lesson, we will train a logistic regression model to classify images from the Fashion MNIST dataset and explore how its softmax outputs can signal whether a given input belongs to the ID classes (e.g., T-shirts or pants) or is OOD (e.g., sandals). While softmax is most naturally applied in models with a logistic activation, alternative approaches, such as applying softmax-like operations post hoc to models with different architectures, are occasionally used. However, these alternatives are less common and may require additional considerations.

By focusing on logistic regression, we aim to illustrate the fundamental principles of softmax-based OOD detection in a simple and interpretable context before extending these ideas to more complex architectures.


```python
# some settings I'm playing around with when designing this lesson
verbose = False
alpha=0.2
max_iter = 10 # increase after testing phase
n_epochs = 10 # increase after testing phase

```
### Prepare the ID (train and test) and OOD data
* ID = T-shirts/Blouses, Pants
* OOD = any other class. For Illustrative purposes, we'll focus on images of sandals as the OOD class.
```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from keras.datasets import fashion_mnist

def prep_ID_OOD_datasests(ID_class_labels, OOD_class_labels):
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
    
    return fig

```
```python
train_data, test_data, ood_data, train_labels, test_labels, ood_labels = prep_ID_OOD_datasests([0,1], [5])
fig = plot_data_sample(train_data, ood_data)
#fig.savefig('../images/OOD-detection_image-data-preview.png', dpi=300, bbox_inches='tight')
plt.show()


```
![Preview of image dataset](https://raw.githubusercontent.com/carpentries-incubator/fair-explainable-ml/main/images/OOD-detection_image-data-preview.png)

## Visualizing OOD and ID data

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
scatter1 = plt.scatter(train_data_pca[train_labels == 0, 0], train_data_pca[train_labels == 0, 1], c='blue', label='T-shirt/top (ID)', alpha=0.5)
scatter2 = plt.scatter(train_data_pca[train_labels == 1, 0], train_data_pca[train_labels == 1, 1], c='red', label='Pants (ID)', alpha=0.5)
scatter3 = plt.scatter(ood_data_pca[:, 0], ood_data_pca[:, 1], c='green', label='Sandals (OOD)', edgecolor='k')

# Create a single legend for all classes
plt.legend(handles=[scatter1, scatter2, scatter3], loc="upper right")
plt.xlabel('First Principal Component')
plt.ylabel('Second Principal Component')
plt.title('PCA of In-Distribution and OOD Data')
#plt.savefig('../images/OOD-detection_PCA-image-dataset.png', dpi=300, bbox_inches='tight')
plt.show()
```
![PCA visualization](https://raw.githubusercontent.com/carpentries-incubator/fair-explainable-ml/main/images/OOD-detection_PCA-image-dataset.png)
From this plot, we see that sandals are more likely to be confused as T-shirts than pants. It also may be surprising to see that these data clouds overlap so much given their semantic differences. Why might this be?

* **Over-reliance on linear relationships**: Part of this has to do with the fact that we're only looking at linear relationships and treating each pixel as its own input feature, which is usually never a great idea when working with image data. In our next example, we'll switch to the more modern approach of CNNs.
* **Semantic gap != feature gap**: Another factor of note is that images that have a wide semantic gap may not necessarily translate to a wide gap in terms of the data's visual features (e.g., ankle boots and bags might both be small, have leather, and have zippers). Part of an effective OOD detection scheme involves thinking carefully about what sorts of data contanimations may be observed by the model, and assessing how similar these contaminations may be to your desired class labels.
## Train and evaluate model on ID data
```python
# Train a logistic regression classifier
model = LogisticRegression(max_iter=max_iter, solver='lbfgs', multi_class='multinomial').fit(train_data_flat, train_labels)
```
Before we worry about the impact of OOD data, let's first verify that we have a reasonably accurate model for the ID data.
```python
# Evaluate the model on in-distribution data
in_dist_preds = model.predict(test_data_flat)
in_dist_accuracy = accuracy_score(test_labels, in_dist_preds)
print(f'In-Distribution Accuracy: {in_dist_accuracy:.2f}')
```
```python
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay

# Generate and display confusion matrix
cm = confusion_matrix(test_labels, in_dist_preds, labels=[0, 1])
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['T-shirt/top', 'Pants'])
disp.plot(cmap=plt.cm.Blues)
#plt.savefig('../images/OOD-detection_ID-confusion-matrix.png', dpi=300, bbox_inches='tight')
plt.show()
```
![ID confusion matrix](https://raw.githubusercontent.com/carpentries-incubator/fair-explainable-ml/main/images/OOD-detection_ID-confusion-matrix.png)

## How does our model view OOD data?

A basic question we can start with is to ask, on average, how are OOD samples classified? Are they more likely to be Tshirts or pants? For this kind of question, we can calculate the probability scores for the OOD data, and compare this to the ID data.
```python
# Predict probabilities using the model on OOD data (Sandals)
ood_probs = model.predict_proba(ood_data_flat)
avg_ood_prob = np.mean(ood_probs, 0)
print(f"Avg. probability of sandal being T-shirt: {avg_ood_prob[0]:.4f}")
print(f"Avg. probability of sandal being pants: {avg_ood_prob[1]:.4f}")

id_probs = model.predict_proba(train_data_flat)
id_probs_shirts = id_probs[train_labels==0,:]
id_probs_pants = id_probs[train_labels==1,:]
avg_tshirt_prob = np.mean(id_probs_shirts, 0)
avg_pants_prob = np.mean(id_probs_pants, 0)

print()
print(f"Avg. probability of T-shirt being T-shirt: {avg_tshirt_prob[0]:.4f}")
print(f"Avg. probability of pants being pants: {avg_pants_prob[1]:.4f}")
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
#plt.savefig('../images/OOD-detection_histograms.png', dpi=300, bbox_inches='tight')
# Displaying the plot
plt.show()

```
![Histograms of ID oand OOD data](https://raw.githubusercontent.com/carpentries-incubator/fair-explainable-ml/main/images/OOD-detection_histograms.png)
Alternatively, for a better comparison across all three classes, we can use a probability density plot. This will allow for an easier comparison when the counts across classes lie on vastly different sclaes (i.e., max of 35 vs max of 5000).
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
plt.legend()

#plt.savefig('../images/OOD-detection_PSDs.png', dpi=300, bbox_inches='tight')

# Displaying the plot
plt.show()

```
![Probability densities](https://raw.githubusercontent.com/carpentries-incubator/fair-explainable-ml/main/images/OOD-detection_PSDs.png)
Unfortunately, we observe a significant amount of overlap between OOD data and high T-shirt probability. Furthermore, the blue line doesn't seem to decrease much as you move from 0.9 to 1, suggesting that even a very high threshold is likely to lead to OOD contamination (while also tossing out a significant portion of ID data).

For pants, the problem is much less severe. It looks like a low threshold (on this T-shirt probability scale) can separate nearly all OOD samples from being pants.

### Setting a threshold
Let's put our observations to the test and produce a confusion matrix that includes ID-pants, ID-Tshirts, and OOD class labels. We'll start with a high threshold of 0.9 to see how that performs.
```python
def softmax_thresh_classifications(probs, threshold):
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
all_true_labels = np.concatenate([-1 * np.ones(ood_classifications.shape), train_labels])

# Confusion matrix
cm = confusion_matrix(all_true_labels, all_predictions, labels=[0, 1, -1])

# Plotting the confusion matrix
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Shirt", "Pants", "OOD"])
disp.plot(cmap=plt.cm.Blues)
plt.title('Confusion Matrix for OOD and ID Classification')

#plt.savefig('../images/OOD-detection_ID-OOD-confusion-matrix1.png', dpi=300, bbox_inches='tight')

plt.show()

# Looking at F1, precision, and recall
precision, recall, f1, _ = precision_recall_fscore_support(all_true_labels, all_predictions, labels=[0, 1], average='macro') # discuss macro vs micro .

print(f"F1: {f1}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")

```
![Probability densities](https://raw.githubusercontent.com/carpentries-incubator/fair-explainable-ml/main/images/OOD-detection_ID-OOD-confusion-matrix1.png)
Even with a high threshold of 0.9, we end up with nearly a couple hundred OOD samples classified as ID. In addition, over 800 ID samples had to be tossed out due to uncertainty.

### Quick exercise
What threhsold is required to ensure that no OOD samples are incorrectly considered as IID? What percentage of ID samples are mistaken as OOD at this threshold? Answer:  0.9999, (3826+2414)/(3826+2414+2174+3586)=52%

With a very conservative threshold, we can make sure very few OOD samples are incorrectly classified as ID. However, the flip side is that conservative thresholds tend to incorrectly classify many ID samples as being OOD. In this case, we incorrectly assume almost 20% of shirts are OOD samples.

## Iterative Threshold Determination

In practice, selecting an appropriate threshold is an iterative process that balances the trade-off between correctly identifying in-distribution (ID) data and accurately flagging out-of-distribution (OOD) data. Here's how you can iteratively determine the threshold:

* **Define Evaluation Metrics**: While confusion matrices are an excellent tool when you're ready to more closely examine the data, we need a single metric that can summarize threshold performance so we can easily compare across threshold. Common metrics include accuracy, precision, recall, or the F1 score for both ID and OOD detection.

* **Evaluate Over a Range of Thresholds**: Test different threshold values and evaluate the performance on a validation set containing both ID and OOD data.

* **Select the Optimal Threshold**: Choose the threshold that provides the best balance according to your chosen metrics.

Use the below code to determine what threshold should be set to ensure precision = 100%. What threshold is required for recall to be 100%? What threshold gives the highest F1 score?

### Callout on averaging schemes
F1 scores can be calculated per class, and then averaged in different ways (macro, micro, or weighted) when dealing with multiclass or multilabel classification problems. Here are the key types of averaging methods:

* Macro-Averaging: Calculates the F1 score for each class independently and then takes the average of these scores. This treats all classes equally, regardless of their support (number of true instances for each class).

* Micro-Averaging: Aggregates the contributions of all classes to compute the average F1 score. This is typically used for imbalanced datasets as it gives more weight to classes with more instances.

* Weighted-Averaging: Calculates the F1 score for each class independently and then takes the average, weighted by the number of true instances for each class. This accounts for class imbalance by giving more weight to classes with more instances.

### Callout on including OOD data in F1 calculation

```python
# from sklearn.metrics import precision_recall_fscore_support, accuracy_score

def eval_softmax_thresholds(thresholds, ood_probs, id_probs):
    # Store evaluation metrics for each threshold
    precisions = []
    recalls = []
    f1_scores = []
    
    for threshold in thresholds:
        # Classifying OOD examples (sandals)
        ood_classifications = softmax_thresh_classifications(ood_probs, threshold)
        
        # Classifying ID examples (T-shirts and pants)
        id_classifications = softmax_thresh_classifications(id_probs, threshold)
        
        # Combine OOD and ID classifications and true labels
        all_predictions = np.concatenate([ood_classifications, id_classifications])
        all_true_labels = np.concatenate([-1 * np.ones(ood_classifications.shape), train_labels])
        
        # Evaluate metrics
        precision, recall, f1, _ = precision_recall_fscore_support(all_true_labels, all_predictions, labels=[0, 1], average='macro') # discuss macro vs micro .
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
    fig, ax = plt.subplots(figsize=(12, 8))

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
#fig.savefig('../images/OOD-detection_metrics_vs_softmax-thresholds.png', dpi=300, bbox_inches='tight')

```
![OOD-detection_metrics_vs_softmax-thresholds](https://raw.githubusercontent.com/carpentries-incubator/fair-explainable-ml/main/images/OOD-detection_metrics_vs_softmax-thresholds.png)
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
all_true_labels = np.concatenate([-1 * np.ones(ood_classifications.shape), train_labels])

# Confusion matrix
cm = confusion_matrix(all_true_labels, all_predictions, labels=[0, 1, -1])

# Plotting the confusion matrix
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Shirt", "Pants", "OOD"])
disp.plot(cmap=plt.cm.Blues)
plt.title('Confusion Matrix for OOD and ID Classification')
#plt.savefig('../images/OOD-detection_ID-OOD-confusion-matrix2.png', dpi=300, bbox_inches='tight')
plt.show()

```
![Optimized threshold confusion matrix](https://raw.githubusercontent.com/carpentries-incubator/fair-explainable-ml/main/images/OOD-detection_ID-OOD-confusion-matrix2.png)

:::::::::::::::::::::::::::::::::::::::: keypoints

- Softmax-based OOD detection uses the model's output probabilities to identify instances that do not belong to the training distribution.
- Threshold selection is critical and involves trade-offs between retaining in-distribution data and detecting OOD samples.
- Visualizations such as PCA and probability density plots help illustrate how OOD data overlaps with in-distribution data in feature space.
- While simple and widely used, softmax-based methods have limitations, including sensitivity to threshold choices and reduced reliability in high-dimensional settings.
- Understanding softmax-based OOD detection lays the groundwork for exploring more advanced techniques like energy-based detection.

::::::::::::::::::::::::::::::::::::::::::::::::::
