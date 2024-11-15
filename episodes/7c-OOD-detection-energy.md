---
title: "OOD detection: energy"
teaching: 0
exercises: 0
---

:::::::::::::::::::::::::::::::::::::::: questions

- What are energy-based methods for out-of-distribution (OOD) detection, and how do they compare to softmax-based approaches?
- How does the energy metric enhance separability between in-distribution and OOD data?
- What are the challenges and limitations of energy-based OOD detection methods?

::::::::::::::::::::::::::::::::::::::::::::::::::

::::::::::::::::::::::::::::::::::::::: objectives

- Understand the concept of energy-based OOD detection and its theoretical foundations.
- Compare energy-based methods to softmax-based approaches, highlighting their strengths and limitations.
- Learn how to implement energy-based OOD detection using tools like PyTorch-OOD.
- Explore challenges in applying energy-based methods, including threshold tuning and generalizability to diverse OOD scenarios.

::::::::::::::::::::::::::::::::::::::::::::::::::


# Example 2: Energy-Based OOD Detection

Traditional approaches, such as softmax-based methods, rely on output probabilities to flag OOD data. While simple and intuitive, these methods often struggle to distinguish OOD data effectively in complex scenarios, especially in high-dimensional spaces.

Energy-based OOD detection offers a modern and robust alternative. This "output-based" approach leverages the **energy score**, a scalar value derived from a model's output logits, to measure the compatibility between input data and the model's learned distribution. The energy score directly ties to the Gibbs distribution, which allows for a nuanced interpretation of data compatibility. By computing energy scores and interpreting them probabilistically, energy-based methods enhance separability between in-distribution (ID) and OOD data. These techniques, particularly effective with neural networks, address some of the key limitations of softmax-based approaches.

In this episode, we will explore the theoretical foundations of energy-based OOD detection, implement it using the PyTorch-OOD library, and compare its performance to softmax-based methods. Along the way, we will provide intuitive explanations of key concepts like the Gibbs distribution to ensure accessibility for ML practitioners. We will also discuss the challenges of energy-based methods, including threshold tuning and generalization to diverse OOD scenarios. This understanding will set the stage for hybrid and training-time regularization methods discussed in future episodes.

:::::::::::::::::::::::::::::::::::::::: callout

### intuition behind the Gibbs distribution and energy scores

The Gibbs distribution is a probability distribution used to model systems in equilibrium, and it connects naturally to the concept of energy. In the context of machine learning:

- **Energy as a compatibility measure**: Think of energy as a score that measures how "compatible" a data point is with the model's learned distribution. A **lower energy score** indicates higher compatibility (i.e., the model sees the input as likely to belong to the training distribution).
  
- **From energy to probability**: The Gibbs distribution converts energy into probabilities. For an input \(x\), the probability of observing \(x\) is proportional to \(e^{-\text{Energy}(x)}\). This exponential relationship means that even small differences in energy can lead to significant changes in probability, making energy scores sensitive and effective for OOD detection.

- **Why energy works for OOD detection**: OOD data often results in higher energy scores because the model's output logits are less aligned with any known class. By setting a threshold on energy scores, we can distinguish OOD data from ID data more effectively than with softmax probabilities alone.

For practitioners, the Gibbs distribution provides a bridge between abstract "energy" values and interpretable probabilities, helping us reason about uncertainty and compatibility in a model's predictions.
   * E(x, y) = energy value
   
   * if x and y are "compatitble", lower energy
   
   * Energy can be turned into probability through Gibbs distribution
       * looks at integral over all possible y's
   
   * With energy scores, ID and OOD distributions become much more separable

**Learn more**: Liu et al., Energy-based Out-of-distribution Detection, NeurIPS 2020; https://arxiv.org/pdf/2010.03759

::::::::::::::::::::::::::::::::::::::::::::::::::


## Introducing PyTorch OOD
The PyTorch-OOD library provides methods for OOD detection and other closely related fields, such as anomoly detection or novelty detection. Visit the docs to learn more: [pytorch-ood.readthedocs.io/en/latest/info.html](https://pytorch-ood.readthedocs.io/en/latest/info.html)

This library will provide a streamlined way to calculate both energy and softmax scores from a trained model. 
### Setup example
In this example, we will train a CNN model on the FashionMNIST dataset. We will then repeat a similar process as we did with softmax scores to evaluate how well the energy metric can separate ID and OOD data. 

We'll start by fresh by loading our data again. This time, let's treat all remaining classes in the MNIST fashion dataset as OOD. This should yield a more robust model that is more reliable when presented with all kinds of data. 
```python
train_data, test_data, ood_data, train_labels, test_labels, ood_labels = prep_ID_OOD_datasests([0,1], list(range(2,10))) # use remaining 8 classes in dataset as OOD
fig = plot_data_sample(train_data, ood_data)
fig.savefig('../images/OOD-detection_image-data-preview.png', dpi=300, bbox_inches='tight')
plt.show()
```

## Visualizing OOD and ID data


### UMAP (or similar)

Recall in our previous example, we used PCA to visualize the ID and OOD data distributions. This was appropriate given that we were evaluating OOD/ID data in the context of a linear model. However, when working with nonlinear models such as CNNs, it makes more sense to investigate how the data is represented in a nonlinear space. Nonlinear embedding methods, such as Uniform Manifold Approximation and Projection (UMAP), are more suitable in such scenarios. 

UMAP  is a non-linear dimensionality reduction technique that preserves both the global structure and the local neighborhood relationships in the data. UMAP is often better at maintaining the continuity of data points that lie on non-linear manifolds. It can reveal nonlinear patterns and structures that PCA might miss, making it a valuable tool for analyzing ID and OOD distributions.
```python
plot_umap = True # leave off for now to save time testing downstream materials
if plot_umap:
    import umap
    # Flatten images for PCA and logistic regression
    train_data_flat = train_data.reshape((train_data.shape[0], -1))
    test_data_flat = test_data.reshape((test_data.shape[0], -1))
    ood_data_flat = ood_data.reshape((ood_data.shape[0], -1))
    
    print(f'train_data_flat.shape={train_data_flat.shape}')
    print(f'test_data_flat.shape={test_data_flat.shape}')
    print(f'ood_data_flat.shape={ood_data_flat.shape}')
    
    # Perform UMAP to visualize the data
    umap_reducer = umap.UMAP(n_components=2, random_state=42)
    combined_data = np.vstack([train_data_flat, ood_data_flat])
    combined_labels = np.hstack([train_labels, np.full(ood_data_flat.shape[0], 2)])  # Use 2 for OOD class
    
    umap_results = umap_reducer.fit_transform(combined_data)
    
    # Split the results back into in-distribution and OOD data
    umap_in_dist = umap_results[:len(train_data_flat)]
    umap_ood = umap_results[len(train_data_flat):]

```
The warning message indicates that UMAP has overridden the n_jobs parameter to 1 due to the random_state being set. This behavior ensures reproducibility by using a single job. If you want to avoid the warning and still use parallelism, you can remove the random_state parameter. However, removing random_state will mean that the results might not be reproducible.
```python
if plot_umap:
    umap_alpha = .02

    # Plotting UMAP components
    plt.figure(figsize=(10, 6))
    
    # Plot in-distribution data
    scatter1 = plt.scatter(umap_in_dist[train_labels == 0, 0], umap_in_dist[train_labels == 0, 1], c='blue', label='T-shirts (ID)', alpha=umap_alpha)
    scatter2 = plt.scatter(umap_in_dist[train_labels == 1, 0], umap_in_dist[train_labels == 1, 1], c='red', label='Trousers (ID)', alpha=umap_alpha)
    
    # Plot OOD data
    scatter3 = plt.scatter(umap_ood[:, 0], umap_ood[:, 1], c='green', label='OOD', edgecolor='k', alpha=alpha)
    
    # Create a single legend for all classes
    plt.legend(handles=[scatter1, scatter2, scatter3], loc="upper right")
    plt.xlabel('First UMAP Component')
    plt.ylabel('Second UMAP Component')
    plt.title('UMAP of In-Distribution and OOD Data')
    plt.show()
```
## Train CNN
```python
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torch.nn.functional as F

# Convert to PyTorch tensors and normalize
train_data_tensor = torch.tensor(train_data, dtype=torch.float32).unsqueeze(1) / 255.0
test_data_tensor = torch.tensor(test_data, dtype=torch.float32).unsqueeze(1) / 255.0
ood_data_tensor = torch.tensor(ood_data, dtype=torch.float32).unsqueeze(1) / 255.0

train_labels_tensor = torch.tensor(train_labels, dtype=torch.long)
test_labels_tensor = torch.tensor(test_labels, dtype=torch.long)

train_dataset = torch.utils.data.TensorDataset(train_data_tensor, train_labels_tensor)
test_dataset = torch.utils.data.TensorDataset(test_data_tensor, test_labels_tensor)
ood_dataset = torch.utils.data.TensorDataset(ood_data_tensor, torch.zeros(ood_data_tensor.shape[0], dtype=torch.long))

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False)
ood_loader = torch.utils.data.DataLoader(ood_dataset, batch_size=64, shuffle=False)

# Define a simple CNN model
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.fc1 = nn.Linear(64*5*5, 128)  # Updated this line
        self.fc2 = nn.Linear(128, 2)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = x.view(-1, 64*5*5)  # Updated this line
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = SimpleCNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

def train_model(model, train_loader, criterion, optimizer, epochs=5):
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f'Epoch {epoch+1}, Loss: {running_loss/len(train_loader)}')

train_model(model, train_loader, criterion, optimizer)

```
```python
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Function to plot confusion matrix
def plot_confusion_matrix(labels, predictions, title):
    cm = confusion_matrix(labels, predictions, labels=[0, 1])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["T-shirt/top", "Trouser"])
    disp.plot(cmap=plt.cm.Blues)
    plt.title(title)
    plt.show()

# Function to evaluate model on a dataset
def evaluate_model(model, dataloader, device):
    model.eval()
    all_labels = []
    all_predictions = []
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(preds.cpu().numpy())
    return np.array(all_labels), np.array(all_predictions)

# Evaluate on train data
train_labels, train_predictions = evaluate_model(model, train_loader, device)
plot_confusion_matrix(train_labels, train_predictions, "Confusion Matrix for Train Data")

# Evaluate on test data
test_labels, test_predictions = evaluate_model(model, test_loader, device)
plot_confusion_matrix(test_labels, test_predictions, "Confusion Matrix for Test Data")

# Evaluate on OOD data
ood_labels, ood_predictions = evaluate_model(model, ood_loader, device)
plot_confusion_matrix(ood_labels, ood_predictions, "Confusion Matrix for OOD Data")

```
```python
from scipy.stats import gaussian_kde
from pytorch_ood.detector import EnergyBased
from sklearn.metrics import precision_recall_fscore_support, accuracy_score

# Compute softmax scores
def get_softmax_scores(model, dataloader):
    model.eval()
    softmax_scores = []
    with torch.no_grad():
        for inputs, _ in dataloader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            softmax = torch.nn.functional.softmax(outputs, dim=1)
            softmax_scores.extend(softmax.cpu().numpy())
    return np.array(softmax_scores)

id_softmax_scores = get_softmax_scores(model, test_loader)
ood_softmax_scores = get_softmax_scores(model, ood_loader)

# Initialize the energy-based OOD detector
energy_detector = EnergyBased(model, t=1.0)

# Compute energy scores
def get_energy_scores(detector, dataloader):
    scores = []
    detector.model.eval()
    with torch.no_grad():
        for inputs, _ in dataloader:
            inputs = inputs.to(device)
            score = detector.predict(inputs)
            scores.extend(score.cpu().numpy())
    return np.array(scores)

id_energy_scores = get_energy_scores(energy_detector, test_loader)
ood_energy_scores = get_energy_scores(energy_detector, ood_loader)

import matplotlib.pyplot as plt


# Plot PSDs

# Function to plot PSD
def plot_psd(id_scores, ood_scores, method_name):
    plt.figure(figsize=(12, 6))
    alpha = 0.3

    # Plot PSD for ID scores
    id_density = gaussian_kde(id_scores)
    x_id = np.linspace(id_scores.min(), id_scores.max(), 1000)
    plt.plot(x_id, id_density(x_id), label=f'ID ({method_name})', color='blue', alpha=alpha)

    # Plot PSD for OOD scores
    ood_density = gaussian_kde(ood_scores)
    x_ood = np.linspace(ood_scores.min(), ood_scores.max(), 1000)
    plt.plot(x_ood, ood_density(x_ood), label=f'OOD ({method_name})', color='red', alpha=alpha)

    plt.xlabel('Score')
    plt.ylabel('Density')
    plt.title(f'Probability Density Distributions for {method_name} Scores')
    plt.legend()
    plt.show()

# Plot PSD for softmax scores
plot_psd(id_softmax_scores[:, 1], ood_softmax_scores[:, 1], 'Softmax')

# Plot PSD for energy scores
plot_psd(id_energy_scores, ood_energy_scores, 'Energy')


```
```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, confusion_matrix, ConfusionMatrixDisplay

# Define thresholds to evaluate
thresholds = np.linspace(id_energy_scores.min(), id_energy_scores.max(), 50)

# Store evaluation metrics for each threshold
accuracies = []
precisions = []
recalls = []
f1_scores = []

# True labels for OOD data (since they are not part of the original labels)
ood_true_labels = np.full(len(ood_energy_scores), -1)

# We need the test_labels to be aligned with the ID data
id_true_labels = test_labels[:len(id_energy_scores)]

for threshold in thresholds:
    # Classify OOD examples based on energy scores
    ood_classifications = np.where(ood_energy_scores >= threshold, -1,  # classified as OOD
                                   np.where(ood_energy_scores < threshold, 0, -1))  # classified as ID

    # Classify ID examples based on energy scores
    id_classifications = np.where(id_energy_scores >= threshold, -1,  # classified as OOD
                                  np.where(id_energy_scores < threshold, id_true_labels, -1))  # classified as ID

    # Combine OOD and ID classifications and true labels
    all_predictions = np.concatenate([ood_classifications, id_classifications])
    all_true_labels = np.concatenate([ood_true_labels, id_true_labels])

    # Evaluate metrics
    precision, recall, f1, _ = precision_recall_fscore_support(all_true_labels, all_predictions, labels=[0, 1], average='macro')#, zero_division=0)
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
plt.title('Evaluation Metrics as Functions of Threshold (Energy-Based OOD Detection)')
plt.legend()
plt.show()

```
```python

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, confusion_matrix, ConfusionMatrixDisplay

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_fscore_support, accuracy_score

def evaluate_ood_detection(id_scores, ood_scores, id_true_labels, id_predictions, ood_predictions, score_type='energy'):
    """
    Evaluate OOD detection based on either energy scores or softmax scores.

    Parameters:
    - id_scores: np.array, scores for in-distribution (ID) data
    - ood_scores: np.array, scores for out-of-distribution (OOD) data
    - id_true_labels: np.array, true labels for ID data
    - id_predictions: np.array, predicted labels for ID data
    - ood_predictions: np.array, predicted labels for OOD data
    - score_type: str, type of score used ('energy' or 'softmax')

    Returns:
    - Best thresholds for F1, Precision, and Recall
    - Plots of Precision, Recall, and F1 Score as functions of the threshold
    """
    # Define thresholds to evaluate
    if score_type == 'softmax':
        thresholds = np.linspace(0.5, 1.0, 200)
    else:
        thresholds = np.linspace(id_scores.min(), id_scores.max(), 50)

    # Store evaluation metrics for each threshold
    accuracies = []
    precisions = []
    recalls = []
    f1_scores = []

    # True labels for OOD data (since they are not part of the original labels)
    if score_type == "energy":
        ood_true_labels = np.full(len(ood_scores), -1)
    else:
        ood_true_labels = np.full(len(ood_scores[:,0]), -1)

    for threshold in thresholds:
        # Classify OOD examples based on scores
        if score_type == 'energy':
            ood_classifications = np.where(ood_scores >= threshold, -1, ood_predictions)
            id_classifications = np.where(id_scores >= threshold, -1, id_predictions)
        elif score_type == 'softmax':
            ood_classifications = np.where(ood_scores[:,0] <= threshold, -1, ood_predictions)
            id_classifications = np.where(id_scores[:,0] <= threshold, -1, id_predictions)
        else:
            raise ValueError("Invalid score_type. Use 'energy' or 'softmax'.")

        # Combine OOD and ID classifications and true labels
        all_predictions = np.concatenate([ood_classifications, id_classifications])
        all_true_labels = np.concatenate([ood_true_labels, id_true_labels])

        # Evaluate metrics
        precision, recall, f1, _ = precision_recall_fscore_support(all_true_labels, all_predictions, labels=[-1, 0], average='macro', zero_division=0)
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
    plt.title(f'Evaluation Metrics as Functions of Threshold ({score_type.capitalize()}-Based OOD Detection)')
    plt.legend()
    plt.show()

    # plot confusion matrix
    # Threshold value for the energy score
    upper_threshold = best_f1_threshold  # Using the best F1 threshold from the previous calculation
    if score_type == 'energy':
        # Classifying OOD examples based on energy scores
        ood_classifications = np.where(ood_energy_scores >= upper_threshold, -1,  # classified as OOD
                                  np.where(ood_energy_scores < upper_threshold, 0, -1))  # classified as ID
        # Classifying ID examples based on energy scores
        id_classifications = np.where(id_energy_scores >= upper_threshold, -1,  # classified as OOD
                                  np.where(id_energy_scores < upper_threshold, id_true_labels, -1))  # classified as ID
    elif score_type == 'softmax':
        # Classifying OOD examples based on softmax scores
        ood_classifications = softmax_thresh_classifications(ood_scores, upper_threshold)

        # Classifying ID examples based on softmax scores
        id_classifications = softmax_thresh_classifications(id_scores, upper_threshold)
    # Combine OOD and ID classifications and true labels
    all_predictions = np.concatenate([ood_classifications, id_classifications])
    all_true_labels = np.concatenate([ood_true_labels, id_true_labels])
    # Confusion matrix
    cm = confusion_matrix(all_true_labels, all_predictions, labels=[0, 1, -1])

    # Plotting the confusion matrix
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Shirt", "Pants", "OOD"])
    disp.plot(cmap=plt.cm.Blues)
    plt.title(f'Confusion Matrix for OOD and ID Classification ({score_type.capitalize()}-Based)')
    plt.show()


    return best_f1_threshold, best_precision_threshold, best_recall_threshold

# Example usage
# Assuming id_energy_scores, ood_energy_scores, id_true_labels, and test_labels are already defined
best_f1_threshold, best_precision_threshold, best_recall_threshold = evaluate_ood_detection(id_energy_scores, ood_energy_scores, test_labels, test_predictions, ood_predictions, score_type='energy')
best_f1_threshold, best_precision_threshold, best_recall_threshold = evaluate_ood_detection(id_softmax_scores, ood_softmax_scores, test_labels, test_predictions, ood_predictions, score_type='softmax')

```
## Limitations of our approach thus far

* Focus on single OOD class: More reliable/accurate thresholds can/should be obtained using a wider variety (more classes) and larger sample of OOD data. This is part of the challenge of OOD detection which is that space of OOD data is vast. **Possible exercise**: Redo thresholding using all remaining classes in dataset.

## References and supplemental resources

* https://www.youtube.com/watch?v=hgLC9_9ZCJI
* Generalized Out-of-Distribution Detection: A Survey: https://arxiv.org/abs/2110.11334


:::::::::::::::::::::::::::::::::::::::: keypoints

- Energy-based OOD detection is a modern and more robust alternative to softmax-based methods, leveraging energy scores to improve separability between in-distribution and OOD data.
- By calculating an energy value for each input, these methods provide a more nuanced measure of compatibility between data and the model's learned distribution.
- Non-linear visualizations, like UMAP, offer better insights into how OOD and ID data are represented in high-dimensional feature spaces compared to linear methods like PCA.
- PyTorch-OOD simplifies the implementation of energy-based and other OOD detection methods, making it accessible for real-world applications.
- While energy-based methods excel in many scenarios, challenges include tuning thresholds across diverse OOD classes and ensuring generalizability to unseen distributions.
- Transitioning to energy-based detection lays the groundwork for exploring training-time regularization and hybrid approaches.

::::::::::::::::::::::::::::::::::::::::::::::::::
