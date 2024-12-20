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

Energy-based OOD detection offers a modern and robust alternative. This "output-based" approach leverages the **energy score**, a scalar value derived from a model's output logits, to measure the compatibility between input data and the model's learned distribution. 

### Understanding energy scores

To understand energy-based OOD detection, we start by defining the **energy function E(x)**, which measures how "compatible" an input x is with a model's learned distribution.

### 1. Energy function
For a given input x and output logits f(x) — the raw outputs of a neural network — the energy of x is defined as:  

$$
E(x) = -\log \left( \sum_{k} \exp(f_k(x)) \right)
$$

where:  
- f_k(x) is the logit corresponding to class k,  
- The sum is taken over all classes k.  

This equation compresses the logits into a single scalar value: the energy score.  

- Lower energy E(x) reflects **higher compatitibility**,  
- Higher energy E(x) reflects **lower compatitibility**.

### 2. Energy to probability
Using the Gibbs distribution, the energy can be converted into a probability that reflects how likely x is under the model's learned distribution. The relationship is:  

$$
P(x) \propto \exp(-E(x))
$$

Here:  
- Lower energy \( E(x) \) leads to a **higher probability**,  
- Higher energy \( E(x) \) leads to a **lower probability**.  

The exponential relationship ensures that even small differences in energy values translate to significant changes in probability. 

If your stakeholders or downstream tasks require interpretable confidence scores, a Gibbs-based probability might make the thresholding process more understandable and adaptable. However, the raw energy scores can be more sensitive to OOD data since they do not compress their values between 0 and 1.

### 3. Why energy works better than softmax

Softmax probabilities are computed as:  

$$
P(y = k \mid x) = \frac{\exp(f_k(x))}{ \sum_{j} \exp(f_j(x))}
$$

The softmax function normalizes the logits \( f(x) \), squeezing the output into a range between 0 and 1. While this is useful for interpreting the model’s predictions as probabilities, it introduces **overconfidence** for OOD inputs. Specifically:

- Even when none of the logits \( f_k(x) \) are strongly aligned with any class (e.g., low magnitudes for all logits), softmax still distributes the probabilities across the known classes.
- The normalization ensures the total probability sums to 1, which can mask the uncertainty by making the scores appear confident for OOD inputs.

Energy-based methods, on the other hand, do not normalize the logits into probabilities by default. Instead, the **energy score** summarizes the raw logits as:  

$$
E(x) = -\log \sum_{j} \exp(f_j(x))
$$

#### Key difference: sensitivity to logits / no normalization

- **Softmax**: The output probabilities are dominated by the largest logit relative to the others, even if all logits are small. This can produce overconfident predictions for OOD data because the softmax function distributes probabilities across known classes.
- **Energy**: By summarizing the raw logits directly, energy scores provide a more nuanced view of the model’s uncertainty, without forcing outputs into an overconfident probability distribution.

### Summary
- Energy E(x) directly measures compatibility with the model.  
- Lower energy → Higher compatibility (in-distribution),  
- Higher energy → Lower compatibility (OOD data).  
- The exponential relationship ensures sensitivity to even small deviations, making energy-based detection more robust than softmax-based methods.

## Worked example: comparing softmax and energy
In this hands-on example, we'll repeat the same investigation as before with a couple of adjustments:

- Use CNN to train model
- Compare both softmax and energy scores with respect to ID and OOD data. We can do this easily using the PyTorch-OOD library.
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
```python
train_data, test_data, ood_data, train_labels, test_labels, ood_labels = prep_ID_OOD_datasests([0,1], [5]) #list(range(2,10)) use remaining 8 classes in dataset as OOD
fig = plot_data_sample(train_data, ood_data)
plt.show()
```
## Visualizing OOD and ID data


### UMAP (or similar)

Recall in our previous example, we used PCA to visualize the ID and OOD data distributions. This was appropriate given that we were evaluating OOD/ID data in the context of a linear model. However, when working with nonlinear models such as CNNs, it makes more sense to investigate how the data is represented in a nonlinear space. Nonlinear embedding methods, such as Uniform Manifold Approximation and Projection (UMAP), are more suitable in such scenarios. 

UMAP  is a non-linear dimensionality reduction technique that preserves both the global structure and the local neighborhood relationships in the data. UMAP is often better at maintaining the continuity of data points that lie on non-linear manifolds. It can reveal nonlinear patterns and structures that PCA might miss, making it a valuable tool for analyzing ID and OOD distributions.
```python
plot_umap = True 
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
    umap_alpha = .1

    # Plotting UMAP components
    plt.figure(figsize=(10, 6))
    
    # Plot in-distribution data
    scatter1 = plt.scatter(umap_in_dist[train_labels == 0, 0], umap_in_dist[train_labels == 0, 1], c='blue', label='T-shirts (ID)', alpha=umap_alpha)
    scatter2 = plt.scatter(umap_in_dist[train_labels == 1, 0], umap_in_dist[train_labels == 1, 1], c='red', label='Trousers (ID)', alpha=umap_alpha)
    
    # Plot OOD data
    scatter3 = plt.scatter(umap_ood[:, 0], umap_ood[:, 1], c='green', label='OOD', edgecolor='k', alpha=umap_alpha)
    
    # Create a single legend for all classes
    plt.legend(handles=[scatter1, scatter2, scatter3], loc="upper right")
    plt.xlabel('First UMAP Component')
    plt.ylabel('Second UMAP Component')
    plt.title('UMAP of In-Distribution and OOD Data')
    plt.show()
```
With UMAP, we see our data clusters into more meaningful groups (compared to PCA). Our nonlinear model should hopefully have no problem separating these three clusters.

### Preparing data for CNN
Next, we'll prepare our data for a pytorch (torch) CNN. 
```python
import torch

# Convert to PyTorch tensors and normalize
train_data_tensor = torch.tensor(train_data, dtype=torch.float32).unsqueeze(1) / 255.0
test_data_tensor = torch.tensor(test_data, dtype=torch.float32).unsqueeze(1) / 255.0
ood_data_tensor = torch.tensor(ood_data, dtype=torch.float32).unsqueeze(1) / 255.0
train_labels_tensor = torch.tensor(train_labels, dtype=torch.long)
test_labels_tensor = torch.tensor(test_labels, dtype=torch.long)

# TensorDataset provides a convenient way to couple input data with their corresponding labels, making it easier to pass them into a DataLoader.
train_dataset = torch.utils.data.TensorDataset(train_data_tensor, train_labels_tensor)
test_dataset = torch.utils.data.TensorDataset(test_data_tensor, test_labels_tensor)
ood_dataset = torch.utils.data.TensorDataset(ood_data_tensor, torch.zeros(ood_data_tensor.shape[0], dtype=torch.long))

# DataLoader is used to efficiently load and manage batches of data
# - It provides iterators over the data for training/testing.
# - Supports options like batch size, shuffling, and parallel data loading
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=False)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False)
ood_loader = torch.utils.data.DataLoader(ood_dataset, batch_size=64, shuffle=False)
```
### Define CNN class
Next, we'll define a simple Convolutional Neural Network (CNN) to classify in-distribution (ID) data. This CNN will serve as the backbone for our experiments, enabling us to analyze its predictions on both ID and OOD data. The model will include convolutional layers for feature extraction and fully connected layers for classification.
```python
import torch.nn as nn  # Import the PyTorch module for building neural networks
import torch.nn.functional as F  # Import functional API for activation and pooling
import torch.optim as optim  # Import optimizers for training the model

# Define a simple CNN model
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        # First convolutional layer
        # Input: 1 channel (e.g., grayscale image), Output: 32 feature maps
        # Kernel size: 3x3 (sliding window over the image)
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3) 
        
        # Second convolutional layer
        # Input: 32 feature maps from conv1, Output: 64 feature maps
        # Kernel size: 3x3
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)

        # Fully connected layer 1 (fc1)
        # Input: Flattened feature maps after two conv+pool layers
        # Output: 128 features
        # Dimensions explained:
        #   After two Conv2d layers with kernel_size=3 and MaxPool2d (2x2):
        #   Input image size: (28x28) -> After conv1: (26x26) -> After pool1: (13x13)
        #   -> After conv2: (11x11) -> After pool2: (5x5)
        #   Total features to flatten: 64 (feature maps) * 5 * 5 (spatial size)
        self.fc1 = nn.Linear(64 * 5 * 5, 128) 
        
        # Fully connected layer 2 (fc2)
        # Input: 128 features from fc1
        # Output: 2 classes (binary classification)
        self.fc2 = nn.Linear(128, 2)

    def forward(self, x):
        # Pass input through first convolutional layer
        # Activation: ReLU (introduces non-linearity)
        # Pooling: MaxPool2d with kernel size 2x2 (reduces spatial dimensions by half)
        x = F.relu(F.max_pool2d(self.conv1(x), 2))  # Output size: (N, 32, 13, 13)
        
        # Pass through second convolutional layer
        x = F.relu(F.max_pool2d(self.conv2(x), 2))  # Output size: (N, 64, 5, 5)
        
        # Flatten the feature maps for the fully connected layer
        # x.view reshapes the tensor to (batch_size, flattened_size)
        x = x.view(-1, 64 * 5 * 5)  # Output size: (N, 1600)
        
        # Pass through first fully connected layer with ReLU activation
        x = F.relu(self.fc1(x))  # Output size: (N, 128)
        
        # Final fully connected layer for classification
        x = self.fc2(x)  # Output size: (N, 2)
        return x


```
```python
device = torch.device('cpu')
model = SimpleCNN().to(device)

```
### Train model
```python
def train_model(model, train_loader, criterion, optimizer, epochs=5):
    """
    Trains a given PyTorch model using a specified dataset, loss function, and optimizer.

    Parameters:
    - model (nn.Module): The neural network model to train.
    - train_loader (DataLoader): DataLoader object providing the training dataset in batches.
    - criterion (nn.Module): Loss function used for optimization (e.g., CrossEntropyLoss).
    - optimizer (torch.optim.Optimizer): Optimizer for adjusting model weights (e.g., Adam, SGD).
    - epochs (int): Number of training iterations over the entire dataset.

    Returns:
    - None: Prints the loss for each epoch during training.

    Workflow:
    1. Iterate over the dataset for the given number of epochs.
    2. For each batch, forward propagate inputs, compute the loss, and backpropagate gradients.
    3. Update model weights using the optimizer and reset gradients after each step.
    """
    model.train()  # Set the model to training mode
    for epoch in range(epochs):
        running_loss = 0.0
        for inputs, labels in train_loader:
            # Move inputs and labels to the appropriate device (CPU or GPU)
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Reset gradients from the previous step to avoid accumulation
            optimizer.zero_grad()
            
            # Forward pass: Compute model predictions
            outputs = model(inputs)
            
            # Compute the loss between predictions and true labels
            loss = criterion(outputs, labels)
            
            # Backward pass: Compute gradients of the loss w.r.t. model parameters
            loss.backward()
            
            # Update model weights using gradients and optimizer rules
            optimizer.step()
            
            # Accumulate the batch loss for reporting
            running_loss += loss.item()
        
        # Print the average loss for the current epoch
        print(f'Epoch {epoch+1}, Loss: {running_loss/len(train_loader)}')

```
```python
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

train_model(model, train_loader, criterion, optimizer)
```
### Evaluate the model
```python
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import torch
import numpy as np

# Function to plot confusion matrix
def plot_confusion_matrix(labels, predictions, title):
    """
    Plots a confusion matrix for a classification task.

    Parameters:
    - labels (array-like): True labels for the dataset.
    - predictions (array-like): Model-predicted labels.
    - title (str): Title for the confusion matrix plot.

    Returns:
    - None: Displays the confusion matrix plot.
    """
    # Compute the confusion matrix
    cm = confusion_matrix(labels, predictions, labels=[0, 1])
    
    # Create a display object for the confusion matrix
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["T-shirt/top", "Trouser"])
    
    # Plot the confusion matrix with a color map
    disp.plot(cmap=plt.cm.Blues)
    plt.title(title)
    plt.show()

```

```python
# Function to evaluate the model on a given dataset
def evaluate_model(model, dataloader, device):
    """
    Evaluates a PyTorch model on a given dataset.

    Parameters:
    - model (torch.nn.Module): The trained PyTorch model to evaluate.
    - dataloader (torch.utils.data.DataLoader): DataLoader object providing the dataset in batches.
    - device (torch.device): Device on which to perform the evaluation (CPU or GPU).

    Returns:
    - all_labels (np.array): True labels for the entire dataset.
    - all_predictions (np.array): Model predictions for the entire dataset.
    """
    model.eval()  # Set the model to evaluation mode
    all_labels = []  # To store true labels
    all_predictions = []  # To store model predictions
    
    # Disable gradient computation during evaluation
    with torch.no_grad():
        for inputs, labels in dataloader:
            # Move inputs and labels to the specified device
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Forward pass to get model outputs
            outputs = model(inputs)
            
            # Get predicted class labels (index with the highest probability)
            _, preds = torch.max(outputs, 1)
            
            # Append true labels and predictions to the lists
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(preds.cpu().numpy())
    
    # Convert lists to NumPy arrays for easier processing
    return np.array(all_labels), np.array(all_predictions)

```
```python
# Evaluate the model on the test dataset
test_labels, test_predictions = evaluate_model(model, test_loader, device)

# Plot confusion matrix for test dataset
plot_confusion_matrix(test_labels, test_predictions, "Confusion Matrix for Test Data")
```
#### Comparing softmax vs energy scores
Let's take a look at both the softmax and energy scores generated by both the ID test set and the OOD data we extracted earlier.

With PyTorch-OOD, we can easily calculate both measures.
```python
# 1. Computing softmax scores
from pytorch_ood.detector import MaxSoftmax

# Initialize the softmax-based OOD detector
softmax_detector = MaxSoftmax(model)

# Compute softmax scores
def get_OOD_scores(detector, dataloader):
    """
    Computes softmax-based scores for a given OOD detector and dataset.

    Parameters:
    - detector: An initialized OOD detector (e.g., MaxSoftmax).
    - dataloader: DataLoader providing the dataset for which scores are to be computed.

    Returns:
    - scores: A NumPy array of softmax scores for all data points.
    """
    scores = []
    detector.model.eval()  # Ensure the model is in evaluation mode
    with torch.no_grad():  # Disable gradient computation for efficiency
        for inputs, _ in dataloader:
            inputs = inputs.to(device)  # Move inputs to the correct device
            score = detector.predict(inputs)  # Get the max softmax score
            scores.extend(score.cpu().numpy())  # Move scores to CPU and convert to NumPy array
    return np.array(scores)

# Compute softmax scores for ID and OOD data
id_softmax_scores = get_OOD_scores(softmax_detector, test_loader)
ood_softmax_scores = get_OOD_scores(softmax_detector, ood_loader)

id_softmax_scores # values are negative to align with other OOD measures, such as energy (more negative is better)
```
```python
# 2. Computing energy
from pytorch_ood.detector import EnergyBased

# Initialize the energy-based OOD detector
energy_detector = EnergyBased(model)

id_energy_scores = get_OOD_scores(energy_detector, test_loader)
ood_energy_scores = get_OOD_scores(energy_detector, ood_loader)
id_energy_scores
```
### Plot probability densities
```python
from scipy.stats import gaussian_kde
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
plot_psd(id_softmax_scores, ood_softmax_scores, 'Softmax')

# Plot PSD for energy scores
plot_psd(id_energy_scores, ood_energy_scores, 'Energy')
```
## Recap and limitations
The energy-based approach for out-of-distribution (OOD) detection has several strengths, particularly its ability to effectively separate in-distribution (ID) and OOD data by leveraging the raw logits of a model. However, it is not without limitations. Here are the key drawbacks:

1. Dependence on well-defined classes: Energy scores rely on logits that correspond to clearly defined classes in the model. If the model's logits are not well-calibrated or if the task involves ambiguous or overlapping classes, the energy scores may not provide reliable OOD separation.
2. Energy thresholds tuned on one dataset may not generalize well to other datasets or domains (depending on how expansive/variable your OOD calibration set is)

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
