---
title: "OOD detection: distance-based"
teaching: 40
exercises: 0
---

:::::::::::::::::::::::::::::::::::::::: questions
- How do distance-based methods like Mahalanobis distance and KNN work for OOD detection?
- What is contrastive learning and how does it improve feature representations?
- How does contrastive learning enhance the effectiveness of distance-based OOD detection methods?
::::::::::::::::::::::::::::::::::::::::::::::::::
::::::::::::::::::::::::::::::::::::::: objectives
- Gain a thorough understanding of distance-based OOD detection methods, including Mahalanobis distance and KNN.
- Learn the principles of contrastive learning and its role in improving feature representations.
- Explore the synergy between contrastive learning and distance-based OOD detection methods to enhance detection performance.
::::::::::::::::::::::::::::::::::::::::::::::::::
### Introduction
Distance-based Out-of-Distribution (OOD) detection relies on measuring the proximity of a data point to the training data's feature space. Unlike threshold-based methods such as softmax or energy, distance-based approaches compute the similarity of the feature representation of an input to the known classes' clusters. 

### Advantages
- **Class-agnostic**: Can detect OOD data regardless of the class label.
- **Highly interpretable**: Uses well-defined mathematical distances like Euclidean or Mahalanobis.

### Disadvantages
- **Requires feature extraction**: Needs a model that produces meaningful embeddings.
- **Computationally intensive**: Calculating distances can be expensive, especially with high-dimensional embeddings.

We will use the Mahalanobis distance as the core metric in this notebook.
### Mahalanobis Distance
The Mahalanobis distance measures the distance of a point from a distribution, accounting for the variance and correlations of the data:

$$
D_M(x) = \sqrt{(x - \mu)^T \Sigma^{-1} (x - \mu)}
$$
where:

- x: The input data point.
- \(mu\): The mean vector of the distribution.
- Sigma: The covariance matrix of the distribution. The inverse of the covariance matrix is used to "whiten" the feature space, ensuring that features with larger variances do not dominate the distance computation. This adjustment also accounts for correlations between features, transforming the data into a space where all features are uncorrelated and standardized.
This approach is robust for high-dimensional data as it accounts for correlations between features.

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

train_data, test_data, ood_data, train_labels, test_labels, ood_labels = prep_ID_OOD_datasests([0,1], [5]) #list(range(2,10)) use remaining 8 classes in dataset as OOD
fig = plot_data_sample(train_data, ood_data)
plt.show()
```
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
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False)
ood_loader = torch.utils.data.DataLoader(ood_dataset, batch_size=64, shuffle=False)
```
### Define CNN class
Next, we'll define a simple Convolutional Neural Network (CNN) to classify in-distribution (ID) data. This CNN will serve as the backbone for our experiments, enabling us to analyze its predictions on both ID and OOD data. The model will include convolutional layers for feature extraction and fully connected layers for classification.
```python
import torch.nn as nn
import torch.nn.functional as F

# Define a simple CNN model
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        # First convolutional layer:
        # Input channels = 1 (grayscale images), output channels = 32, kernel size = 3x3
        # Output size after conv1: (32, H-2, W-2) due to 3x3 kernel (reduces spatial dimensions by 2 in each direction)
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3)

        # Second convolutional layer:
        # Input channels = 32, output channels = 64, kernel size = 3x3
        # Output size after conv2: (64, H-4, W-4) due to two 3x3 kernels
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)

        # Fully connected layer (penultimate layer):
        # Input size = 64 * 5 * 5, output size = 30
        # 5x5 is derived from input image size (28x28) reduced by two 3x3 kernels and two 2x2 max-pooling operations
        self.fc1 = nn.Linear(64 * 5 * 5, 30)

        # Final fully connected layer (classification layer):
        # Input size = 128 (penultimate layer output), output size = 2 (binary classification)
        self.fc2 = nn.Linear(30, 2)

    def forward(self, x):
        """
        Defines the forward pass of the model.
        
        Parameters:
        - x: Input tensor of shape (batch_size, channels, height, width), e.g., (64, 1, 28, 28) for grayscale images.
        
        Returns:
        - logits: Output tensor of shape (batch_size, num_classes), e.g., (64, 2).
        """

        # Apply first convolutional layer followed by ReLU and 2x2 max-pooling
        # Input size: (batch_size, 1, 28, 28)
        # Output size after conv1: (batch_size, 32, 26, 26)
        # Output size after max-pooling: (batch_size, 32, 13, 13)
        x = F.relu(F.max_pool2d(self.conv1(x), 2))

        # Apply second convolutional layer followed by ReLU and 2x2 max-pooling
        # Input size: (batch_size, 32, 13, 13)
        # Output size after conv2: (batch_size, 64, 11, 11)
        # Output size after max-pooling: (batch_size, 64, 5, 5)
        x = F.relu(F.max_pool2d(self.conv2(x), 2))

        # Flatten the tensor for the fully connected layers
        # Input size: (batch_size, 64, 5, 5)
        # Output size after flattening: (batch_size, 64*5*5)
        x = x.view(-1, 64 * 5 * 5)

        # Apply the first fully connected layer (penultimate layer) with ReLU
        # Input size: (batch_size, 64*5*5)
        # Output size: (batch_size, 128)
        x = F.relu(self.fc1(x))

        # Apply the final fully connected layer (classification layer)
        # Input size: (batch_size, 128)
        # Output size: (batch_size, 2)
        logits = self.fc2(x)

        return logits

    def extract_penultimate(self, x):
        """
        Extracts embeddings from the penultimate layer of the model.
        
        Parameters:
        - x: Input tensor of shape (batch_size, channels, height, width), e.g., (64, 1, 28, 28).
        
        Returns:
        - embeddings: Output tensor from the penultimate layer of shape (batch_size, 128).
        """

        # Apply convolutional layers and max-pooling (same as in forward)
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = x.view(-1, 64 * 5 * 5)

        # Stop at the penultimate layer (fc1) and return the output
        embeddings = F.relu(self.fc1(x))
        return embeddings

```
```python
device = torch.device('cpu')
model = SimpleCNN().to(device)

```
### Train CNN
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
import torch.optim as optim

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

train_model(model, train_loader, criterion, optimizer)
```
```python
def extract_features(model, dataloader, device):
    """
    Extracts embeddings from the penultimate layer of the model.

    Parameters:
    - model: The trained PyTorch model.
    - dataloader: DataLoader providing the dataset.
    - device: Torch device (e.g., 'cpu' or 'cuda').

    Returns:
    - features: NumPy array of embeddings from the penultimate layer.
    - labels: NumPy array of corresponding labels.
    """
    model.eval()
    features, labels = [], []
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs = inputs.to(device)
            embeddings = model.extract_penultimate(inputs)  # Extract embeddings
            # embeddings = model(inputs)  # Extract embeddings from output neurons (N= number of classes; limited feature representation)

            features.append(embeddings.cpu().numpy())
            labels.append(targets.cpu().numpy())

    # Combine features and labels into arrays
    features = np.vstack(features)
    labels = np.concatenate(labels)

    # Report shape as a sanity check
    print(f"Extracted features shape: {features.shape}")
    return features, labels

```
```python
import numpy as np
# from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# Compute Mahalanobis distance
def compute_mahalanobis_distance(features, mean, covariance):
    inv_covariance = np.linalg.inv(covariance)
    distances = []
    for x in features:
        diff = x - mean
        distance = np.sqrt(np.dot(np.dot(diff, inv_covariance), diff.T))
        distances.append(distance)
    return np.array(distances)

```
```python
# Calculate mean and covariance of ID features
id_features, id_labels = extract_features(model, train_loader, device)
mean = np.mean(id_features, axis=0)

# from sklearn.covariance import EmpiricalCovariance
# covariance = EmpiricalCovariance().fit(id_features).covariance_

from sklearn.covariance import LedoitWolf
# Use a shrinkage estimator for covariance
covariance = LedoitWolf().fit(id_features).covariance_

# Compute Mahalanobis distances for ID and OOD data
ood_features, _ = extract_features(model, ood_loader, device)
id_distances = compute_mahalanobis_distance(id_features, mean, covariance)
ood_distances = compute_mahalanobis_distance(ood_features, mean, covariance)

```
```python
# Visualize Mahalanobis distances
plt.hist(id_distances, bins=50, alpha=0.5, label='ID')
plt.hist(ood_distances, bins=50, alpha=0.5, label='OOD')
plt.xlabel('Mahalanobis Distance')
plt.ylabel('Frequency')
plt.legend()
plt.title('Mahalanobis Distances for ID and OOD Data')
plt.show()
```
### Discussion: Overlapping Mahalanobis distance distributions
After plotting the Mahalanobis distances for in-distribution (ID) and out-of-distribution (OOD) data, we may observe some overlap between the two distributions. This overlap reveals one of the limitations of distance-based methods: **the separability of ID and OOD data is highly dependent on the quality of the feature representations**. The model's learned features might not adequately distinguish between ID and OOD data, especially when OOD samples share semantic or structural similarities with ID data.

### A solution? Contrastive learning
In classical training regimes, models are trained with a *limited worldview*. They learn to distinguish between pre-defined classes based only on the data they’ve seen during training, and simply don't know what they don't know.

An analogy: consider a child learning to identify animals based on a set of flashcards with pictures of cats, dogs, and birds. If you show them a picture of a fox or a turtle, they might struggle because their understanding is constrained by the categories they’ve been explicitly taught. This is analogous to the way models trained with supervised learning approach classification—they build decision boundaries tailored to the training classes but struggle with new, unseen data.

Now, consider teaching the child differently. Instead of focusing solely on identifying "cat" or "dog," you teach them to group animals by broader characteristics—like furry vs. scaly or walking vs. swimming. This approach helps the child form a more generalized understanding of the world, enabling them to recognize new animals by connecting them to familiar patterns. Contrastive learning aims to achieve something similar for machine learning models.

**Contrastive learning** creates feature spaces that are less dependent on specific classes and more attuned to broader semantic relationships. By learning to pull similar data points closer in feature space and push dissimilar ones apart, contrastive methods generate representations that are robust to shifts in data and can naturally cluster unseen categories. This makes contrastive learning particularly promising for improving OOD detection, as it helps models generalize beyond their training distribution.

Unlike traditional training methods that rely heavily on explicit class labels, contrastive learning optimizes the feature space itself, encouraging the model to group similar data points together and push dissimilar ones apart. For example:

- Positive pairs (e.g., augmented views of the same image) are encouraged to be close in the feature space.
- Negative pairs (e.g., different images or samples from different distributions) are pushed apart.

This results in a *feature space with semantic clusters*, where data points with similar meanings are grouped together, even across unseen distributions.

#### Challenges and trade-offs
- **Training complexity:** Contrastive learning requires large amounts of diverse data and careful design of augmentations or sampling strategies.
- **Unsupervised nature:** While contrastive learning does not rely on explicit labels, defining meaningful positive and negative pairs is non-trivial.

### Concluding thoughts and future directions

While contrastive learning provides an exciting opportunity to improve OOD detection, it represents a shift from the traditional threshold- or distance-based approaches we have discussed so far. By learning a feature space that is inherently more generalizable and robust, contrastive learning offers a promising solution to the challenges posed by overlapping Mahalanobis distance distributions.

If you're interested, we can explore specific contrastive learning methods like **SimCLR** or **MoCo** in future sessions, diving into how their objectives help create robust feature spaces!
