---
title: "Explainability methods: linear probe"
teaching: 0
exercises: 0
---
:::::::::::::::::::::::::::::::::::::: questions 

- TODO

::::::::::::::::::::::::::::::::::::::::::::::::

::::::::::::::::::::::::::::::::::::: objectives

- TODO

::::::::::::::::::::::::::::::::::::::::::::::::

```python
# Let's start by importing the necessary libraries.

import os
import torch
import logging
import numpy as np
from typing import Tuple
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from datasets import load_dataset, Dataset
from transformers import AutoModel, AutoTokenizer, AutoConfig

logging.basicConfig(level=logging.INFO)
os.environ['TOKENIZERS_PARALLELISM'] = 'false'  # This is needed to avoid a warning from huggingface
```
Now, let's set the random seed to ensure reproducibility. Setting random seeds is like setting a starting point for your machine learning adventure. It ensures that every time you train your model, it starts from the same place, using the same random numbers, making your results consistent and comparable.
```python
# Set random seeds for reproducibility - pick any number of your choice to set the seed. We use 42, since that is the answer to everything, after all.
torch.manual_seed(42)
```
```python
# Set the GPU to use
os.environ['CUDA_VISIBLE_DEVICES'] = '0'  
```
##### Loading the Dataset
Let's load our data: the IMDB Movie Review dataset. The dataset contains text reviews and their corresponding sentiment labels (positive or negative). 
The label 1 corresponds to a positive review, and 0 corresponds to a negative review.
```python
def load_imdb_dataset(keep_samples: int = 100) -> Tuple[Dataset, Dataset, Dataset]:
    '''
    Load the IMDB dataset from huggingface.
    The dataset contains text reviews and their corresponding sentiment labels (positive or negative).
    The label 1 corresponds to a positive review, and 0 corresponds to a negative review.
    :param keep_samples: Number of samples to keep, for faster training.
    :return: train, dev, test datasets. Each can be treated as a dictionary with keys 'text' and 'label'.
    '''
    dataset = load_dataset('imdb')

    # Keep only a subset of the data for faster training
    train_dataset = Dataset.from_dict(dataset['train'].shuffle(seed=42)[:keep_samples])
    dev_dataset = Dataset.from_dict(dataset['test'].shuffle(seed=42)[:keep_samples])
    test_dataset = Dataset.from_dict(dataset['test'].shuffle(seed=42)[keep_samples:2*keep_samples])

    # train_dataset[0] will return {'text': ...., 'label': 0}
    logging.info(f'Loaded IMDB dataset: {len(train_dataset)} training samples, {len(dev_dataset)} dev samples, {len(test_dataset)} test samples.')
    return train_dataset, dev_dataset, test_dataset
```
```python
train_dataset, dev_dataset, test_dataset = load_imdb_dataset(keep_samples=50)
```
##### Loading the Model

We will load a model from huggingface, and use this model to get the embeddings for the probe.
We use BERT for this example, but feel free to explore other models from huggingface after the exercise.

BERT is a transformer-based model, and is known to perform well on a variety of NLP tasks.
The model is pre-trained on a large corpus of text, and can be fine-tuned for specific tasks.
```python
def load_model(model_name: str) -> Tuple[AutoModel, AutoTokenizer]:
    '''
    Load a model from huggingface.
    :param model_name: Check huggingface for acceptable model names.
    :return: Model and tokenizer.
    '''
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    config = AutoConfig.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name, config=config)
    model.config.max_position_embeddings = 128  # Reducing from default 512 to 128 for computational efficiency

    logging.info(f'Loaded model and tokenizer: {model_name} with {model.config.num_hidden_layers} layers, '
                 f'hidden size {model.config.hidden_size} and sequence length {model.config.max_position_embeddings}.')
    return model, tokenizer
```
```python
# To play around with other models, find a list of models and their model_ids at: https://huggingface.co/models
model, tokenizer = load_model('bert-base-uncased')
```
Let's see what the model's architecture looks like. How many layers does it have?
```python
print(model)
```
Let's see if your answer matches the actual number of layers in the model.
```python
num_layers = model.config.num_hidden_layers
print(f'The model has {num_layers} layers.')
```
##### Setting up the Probe
Before we define the probing classifier or probe, let's set up some utility functions the probe will use. 
The probe will be trained from hidden representations from a specific layer of the BERT model. The `get_embeddings_from_model` function will retrieve the intermediate layer representations (also known as embeddings) from a user defined layer number.

The `visualize_embeddings` method can be used to see what these high dimensional hidden embeddings would look like when converted into a 2D view. The visualization is not intended to be informative in itself, and is only an additional tool used to get a sense of what the inputs to the probing classifier may look like. 
```python
def get_embeddings_from_model(model: AutoModel, tokenizer: AutoTokenizer, layer_num: int, data: list[str]) -> torch.Tensor:
    '''
    Get the embeddings from a model.
    :param model: The model to use. This is needed to get the embeddings.
    :param tokenizer: The tokenizer to use. This is needed to convert the data to input IDs.
    :param layer_num: The layer to get embeddings from. 0 is the input embeddings, and the last layer is the output embeddings.
    :param data: The data to get embeddings for. A list of strings.
    :return: The embeddings. Shape is N, L, D, where N is the number of samples, L is the length of the sequence, and D is the dimensionality of the embeddings.
    '''
    logging.info(f'Getting embeddings from layer {layer_num} for {len(data)} samples...')

    # Batch the data for computational efficiency
    batch_size = 32
    batch_num = 1
    for i in range(0, len(data), batch_size):
        batch = data[i:i+batch_size]
        logging.info(f'Getting embeddings for batch {batch_num}...')
        batch_num += 1

        # Tokenize the batch of data
        inputs = tokenizer(batch, return_tensors='pt', padding=True, truncation=True)

        # Get the embeddings from the model
        outputs = model(**inputs, output_hidden_states=True)

        # Get the embeddings for the specific the layer
        embeddings = outputs.hidden_states[layer_num]

        # Concatenate the embeddings from each batch
        if i == 0:
            all_embeddings = embeddings
        else:
            all_embeddings = torch.cat([all_embeddings, embeddings], dim=0)

    logging.info(f'Got embeddings for {len(data)} samples from layer {layer_num}. Shape: {all_embeddings.shape}')
    return all_embeddings
```

```python
def visualize_embeddings(embeddings: torch.Tensor, labels: list, layer_num: int, save_plot: bool = False) -> None:
    '''
    Visualize the embeddings using t-SNE.
    :param embeddings: The embeddings to visualize. Shape is N, L, D, where N is the number of samples, L is the length of the sequence, and D is the dimensionality of the embeddings.
    :param labels: The labels for the embeddings. A list of integers.
    :return: None
    '''

    # Since we are working with sentiment analysis, which is sentence based task, we can use sentence embeddings.
    # The sentence embeddings are simply the mean of the token embeddings of that sentence.
    sentence_embeddings = torch.mean(embeddings, dim=1)  # N, D

    # Convert to numpy
    sentence_embeddings = sentence_embeddings.detach().numpy()
    labels = np.array(labels)

    # Visualize the embeddings using t-SNE
    tsne = TSNE(n_components=2, random_state=0)
    embeddings_2d = tsne.fit_transform(sentence_embeddings)

    negative_points = embeddings_2d[labels == 0]
    positive_points = embeddings_2d[labels == 1]

    # Plot the embeddings. We want to colour the datapoints by label.
    fig, ax = plt.subplots()
    ax.scatter(negative_points[:, 0], negative_points[:, 1], label='Negative', color='red', marker='o', s=10, alpha=0.7)
    ax.scatter(positive_points[:, 0], positive_points[:, 1], label='Positive', color='blue', marker='o', s=10, alpha=0.7)
    plt.xlabel('t-SNE dimension 1')
    plt.ylabel('t-SNE dimension 2')
    plt.title(f't-SNE of Sentence Embeddings - Layer{layer_num}')
    plt.legend()

    # Save the plot if needed, then display it
    if save_plot:
        plt.savefig(f'tsne_layer_{layer_num}.png')
    plt.show()

    logging.info('Visualized embeddings using t-SNE.')

```
Now, it's finally time to define our probe! We set this up as a class, where the probe itself is an object of this class. 
The class also contains methods used to train and evaluate the probe. 

Read through this code block in a bit more detail - from this whole exercise, this part provides you with the most useful takeaways on ways to define and train neural networks!
```python
class Probe():
    def __init__(self, hidden_dim: int = 768, class_size: int = 2)  -> None:
        '''
        Initialize the probe.
        :param hidden_dim: The dimensionality of the hidden layer of the probe.
        :param num_layers: The number of layers in the probe.
        :return: None
        '''

        # The probe is a simple linear classifier, with a hidden layer and an output layer.
        # The input to the probe is the embeddings from the model, and the output is the predicted class.

        # Exercise: Try playing around with the hidden_dim and num_layers to see how it affects the probe's performance.
        # But watch out: if a complex probe performs well on the task, we don't know if the performance
        # is because of the model embeddings, or the probe itself learning the task!

        self.probe = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, class_size),

            # Add more layers here if needed

            # Sigmoid is used to convert the hidden states into a probability distribution over the classes
            torch.nn.Sigmoid()
        )


    def train(self, data_embeddings: torch.Tensor, labels: torch.Tensor, num_epochs: int = 10,
              learning_rate: float = 0.001, batch_size: int = 32) -> None:
        '''
        Train the probe on the embeddings of data from the model.
        :param data_embeddings: A tensor of shape N, L, D, where N is the number of samples, L is the length of the sequence, and D is the dimensionality of the embeddings.
        :param labels: A tensor of shape N, where N is the number of samples. Each element is the label for the corresponding sample.
        :param num_epochs: The number of epochs to train the probe for. An epoch is one pass through the entire dataset.
        :param learning_rate: How fast the probe learns. A hyperparameter.
        :param batch_size: Used to batch the data for computational efficiency. A hyperparameter.
        :return:
        '''

        # Setup the loss function (training objective) for the training process.
        # The cross-entropy loss is used for multi-class classification, and represents the negative log likelihood of the true class.
        criterion = torch.nn.CrossEntropyLoss()

        # Setup the optimization algorithm to update the probe's parameters during training.
        # The Adam optimizer is an extension to stochastic gradient descent, and is a popular choice.
        optimizer = torch.optim.Adam(self.probe.parameters(), lr=learning_rate)

        # Train the probe
        logging.info('Training the probe...')
        for epoch in range(num_epochs):  # Pass over the data num_epochs times

            for i in range(0, len(data_embeddings), batch_size):

                # Iterate through one batch of data at a time
                batch_embeddings = data_embeddings[i:i+batch_size].detach()
                batch_labels = labels[i:i+batch_size]

                # Convert to sentence embeddings, since we are performing a sentence classification task
                batch_embeddings = torch.mean(batch_embeddings, dim=1)  # N, D

                # Get the probe's predictions, given the embeddings from the model
                outputs = self.probe(batch_embeddings)

                # Calculate the loss of the predictions, against the true labels
                loss = criterion(outputs, batch_labels)

                # Backward pass - update the probe's parameters
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        logging.info('Trained the probe.')


    def predict(self, data_embeddings: torch.Tensor, batch_size: int = 32) -> torch.Tensor:
        '''
        Get the probe's predictions on the embeddings from the model, for unseen data.
        :param data_embeddings: A tensor of shape N, L, D, where N is the number of samples, L is the length of the sequence, and D is the dimensionality of the embeddings.
        :param batch_size: Used to batch the data for computational efficiency.
        :return: A tensor of shape N, where N is the number of samples. Each element is the predicted class for the corresponding sample.
        '''

        # Iterate through batches
        for i in range(0, len(data_embeddings), batch_size):

            # Iterate through one batch of data at a time
            batch_embeddings = data_embeddings[i:i+batch_size]

            # Get the probe's predictions
            outputs = self.probe(batch_embeddings)

            # Get the predicted class for each sample
            _, predicted = torch.max(outputs, 1)

            # Concatenate the predictions from each batch
            if i == 0:
                all_predicted = predicted
            else:
                all_predicted = torch.cat([all_predicted, predicted], dim=0)

        return all_predicted


    def evaluate(self, data_embeddings: torch.tensor, labels: torch.tensor, batch_size: int = 32) -> float:
        '''
        Evaluate the probe's performance by testing it on unseen data.
        :param data_embeddings: A tensor of shape N, L, D, where N is the number of samples, L is the length of the sequence, and D is the dimensionality of the embeddings.
        :param labels: A tensor of shape N, where N is the number of samples. Each element is the label for the corresponding sample.
        :return: The accuracy of the probe on the unseen data.
        '''

        # Iterate through batches
        for i in range(0, len(data_embeddings), batch_size):

            # Iterate through one batch of data at a time
            batch_embeddings = data_embeddings[i:i+batch_size]
            batch_labels = labels[i:i+batch_size]

            # Convert to sentence embeddings, since we are performing a sentence classification task
            batch_embeddings = torch.mean(batch_embeddings, dim=1)  # N, D

            # Get the probe's predictions
            with torch.no_grad():
                outputs = self.probe(batch_embeddings)

            # Get the predicted class for each sample
            _, predicted = torch.max(outputs, dim=-1)

            # Concatenate the predictions from each batch
            if i == 0:
                all_predicted = predicted
                all_labels = batch_labels
            else:
                all_predicted = torch.cat([all_predicted, predicted], dim=0)
                all_labels = torch.cat([all_labels, batch_labels], dim=0)

        # Calculate the accuracy of the probe
        correct = (all_predicted == all_labels).sum().item()
        accuracy = correct / all_labels.shape[0]
        logging.info(f'Probe accuracy: {accuracy:.2f}')

        return accuracy
```
```python
# Initialize the probing classifier (or probe)
probe = Probe()
```
##### Analysing the model using Probes

Time to start evaluating the model using our probing tool! Let's see which layer has most information about sentiment analysis on IMDB.
For this, we will train the probe on embeddings from each layer of the model, and see which layer performs the best on the dev set.
```python
layer_wise_accuracies = []
best_probe, best_layer, best_accuracy = None, -1, 0

for layer_num in range(num_layers):
    logging.info(f'\n\nEvaluating representations of layer {layer_num+1}...')

    train_embeddings = get_embeddings_from_model(model, tokenizer, layer_num=layer_num, data=train_dataset['text'])
    dev_embeddings = get_embeddings_from_model(model, tokenizer, layer_num=layer_num, data=dev_dataset['text'])
    train_labels, dev_labels = torch.tensor(train_dataset['label'],  dtype=torch.long), torch.tensor(dev_dataset['label'],  dtype=torch.long)

    # Before training the probe, let's visualize the embeddings using t-SNE.
    # If the layer has information about sentiment analysis, would we see some structure in the embeddings?
    # Compare plots from layers where the probe does poorly, with ones where it does well. What do you notice?
    visualize_embeddings(embeddings=train_embeddings, labels=train_dataset['label'], layer_num=layer_num, save_plot=False)

    # Now, let's train the probe on the embeddings from the model.
    # Feel free to play around with the training hyperparameters, and see what works best for your probe.
    probe = Probe()
    probe.train(data_embeddings=train_embeddings, labels=train_labels,
                num_epochs=5, learning_rate=0.001, batch_size=32)

    # Let's see how well our probe does on a held out dev set
    accuracy = probe.evaluate(data_embeddings=dev_embeddings, labels=dev_labels)
    layer_wise_accuracies.append(accuracy)

    # Keep track of the best probe
    if accuracy > best_accuracy:
        best_probe, best_layer, best_accuracy = probe, layer_num, accuracy
```
```python
# Seeing a list of accuracies can be hard to interpret. Let's plot the layer-wise accuracies to see which layer is best.
plt.plot(layer_wise_accuracies)
plt.xlabel('Layer')
plt.ylabel('Accuracy')
plt.title('Probe Accuracy by Layer')
plt.grid(alpha=0.3)
plt.show()
```
Which layer has the best accuracy? What does this tell us about the model?

Let's go ahead and stress test this. Is the best layer able to predict sentiment for sentences outside the IMDB dataset?

For answering this question, you are the test set! Try to think of challenging sequences for which the model may not be able to predict sentiment.
```python
test_sequences = ['Your sentence here', 'Here is another sentence']
embeddings = get_embeddings_from_model(model=model, tokenizer=tokenizer, layer_num=best_layer, data=test_sequences)
preds = probe.predict(data_embeddings=embeddings)
predictions = ['Positive' if pred == 1 else 'Negative' for pred in preds]
print(f'Predictions for test sequences: {predictions}')
```