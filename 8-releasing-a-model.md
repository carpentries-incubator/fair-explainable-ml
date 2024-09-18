---
title: "Documenting and releasing a model"
teaching: 0
exercises: 0
---

:::::::::::::::::::::::::::::::::::::: questions 

- Why is model sharing important in the context of reproducibility and responsible use?
- What are the challenges, risks, and ethical considerations related to sharing models?
- How can model-sharing best practices be applied using tools like model cards and the Hugging Face platform?

::::::::::::::::::::::::::::::::::::::::::::::::

::::::::::::::::::::::::::::::::::::: objectives

- Understand the importance of model sharing and best practices to ensure reproducibility and responsible use of models.
- Understand the challenges, risks, and ethical concerns associated with model sharing.
- Apply model-sharing best practices through using model cards and the Hugging Face platform.

::::::::::::::::::::::::::::::::::::::::::::::::

::::::::::::::::::::::::::::::::::::: keypoints 

- Model cards are the standard technique for communicating information about how machine learning systems were trained and how they should and should not be used.
- Models can be shared and reused via the Hugging Face platform.

::::::::::::::::::::::::::::::::::::::::::::::::

:::::::::::::::::::::::::::::::::::::: challenge

### Why should we share trained models?
Discuss in small groups and report out: *Why do you believe it is or isn’t important to share ML models? How has model-sharing contributed to your experiences or projects?*

::::::::::::::::::::::::::::::::::::::::::::::::::

:::::::::::::: solution

### Solution

* **Accelerating research**: Sharing models allows researchers and practitioners to build upon existing work, accelerating the pace of innovation in the field.
* **Knowledge exchange**: Model sharing promotes knowledge exchange and collaboration within the machine learning community, fostering a culture of open science.
* **Reproducibility**: Sharing models, along with associated code and data, enhances reproducibility, enabling others to validate and verify the results reported in research papers.
* **Benchmarking**: Shared models serve as benchmarks for comparing new models and algorithms, facilitating the evaluation and improvement of state-of-the-art techniques.
* **Education / Accessibility to state-of-the-art architectures**: Shared models provide valuable resources for educational purposes, allowing students and learners to explore and experiment with advanced machine learning techniques.
* **Repurpose (transfer learning and finetuning)**: Some models (i.e., foundation models) can be repurposed for a wide variety of tasks. This is especially useful when working with limited data.
Data scarcity
* **Resource efficiency**: Instead of training a model from the ground up, practitioners can use existing models as a starting point, saving time, computational resources, and energy.

:::::::::::::::::::::::::




:::::::::::::::::::::::::::::::::::::: challenge

### Challenges and risks of model sharing
Discuss in small groups and report out: *What are some potential challenges, risks, or ethical concerns associated with model sharing and reproducing ML workflows?*

::::::::::::::::::::::::::::::::::::::::::::::::::

:::::::::::::: solution

### Solution
* **Privacy concerns**: Sharing models that were trained on sensitive or private data raises privacy concerns. The potential disclosure of personal information through the model poses a risk to individuals and can lead to unintended consequences.
* **Informed consent**: If models involve user data, ensuring informed consent is crucial. Sharing models trained on user-generated content without clear consent may violate privacy norms and regulations.
* **Data bias and fairness**: Models trained on biased datasets may perpetuate or exacerbate existing biases. Reproducing workflows without addressing bias in the data may result in unfair outcomes, particularly in applications like hiring or criminal justice.
* **Intellectual property**: Models may be developed within organizations with proprietary data and methodologies. Sharing such models without proper consent or authorization may lead to intellectual property disputes and legal consequences.
* **Model robustness and generalization**: Reproduced models may not generalize well to new datasets or real-world scenarios. Failure to account for the limitations of the original model can result in reduced performance and reliability in diverse settings.
* **Lack of reproducibility**: Incomplete documentation, missing details, or changes in dependencies over time can hinder the reproducibility of ML workflows. This lack of reproducibility can impede scientific progress and validation of research findings.
* **Unintended use and misuse**: Shared models may be used in unintended ways, leading to ethical concerns. Developers should consider the potential consequences of misuse, particularly in applications with societal impact, such as healthcare or law enforcement.
* **Responsible AI considerations**: Ethical considerations, such as fairness, accountability, and transparency, should be addressed during model sharing. Failing to consider these aspects can result in models that inadvertently discriminate or lack interpretability. Models used for decision-making, especially in critical areas like healthcare or finance, should be ethically deployed. Transparent documentation and disclosure of how decisions are made are essential for responsible AI adoption.
:::::::::::::::::::::::::


## Saving model locally
Let's review the simplest method for sharing a model first — saving the model locally. When working with PyTorch, it's important to know how to save and load models efficiently. This process ensures that you can pause your work, share your models, or deploy them for inference without having to retrain them from scratch each time.

### Define model
As an example, we'll configure a simple perceptron (single hidden layer) in PyTorch. We'll define a bare bones class for this just so we can initialize the model.

```python
from typing import Dict, Any
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self, config: Dict[str, int]):
        super().__init__()
        # Parameter is a trainable tensor initialized with random values
        self.param = nn.Parameter(torch.rand(config["num_channels"], config["hidden_size"]))
        # Linear layer (fully connected layer) for the output
        self.linear = nn.Linear(config["hidden_size"], config["num_classes"])
        # Store the configuration
        self.config = config

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Forward pass: Add the input to the param tensor, then pass through the linear layer
        return self.linear(x + self.param)
```

Initialize model by calling the class with configuration settings.

```python
# Create model instance with specific configuration
config = {"num_channels": 3, "hidden_size": 32, "num_classes": 10}
model = MyModel(config=config)
```

We can then write a function to save out the model. We'll need both the model weights and the model's configuration (hyperparameter settings). We'll save the configurations as a json since a key/value format is convenient here.

```python
import json

# Function to save model and config locally
def save_model(model: nn.Module, model_path: str, config_path: str) -> None:
    # Save model state dict (weights and biases) as a .pth file
    torch.save(model.state_dict(), model_path) #
    # Save config
    with open(config_path, 'w') as f:
        json.dump(model.config, f)
```

```python
# Save the model and config locally
save_model(model, "my_awesome_model.pth", "my_awesome_model_config.json")

```

To load the model back in, we can write another function

```python
# Function to load model and config locally
def load_model(model_class: Any, model_path: str, config_path: str) -> nn.Module:
    # Load config
    with open(config_path, 'r') as f:
        config = json.load(f)
    # Create model instance with config
    model = model_class(config=config)
    # Load model state dict
    model.load_state_dict(torch.load(model_path))
    return model
```

```python
# Load the model and config locally
loaded_model = load_model(MyModel, "my_awesome_model.pth", "my_awesome_model_config.json")

# Verify the loaded model
print(loaded_model)
```

## Saving a model to Hugging Face
To share your model with a wider audience, we recommend uploading your model to Hugging Face. Hugging Face is a very popular machine learning (ML) platform and community that helps users build, deploy, share, and train machine learning models. It has quickly become the go-to option for sharing models with the public.

### Create a Hugging Face account and access Token
If you haven't completed these steps from the setup, make sure to do this now.

**Create account**: To create an account on Hugging Face, visit: [huggingface.co/join](https://huggingface.co/join). Enter an email address and password, and follow the instructions provided via Hugging Face (you may need to verify your email address) to complete the process.

**Setup access token**: Once you have your account created, you’ll need to generate an access token so that you can upload/share models to your Hugging Face account during the workshop. To generate a token, visit the [Access Tokens setting page](https://huggingface.co/settings/tokens) after logging in. 

### Login to Hugging Face account
To login, you will need to retrieve your access token from the [Access Tokens setting page](https://huggingface.co/settings/tokens)

```python
!huggingface-cli login
```

You might get a message saying you cannot authenticate through git-credential as no helper is defined on your machine. TODO: What does this warning mean?

Once logged in, we will need to edit our model class definition to include Hugging Face's "push_to_hub" attribute. To enable the push_to_hub functionality, you'll need to include the PyTorchModelHubMixin "mixin class" provided by the huggingface_hub library. A mixin class is a type of class used in object-oriented programming to "mix in" additional properties and methods into a class. The PyTorchModelHubMixin class adds methods to your PyTorch model to enable easy saving and loading from the Hugging Face Model Hub.

 Here's how you can adjust the code to incorporate both saving/loading locally and pushing the model to the Hugging Face Hub.

```python
from huggingface_hub import PyTorchModelHubMixin # NEW

class MyModel(nn.Module, PyTorchModelHubMixin): # PyTorchModelHubMixin is new
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        # Initialize layers and parameters
        self.param = nn.Parameter(torch.rand(config["num_channels"], config["hidden_size"]))
        self.linear = nn.Linear(config["hidden_size"], config["num_classes"])
        self.config = config

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x + self.param)
```

```python
# Create model instance with specific configuration
config = {"num_channels": 3, "hidden_size": 32, "num_classes": 10}
model = MyModel(config=config)
print(model)
```

```python
# push to the hub
model.push_to_hub("my-awesome-model", config=config)

```

**Verifying**: To check your work, head back over to your Hugging Face account and click your profile icon in the top-right of the website. Click "Profile" from there to view all of your uploaded models. Alternatively, you can search for your username (or model name) from the [Model Hub](https://huggingface.co/models).

#### Loading the model from Hugging Face

```python
# reload
model = MyModel.from_pretrained("your-username/my-awesome-model")
```

## Uploading transformer models to Hugging Face

Key Differences

* **Saving and Loading the Tokenizer**: Transformer models require a tokenizer that needs to be saved and loaded with the model. This is not necessary for custom PyTorch models that typically do not require a separate tokenizer.
* **Using Pre-trained Classes**: Transformer models use classes like AutoModelForSequenceClassification and AutoTokenizer from the transformers library, which are pre-built and designed for specific tasks (e.g., sequence classification).
* **Methods for Saving and Loading**: The transformers library provides save_pretrained and from_pretrained methods for both models and tokenizers, which handle the serialization and deserialization processes seamlessly.

```python
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# Load a pre-trained model and tokenizer
model_name = "bert-base-uncased"
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Save the model and tokenizer locally
model.save_pretrained("my_transformer_model")
tokenizer.save_pretrained("my_transformer_model")

# Load the model and tokenizer from the saved directory
loaded_model = AutoModelForSequenceClassification.from_pretrained("my_transformer_model")
loaded_tokenizer = AutoTokenizer.from_pretrained("my_transformer_model")

# Verify the loaded model and tokenizer
print(loaded_model)
print(loaded_tokenizer)
```

```python
# Push the model and tokenizer to Hugging Face Hub
model.push_to_hub("my-awesome-transformer-model")
tokenizer.push_to_hub("my-awesome-transformer-model")

# Load the model and tokenizer from the Hugging Face Hub
hub_model = AutoModelForSequenceClassification.from_pretrained("user-name/my-awesome-transformer-model")
hub_tokenizer = AutoTokenizer.from_pretrained("user-name/my-awesome-transformer-model")

# Verify the model and tokenizer loaded from the hub
print(hub_model)
print(hub_tokenizer)
```
:::::::::::::::::::::::::::::::::::::: challenge

### What pieces must be well-documented to ensure reproducible and responsible model sharing?
Discuss in small groups and report out: *What type of information needs to be included in the documentation when sharing a model?*

::::::::::::::::::::::::::::::::::::::::::::::::::

:::::::::::::: solution

### Solution

* Environment setup
* Training data
  * How the data was collected
  * Who owns the data: data license and usage terms
  * Basic descriptive statistics: number of samples, features, classes, etc.
  * Note any class imbalance or general bias issues
  * Description of data distribution to help prevent out-of-distribution failures.
* Preprocessing steps. 
  * Data splitting
  * Standardization method
  * Feature selection
  * Outlier detection and other filters
* Model architecture, hyperparameters and, training procedure (e.g., dropout or early stopping)
* Model weights
* Evaluation metrics. Results and performance. The more tasks/datasets you can evaluate on, the better.
* Ethical considerations:  Include investigations of bias/fairness when applicable (i.e., if your model involves human data or affects decision-making involving humans)
* Contact info
* Acknowledgments
* Examples and demos (highly recommended)
:::::::::::::::::::::::::

:::::::::::::::::::::::::::::::::::::: challenge

### Document your model
TODO

::::::::::::::::::::::::::::::::::::::::::::::::::

:::::::::::::: solution

### Solution
TODO


:::::::::::::::::::::::::
