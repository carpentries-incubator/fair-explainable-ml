---
title: Setup
---


# Setup
The full workshop setup includes (1) software installation, (2) downloading the data, and (3) setting up a HuggingFace account & access token. If you have any trouble with the steps outlined below, please contact the workshop organizers ASAP to make sure you have everything completed before the workshop starts. 

## Software setup
You will need a terminal (or Git Bash recommended for Windows), Python 3.8+, and the ability to create Python virtual environments. You will also need to install a variety of packages within your virtual environment. 

### 1) Installing Git Bash (Windows only)
We will be launching Jupyter Lab (IDE) from a terminal (Mac/Linux) or Git Bash (Windows) during this workshop. If you will be using a Windows machine for this workshop, please [install Git Bash ("Git for Windows")](https://git-scm.com/downloads/win).

### 2) Installing Python 
To install Python, follow the [Beginner's Guide](https://wiki.python.org/moin/BeginnersGuide/Download) or head straight to the [download page](https://www.python.org/downloads/).
If you encounter problems with the installation procedure, ask your workshop organizers via e-mail for assistance so you are ready to go as soon as the workshop begins.

To verify installation, open terminal / Git Bash and run...

```sh
python3 --version # mac
python --version # windows
```

### 3) Create a workshop folder on your Desktop called "trustworthy_ML"
We'll use this folder to store code throughput the workshop. We'll also add our virtual environment to this folder.

In terminal (Mac/Linux) or Git Bash, create folder using.
```sh
mkdir Desktop/trustworthy_ML # create folder
cd Desktop/trustworthy_ML # change directory to folder
```

### 4) Creating a new virtual environment
We'll install the prerequisite libraries in a virtual environment, to prevent them from cluttering up your Python environment and causing conflicts.

To create a new virtual environment ("venv") for the project, open the terminal (Mac/Linux), Git Bash (Windows), or Anaconda Prompt (Windows), and type one of the below OS-specific options below. 

Make sure you are already CD'd into your workshop folder, `Desktop/trustworthy_ML`. The code below will create a new virtual environment in a folder named `venv/`` in the current working directory. 

```sh
cd Desktop/trustworthy_ML # if you're not already in this folder, CD to it (adjust path, if necesssary)

# Run one of the below options (OS-specific)
python3 -m venv venv # mac/linux
python -m venv venv # windows
```

If you run `ls` (list files), you should see a new `venv/`` folder in your trustworthy_ML folder.
```sh
ls
```

> If you're on Linux and this doesn't work, you may need to install venv first. Try running `sudo apt-get install python3-venv` first, then `python3 -m venv intro_ml`

### 5) Activating the environment
To activate the environment, run the following OS-specific commands in Terminal (Mac/Linux) or Git Bash (Windows):

```sh
source venv/Scripts/activate # Windows + Git Bash
source venv/bin/activate # Mac/Linux
```


### 6) Installing your prerequisites
Once the virtual environment is activated, install the prerequisites by running the following commands:

First, make sure you have the latest version of pip by running:

```sh
python.exe -m pip install --upgrade pip
```

Then, install the required libraries. We've chosen a CPU-only (no GPUs enabled) setup for this lesson to make the environment simpler and more accessible for everyone. By avoiding GPU-specific dependencies like CUDA, we reduce the storage requirements by 3-4 GB and eliminate potential compatibility issues related to GPU hardware.

**Note**: If prompted to Proceed ([y]/n) during environment setup, press y. It may take around 10-20 minutes to complete the full environment setup. Please reach out to the workshop organizers sooner rather than later to fix setup issues prior to the workshop. 

```sh
pip install torch torchvision torchaudio \
            jupyter scikit-learn pandas matplotlib keras tensorflow umap-learn \
            datasets grad-cam pytorch-ood transformers fairlearn "aif360[Reductions]" "aif360[inFairness]"
```

### 7) Adding your virtual environment to JupyterLab
We want Jupyter Lab to have access to the enviornment we just built. To use this virtual environment in JupyterLab, follow these steps:

1. Install the `ipykernel` package:
```sh
pip install ipykernel
```

2. Add the virtual environment as a Jupyter kernel:
```sh
python -m ipykernel install --user --name=venv --display-name "trustworthy_ML"
```

3. When you launch JupyterLab, select the `trustworthy_ML` kernel to ensure your code runs in the correct environment.

### 8) Verify the setup

Change directory to your code folder before launching Jupyter. This will help us keep our code organized in one place.

```sh
cd Desktop/trustworthy_ML
```

To start jupyter lab, open a terminal (Mac/Linux) or Git Bash (Windows) and type the command:

```sh
jupyter lab
```

After launching, start a new notebook using the `trustworthy_ML` kernel to ensure your code runs in the correct environment. Then run the following lines of code:

```python
import torch
import pandas as pd
import sklearn
import jupyter
import tensorflow as tf
import transformers
import pytorch_ood
import fairlearn
import umap

# Tested versions in this workshop:
print("Torch version:", torch.__version__)  # >= 2.5.1
print("Pandas version:", pd.__version__)  # >= 2.2.3
print("Scikit-learn version:", sklearn.__version__)  # >= 1.5.2
print("TensorFlow version:", tf.__version__)  # >= 2.18.0
print("Transformers version:", transformers.__version__)  # >= 4.46.3
print("PyTorch-OOD version:", pytorch_ood.__version__)  # >= 0.2.0
print("Fairlearn version:", fairlearn.__version__)  # >= 0.11.0
print("UMAP version:", umap.__version__)  # >= 0.5.7

```

This should output the versions of all required packages without giving errors.
Most versions should work fine with this lesson, but we've only tested thoroughly with the versions commented above.

### Fallback option: cloud environment
If a local installation does not work for you, it is also possible to run (most of) this lesson in [Google Colab](https://colab.research.google.com/). Some packages may need to be installed on the fly within the notebook (TBD).

### Deactivating/activating environment
To deactivate your virtual environment, simply run `deactivate` in your terminal. If you close the terminal or Git Bash without deactivating, the environment will automatically close as the session ends. Later, you can reactivate the environment using the "Activate environment" instructions above to continue working. If you want to keep coding in the same terminal but no longer need this environment, it’s best to explicitly deactivate it. This ensures that the software installed for this workshop doesn’t interfere with your default Python setup or other projects.

 ```sh
deactivate
 ```
    
## Download and move the data needed
For the fairness evaluation episode, you will need access to the Medical Expenditure Panel Survey Dataset. Please complete these steps to ensure you have access:

1. Download AI 360 Fairness example data: [Medical Expenditure Panel Survey data](https://raw.githubusercontent.com/carpentries-incubator/fair-explainable-ml/main/data/h181.zip) (zip file)
2. Unzip h181.zip (right-click and extract all on Windows; double-click zip file on Mac)
3. In the unzipped folder, find the h181.csv file. Place the `h181.csv` file in the `aif360` package's data directory within your virtual environment folder:

     - **Windows**:  
       ```
       Desktop/trustworthy_ML/venv/Lib/site-packages/aif360/data/raw/meps/h181.csv
       ```
     - **Mac/Linux**:  
       ```
       [path-to-venv]/lib/python3.x/site-packages/aif360/data/raw/meps/h181.csv
       ```
     

## Create a Hugging Face account and access Token
You will need a Hugging Face account for the workshop episode on model sharing. Hugging Face is a very popular machine learning (ML) platform and community that helps users build, deploy, share, and train machine learning models. 

**Create account**: To create an account on Hugging Face, visit: [huggingface.co/join](https://huggingface.co/join). Enter an email address and password, and follow the instructions provided via Hugging Face (you may need to verify your email address) to complete the process.

**Setup access token**: Once you have your account created, you'll need to generate an access token so that you can upload/share models to your Hugging Face account during the workshop. To generate a token, visit the [Access Tokens setting page](https://huggingface.co/settings/tokens) after logging in. Once there, click “New token” to generate an access token. We'll use this token later to log in to Hugging Face via Python.
