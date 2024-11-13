---
title: Setup
---

## 1) Software setup

::::::::::::::::::::::::::::::::::::::: discussion

### Installing Python using Anaconda

[Python][python] is a popular language for scientific computing, and a frequent choice
for machine learning as well. Installing all of its scientific packages
individually can be a bit difficult, however, so we recommend the installer [Anaconda][anaconda]
which includes most (but not all) of the software you will need.

Regardless of how you choose to install it, please make sure you install Python
version 3.x (e.g., 3.4 is fine). Also, please set up your python environment at
least a day in advance of the workshop.  If you encounter problems with the
installation procedure, ask your workshop organizers via e-mail for assistance so
you are ready to go as soon as the workshop begins.

:::::::::::::::::::::::::::::::::::::::::::::::::::

:::::::::::::::: spoiler

### Windows

Checkout the [video tutorial][video-windows] or:

1. Open [https://www.anaconda.com/products/distribution][anaconda-distribution]
with your web browser.
2. Download the Python 3 installer for Windows.
3. Double-click the executable and install Python 3 using _MOST_ of the
   default settings. The only exception is to check the
   **Make Anaconda the default Python** option.

:::::::::::::::::::::::::

:::::::::::::::: spoiler

### MacOS

Checkout the [video tutorial][video-mac] or:

1. Open [https://www.anaconda.com/products/distribution][anaconda-distribution]
   with your web browser.
2. Download the Python 3 installer for OS X.
   Make sure to use the correct version for your hardware, 
   i.e. choose the options with "(M1)" if yours is one of the more recent models
   containing Apple's chip.
3. Install Python 3 using all of the defaults for installation.

:::::::::::::::::::::::::


:::::::::::::::: spoiler

### Linux

Note that the following installation steps require you to work from the shell.
If you run into any difficulties, please request help before the workshop begins.

1.  Open [https://www.anaconda.com/products/distribution][anaconda-distribution] with your web browser.
2.  Download the Python 3 installer for Linux.
3.  Install Python 3 using all of the defaults for installation.
    a.  Open a terminal window.
    b.  Navigate to the folder where you downloaded the installer
    c.  Type
    ```bash
    bash Anaconda3-
    ```
    and press tab.  The name of the file you just downloaded should appear.
    d.  Press enter.
    e.  Follow the text-only prompts.  When the license agreement appears (a colon
        will be present at the bottom of the screen) hold the down arrow until the
        bottom of the text. Type `yes` and press enter to approve the license. Press
        enter again to approve the default location for the files. Type `yes` and
        press enter to prepend Anaconda to your `PATH` (this makes the Anaconda
        distribution the default Python).

:::::::::::::::::::::::::

### Installing the required packages

[Conda](https://docs.conda.io/projects/conda/en/latest/) is the package management system associated with [Anaconda](https://anaconda.org) and runs on Windows, macOS, and Linux.
Conda should already be available in your system once you installed Anaconda successfully. Conda thus works regardless of the operating system.

1. Make sure you have an up-to-date version of Conda running.
   See [these instructions](https://docs.anaconda.com/anaconda/install/update-version/) for updating Conda if required.

2. Create the Conda Environment: To create a conda environment called `trustworthy_ML` with the required packages, open a terminal (Mac/Linux) or Anaconda prompt (Windows) and type the below command. This command creates a new conda environment named `trustworthy_ML` and installs the necessary packages from the `conda-forge` and `pytorch` channels. When prompted to Proceed ([y]/n) during environment setup, press y. It may take around 10-20 minutes to complete the full environment setup. Please reach out to the workshop organizers sooner rather than later to fix setup issues prior to the workshop. 
   
    ```sh
    conda create --name trustworthy_ML python=3.9 pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia 
    ```
    To install other pytorch version based on your cuda version / more detailed instructions, you can checkout https://pytorch.org/get-started/locally/.

3. Activate the Conda Environment: After creating the environment, activate it using the following command.
   
    ```sh
    conda activate trustworthy_ML
    ```

4. Install `pytorch-ood`, `fairlearn`, `aif360[Reductions]`, and `aif360[inFairness]` using pip. Make sure to do this AFTER activating the environment.
   
    ```sh
    conda install jupyter scikit-learn pandas matplotlib keras tensorflow umap-learn aif360 -c conda-forge
    pip install pytorch-ood
    pip install fairlearn
    pip install aif360[Reductions]
    pip install aif360[inFairness]
    ```

    Depending on your AIF360 installation, the final two `pip install` commands may or may not work. If they do not work, then installing these sub-packages is not necessary -- you can continue on. 

5. Deactivating environment (complete at end of each day). Deactivating environments is part of good workflow hygiene. If you keep this environment active and then start working on another project, you may inadvertently use the wrong environment. This can lead to package conflicts or incorrect dependencies being used. To deactive your environment, use:

    ```sh
    conda deactivate
    ```

#### Notes
- The `conda-forge` is a community-driven conda channel that provides a wide array of up-to-date packages, ensuring better compatibility and a more extensive package library.


### Starting Jupyter Lab

We will teach using Python in Jupyter lab, a
programming environment that runs in a web browser. Jupyter requires a reasonably
up-to-date browser, preferably a current version of Chrome, Safari, or Firefox
(note that Internet Explorer version 9 and below are *not* supported). If you
installed Python using Anaconda, Jupyter should already be on your system. If
you did not use Anaconda, use the Python package manager pip to acquire Jupyter
(see the [Jupyter website](https://jupyter.org/install) for details.)

To start jupyter lab, open a terminal (Mac/Linux) or Anaconda prompt (Windows) and type the command:

```bash
jupyter lab
```

### Check your software setup
To check whether all packages installed correctly, start a jupyter notebook in jupyter lab as
explained above. Run the following lines of code:
```python
import sklearn
print('sklearn version: ', sklearn.__version__)

import pandas
print('pandas version: ', pandas.__version__)

import torch
print('torch version: ', torch.__version__)
```

This should output the versions of all required packages without giving errors.
Most versions will work fine with this lesson, but:
- For pytorch, the minimum version is 2.0
- For sklearn, the minimum version is 1.2.2

### Fallback option: cloud environment
If a local installation does not work for you, it is also possible to run this lesson in [Google Colab](https://colab.research.google.com/). Some packages may need to be installed on the fly within the notebook (TBD).

## 2) Download and move the data needed
For the fairness evaluation episode, you will need access to the Medical Expenditure Panel Survey Dataset. Please complete these steps to ensure you have access:

1. Download AI 360 Fairness example data: [Medical Expenditure Panel Survey data](https://raw.githubusercontent.com/carpentries-incubator/fair-explainable-ml/main/data/h181.zip) (zip file)
2. Unzip h181.zip (right-click and extract all on Windows; double-click zip file on Mac)
3. In the unzipped folder, find the h181.csv file. If you installed `conda` with Anaconda, i.e., as described earlier in this document, move this file to the following location:

      * Windows: `C:\Users\[Usernmae]\anaconda3\envs\trustworthy_ML\Lib\site-packages\aif360\data\raw\meps\h181.csv`
      * Mac:  `/Users/[Username]/opt/anaconda3/envs/trustworthy_ML/lib/python3.9/site-packages/aif360/data/raw/meps/h181.csv`

    If you installed `conda` in a different way, or don't remember how you installed it, check the location of your `trustworthy_ML` environment (make sure this environment is active, first!):

    * Windows: `where python3.9`
    * Mac: `which python3.9`. 
    
    Follow the instructions above, but replace everything before `/trustworthy_ML` with the printed path to `/trustworthy_ML`.
     

## 3) Create a Hugging Face account and access Token
You will need a Hugging Face account for the workshop episode on model sharing. Hugging Face is a very popular machine learning (ML) platform and community that helps users build, deploy, share, and train machine learning models. 

**Create account**: To create an account on Hugging Face, visit: [huggingface.co/join](https://huggingface.co/join). Enter an email address and password, and follow the instructions provided via Hugging Face (you may need to verify your email address) to complete the process.

**Setup access token**: Once you have your account created, you’ll need to generate an access token so that you can upload/share models to your Hugging Face account during the workshop. To generate a token, visit the [Access Tokens setting page](https://huggingface.co/settings/tokens) after logging in. Once there, click “New token” to generate an access token. We’ll use this token later to log in to Hugging Face via Python




