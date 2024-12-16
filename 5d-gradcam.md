---
title: "Explainability methods: GradCAM"
teaching: 0
exercises: 0
---
:::::::::::::::::::::::::::::::::::::: questions 
- How can we identify which parts of an input contribute most to a model’s prediction?  
- What insights can saliency maps, GradCAM, and similar techniques provide about model behavior?  
- What are the strengths and limitations of gradient-based explainability methods?  

::::::::::::::::::::::::::::::::::::::::::::::::

::::::::::::::::::::::::::::::::::::: objectives

- Explain how saliency maps and GradCAM work and their applications in understanding model predictions.  
- Introduce GradCAM as a method to visualize the important features used by a model.  
- Gain familiarity with the PyTorch and GradCam libraries for vision models. 
::::::::::::::::::::::::::::::::::::::::::::::::

## What part of my input causes this prediction?

When a model makes a prediction, we often want to know which parts of the input were most important in generating that prediction.
This helps confirm if the model is making its predictions for the right reasons. 
Sometimes, models use features totally unrelated to the task for their prediction - these are known as 'spurious correlations'.
For example, a model might predict that a picture contains a dog because it was taken in a park, and not because there is actually a dog in the picture.

**[Saliency Maps](https://arxiv.org/abs/1312.6034)** are among the most simple and popular methods used towards this end. 
We will be working with a more sophisticated version of this method, known as **[GradCAM](https://arxiv.org/abs/1610.02391)**.

#### Method and Examples

A saliency map is a kind of visualization - it is a heatmap across the input that shows which parts of the input are most important in generating the model's prediction.
They can be calculated using the gradients of a neural network, or by perturbing the input to any ML model and observing how the model reacts to these perturbations.
The key intuition is that if a small change in a part of the input causes a large change in the model's prediction, then that part of the input is important for the prediction.
Gradients are useful in this because they provide a signal towards how much the model's prediction would change if the input was changed slightly.

For example, in an image classification task, a saliency map can be used to highlight the parts of the image that the model is focusing on to make its prediction.
In a text classification task, a saliency map can be used to highlight the words or phrases that are most important for the model's prediction.

GradCAM is an extension of this idea, which uses the gradients of the final layer of a convolutional neural network to generate a heatmap that highlights the important regions of an image.
This heatmap can be overlaid on the original image to visualize which parts of the image are most important for the model's prediction.

Other variants of this method include [Integrated Gradients](https://arxiv.org/abs/1703.01365), [SmoothGrad](https://arxiv.org/pdf/1806.03000), and others, which are designed to provide more robust and reliable explanations for model predictions.
However, GradCAM is a good starting point for understanding how saliency maps work, and is a popularly used approach.

Alternative approaches, which may not directly generate heatmaps, include [LIME](https://arxiv.org/abs/1602.04938) and [SHAP](https://arxiv.org/abs/1705.07874), which are also popular and recommended for further reading. 

#### Limitations and Extensions

Gradient based saliency methods like GradCam are fast to compute, requiring only a handful of backpropagation steps on the model to generate the heatmap.
The method is also model-agnostic, meaning it can be applied to any model that can be trained using gradient descent.
Additionally, the results obtained from these methods are intuitive and easy to understand, making them useful for explaining model predictions to non-experts.

However, their use is limited to models that can be trained using gradient descent, and have white-box access. 
It is also difficult to apply these methods to tasks beyond classification, making their application limited with many recent
generative models (think LLMs).

Another limitation is that the insights gained from these methods are not actionable - knowing which part of the input caused the prediction does not highlight why that part caused it.
On finding issues in the prediction process, it is also hard to pick up on if there is an underlying issue in the model, or just the specific inputs tested on.


## Implementing GradCAM

```python
# Packages to download test images
import requests

# Packages to view and process images
import matplotlib.pyplot as plt
import cv2
import numpy as np
from PIL import Image

# Packages to load the model
import torch
from torchvision.models import resnet50

# GradCAM Packaes
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image, preprocess_image
```
```python
device = 'cpu' # we're using the CPU only version of this workshop 
```
##### Load Model

We'll load the ResNet-50 model from torchvision. This model is pre-trained on the ImageNet dataset, which contains 1.2 million images across 1000 classes.
ResNet-50 is popular model that is a type of convolutional neural network. You can learn more about it here: https://pytorch.org/hub/pytorch_vision_resnet/
```python
model = resnet50(pretrained=True).to(device).eval()
```
##### Load Test Image
Let's first take a look at the image, which we source from the GradCAM package
```python
url = "https://raw.githubusercontent.com/jacobgil/pytorch-grad-cam/master/examples/both.png"
Image.open(requests.get(url, stream=True).raw)
```
Cute, isn't it? Do you prefer dogs or cats?

We will need to convert the image into a tensor to feed it into the model.
Let's create a function to do this for us.
```python
def load_image(url):
    rgb_img = np.array(Image.open(requests.get(url, stream=True).raw))
    rgb_img = np.float32(rgb_img) / 255
    input_tensor = preprocess_image(rgb_img).to(device)
    return input_tensor, rgb_img
```
```python
input_tensor, rgb_image = load_image(url)
```
### Grad-CAM Time!
Let's start by selecting which layers of the model we want to use to generate the CAM.
For that, we will need to inspect the model architecture.
We can do that by simply printing the model object.
```python
print(model)
```
Here we want to interpret what the model as a whole is doing (not what a specific layer is doing).
That means that we want to use the embeddings of the last layer before the final classification layer.
This is the layer that contains the information about the image encoded by the model as a whole.

Looking at the model, we can see that the last layer before the final classification layer is `layer4`.
```python
target_layers = [model.layer4]
```
We also want to pick a label for the CAM - this is the class we want to visualize the activation for.
Essentially, we want to see what the model is looking at when it is predicting a certain class.

Since ResNet was trained on the ImageNet dataset with 1000 classes, let's get an indexed list of those classes. We can then pick the index of the class we want to visualize.
```python
imagenet_categories_url = \
     "https://gist.githubusercontent.com/yrevar/942d3a0ac09ec9e5eb3a/raw/238f720ff059c1f82f368259d1ca4ffa5dd8f9f5/imagenet1000_clsidx_to_labels.txt"
labels = eval(requests.get(imagenet_categories_url).text)
labels
```
Well, that's a lot! To simplify things, we have already picked out the indices of a few interesting classes.

- 157: Siberian Husky
- 162: Beagle
- 245: French Bulldog
- 281: Tabby Cat
- 285: Egyptian cat
- 360: Otter
- 537: Dog Sleigh
- 799: Sliding Door
- 918: Street Sign
```python
# Specify the target class for visualization here. If you set this to None, the class with the highest score from the model will automatically be used.
visualized_class_id = 245
```
```python
def viz_gradcam(model, target_layers, class_id):

    if class_id is None:
        targets = None
    else:
        targets = [ClassifierOutputTarget(class_id)]

    cam_algorithm = GradCAM
    with cam_algorithm(model=model, target_layers=target_layers) as cam:
        grayscale_cam = cam(input_tensor=input_tensor, targets=targets)

        grayscale_cam = grayscale_cam[0, :]

        cam_image = show_cam_on_image(rgb_image, grayscale_cam, use_rgb=True)
        cam_image = cv2.cvtColor(cam_image, cv2.COLOR_RGB2BGR)

    plt.imshow(cam_image)
    plt.axis("off")
    plt.show()
```
Finally, we can start visualizing! Let's begin by seeing what parts of the image the model looks at to make its most confident prediction.
```python
viz_gradcam(model=model, target_layers=target_layers, class_id=None)
```
Interesting, it looks like the model totally ignores the cat and makes a prediction based on the dog.
If we set the output class to "French Bulldog" (`class_id=245`), we see the same visualization - meaning that the model is indeed looking at the correct part of the image to make the correct prediction.

Let's see what the heatmap looks like when we force the model to look at the cat.
```python
viz_gradcam(model=model, target_layers=target_layers, class_id=281)
```
The model is indeed looking at the cat when asked to predict the class "Tabby Cat" (`class_id=281`)!
But why is it still predicting the dog? Well, the model was trained on the ImageNet dataset, which contains a lot of images of dogs and cats.
The model has learned that the dog is a better indicator of the class "Tabby Cat" than the cat itself.

Let's see another example of this. The image has not only a dog and a cat, but also a items in the background. Can the model correctly identify the door?
```python
viz_gradcam(model=model, target_layers=target_layers, class_id=799)
```
It can! However, it seems to also think of the shelf behind the dog as a door.

Let's try an unrelated object now. Where in the image does the model see a street sign?
```python
viz_gradcam(model=model, target_layers=target_layers, class_id=918)
```
Looks like our analysis has revealed a shortcoming of the model! It seems to percieve cats and street signs similarly.

Ideally, when the target class is some unrelated object, a good model will look at no significant part of the image. For example, the model does a good job with the class for Dog Sleigh.
```python
viz_gradcam(model=model, target_layers=target_layers, class_id=537)
```
Explaining model predictions though visualization techniques like this can be very subjective and prone to error. However, this still provides some degree of insight a completely black box model would not provide.

Spend some time playing around with different classes and seeing which part of the image the model looks at. Feel free to play around with other base images as well. Have fun!
