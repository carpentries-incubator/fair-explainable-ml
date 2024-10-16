---
title: "Explainability methods: GradCAM"
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
# Let's begin by installing the grad-cam package - this will significantly simplify our implementation
!pip install grad-cam
```
```python
# Packages to download test images
import requests

# Packages to view and process images
import cv2
import numpy as np
from PIL import Image
from google.colab.patches import cv2_imshow

# Packages to load the model
import torch
from torchvision.models import resnet50

# GradCAM Packaes
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image, preprocess_image
```
```python
device = 'gpu' if torch.cuda.is_available() else 'cpu'
```
##### Load Model

We'll load the ResNet-50 model from torchvision. This model is pre-trained on the ImageNet dataset, which contains 1.2 million images across 1000 classes.
ResNet-50 is popular model that is a type of convolutional neural network. You can learn more about it here: https://pytorch.org/hub/pytorch_vision_resnet/
```python
model = resnet50(pretrained=True).to(device).eval()
```
##### Load Test Image
```python
# Let's first take a look at the image, which we source from the GradCAM package

url = "https://raw.githubusercontent.com/jacobgil/pytorch-grad-cam/master/examples/both.png"
Image.open(requests.get(url, stream=True).raw)
```
```python
# Cute, isn't it? Do you prefer dogs or cats?

# We will need to convert the image into a tensor to feed it into the model.
# Let's create a function to do this for us.
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
```python
# Let's start by selecting which layers of the model we want to use to generate the CAM.
# For that, we will need to inspect the model architecture.
# We can do that by simply printing the model object.
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
      grayscale_cam = cam(input_tensor=input_tensor,
                          targets=targets)

      grayscale_cam = grayscale_cam[0, :]

      cam_image = show_cam_on_image(rgb_image, grayscale_cam, use_rgb=True)
      cam_image = cv2.cvtColor(cam_image, cv2.COLOR_RGB2BGR)

  cv2_imshow(cam_image)
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