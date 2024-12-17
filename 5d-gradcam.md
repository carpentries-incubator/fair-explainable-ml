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
device = 'cpu' # we're using the CPU only version of this workshop 
```
##### Load Model

We'll load the ResNet-50 model from torchvision. This model is pre-trained on the ImageNet dataset, which contains 1.2 million images across 1000 classes.
ResNet-50 is popular model that is a type of convolutional neural network. You can learn more about it here: https://pytorch.org/hub/pytorch_vision_resnet/

```python
from torchvision.models import resnet50

model = resnet50(pretrained=True).to(device).eval() # set to evaluation/inference mode (rather than training)
```

##### Load Test Image
Let's first take a look at the image, which we source from the GradCAM package
```python
# Packages to download images
import requests
from PIL import Image

url = "https://raw.githubusercontent.com/jacobgil/pytorch-grad-cam/master/examples/both.png"
Image.open(requests.get(url, stream=True).raw)
```
Cute, isn't it? Do you prefer dogs or cats?

We will need to convert the image into a tensor to feed it into the model.
Let's create a function to do this for us.

**ML reminder:** A tensor is a mathematical object that can be thought of as a generalization of scalars, vectors, and matrices.
Tensors have a rank (or order), which determines their dimensionality:

* Rank 0: Scalar (a single number, e.g., 5)
* Rank 1: Vector (a 1-dimensional array, e.g., [1, 2, 3])
* Rank 2: Matrix (a 2-dimensional array, e.g., [[1, 2], [3, 4]])
* Rank ≥ 3: Higher-dimensional tensors (e.g., a 3D tensor for images, a 4D tensor for batch processing, etc.)

```python
import numpy as np
from pytorch_grad_cam.utils.image import show_cam_on_image, preprocess_image

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
import matplotlib.pyplot as plt
import cv2
import torch
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

def viz_gradcam(model, target_layers, class_id, input_tensor, rgb_image):
    """
    Visualize Grad-CAM heatmaps for a given model and target class.

    Parameters:
    1. model (torch.nn.Module): The neural network model.
    2. target_layers (list): List of layers to compute Grad-CAM for (usually the last convolutional layer).
    3. class_id (int or None): Target class ID for which Grad-CAM is computed. If None, the model's prediction is used.
    4. input_tensor (torch.Tensor): The input image tensor expected by the model.
    5. rgb_image (numpy.ndarray): The original input image in RGB format, scaled to [0, 1].

    Returns:
    None. Displays a Grad-CAM heatmap over the input image.
    """

    # Step 1: Get predicted class if class_id is not specified
    if class_id is None:
        with torch.no_grad():  # Disable gradient computation for efficiency (not needed for inference)
            outputs = model(input_tensor)  # Run the input image through the model to get output scores
            
            # torch.argmax finds the index of the maximum value in the output tensor.
            # dim=1 indicates we are finding the maximum value **along the class dimension** 
            # (assuming the shape of outputs is [batch_size, num_classes]).
            predicted_class = torch.argmax(outputs, dim=1).item()  # Extract the top class index.
            
            # .item() converts the PyTorch scalar tensor to a Python integer (e.g., tensor(245) -> 245).
            # This is necessary for further operations like accessing the class label from a list.
            print(f"Predicted Class: {labels[predicted_class]} ({predicted_class})")  # Print the predicted label
            
            # Define the target for Grad-CAM visualization.
            # ClassifierOutputTarget wraps the target class for Grad-CAM to calculate activations.
            targets = [ClassifierOutputTarget(predicted_class)]
    else:
        # If a specific class_id is provided, use it directly.
        print(f"Target Class: {labels[class_id]} ({class_id})")
        targets = [ClassifierOutputTarget(class_id)]
    
    # Step 2: Select the Grad-CAM algorithm.
    # Here, we use GradCAM, but this can be swapped for other algorithms like GradCAM++.
    cam_algorithm = GradCAM

    # Step 3: Initialize the Grad-CAM object.
    # This links the model and the target layers where Grad-CAM will compute the gradients.
    cam = cam_algorithm(model=model, target_layers=target_layers)

    # Step 4: Generate the Grad-CAM heatmap.
    # - input_tensor: The input image tensor (preprocessed as required by the model).
    # - targets: The target class for which we compute Grad-CAM (if None, model's prediction is used).
    grayscale_cam = cam(input_tensor=input_tensor, targets=targets)

    # Step 5: Extract the heatmap corresponding to the first input image.
    # The result is [batch_size, height, width], so we select the first image: grayscale_cam[0, :].
    grayscale_cam = grayscale_cam[0, :]

    # Step 6: Overlay the Grad-CAM heatmap on the original input image.
    # - show_cam_on_image: Combines the heatmap with the RGB image (values must be in [0, 1]).
    cam_image = show_cam_on_image(rgb_image, grayscale_cam, use_rgb=True)

    # Step 7: Convert the image from RGB to BGR (OpenCV's default format).
    cam_image = cv2.cvtColor(cam_image, cv2.COLOR_RGB2BGR)

    # Step 8: Display the Grad-CAM heatmap overlaid on the input image.
    plt.imshow(cam_image)  # Show the image with the heatmap.
    plt.axis("off")       # Remove axes for cleaner visualization.
    plt.show()             # Display the plot.
```

Finally, we can start visualizing! Let's begin by seeing what parts of the image the model looks at to make its most confident prediction.
```python
viz_gradcam(model=model, target_layers=target_layers, class_id=None, input_tensor=input_tensor, rgb_image=rgb_image)
```
Interesting, it looks like the model totally ignores the cat and makes a prediction based on the dog.
If we set the output class to "French Bulldog" (`class_id=245`), we see the same visualization - meaning that the model is indeed looking at the correct part of the image to make the correct prediction.
```python
viz_gradcam(model=model, target_layers=target_layers, class_id=245, input_tensor=input_tensor, rgb_image=rgb_image)
```

Let's see what the heatmap looks like when we force the model to look at the cat.
```python
viz_gradcam(model=model, target_layers=target_layers, class_id=281, input_tensor=input_tensor, rgb_image=rgb_image)
```
The model is indeed looking at the cat when asked to predict the class “Tabby Cat” (class_id=281), as Grad-CAM highlights regions relevant to that class. However, the model may still predict the "dog" class overall because the dog's features dominate the output logits when no specific target class is specified.

Let's see another example of this. The image has not only a dog and a cat, but also a items in the background. Can the model correctly identify the door?
```python
viz_gradcam(model=model, target_layers=target_layers, class_id=799, input_tensor=input_tensor, rgb_image=rgb_image)
```
It can! However, it seems to also think of the shelf behind the dog as a door.

Let's try an unrelated object now. Where in the image does the model see a crossword puzzle?
```python
viz_gradcam(model=model, target_layers=target_layers, class_id=918, input_tensor=input_tensor, rgb_image=rgb_image)
```

Looks like our analysis has revealed a shortcoming of the model! It seems to percieve cats and street signs similarly.

Ideally, when the target class is some unrelated object, a good model will look at no significant part of the image. For example, the model does a good job with the class for Dog Sleigh.
```python
viz_gradcam(model=model, target_layers=target_layers, class_id=537, input_tensor=input_tensor, rgb_image=rgb_image)
```
Explaining model predictions though visualization techniques like this can be very subjective and prone to error. However, this still provides some degree of insight a completely black box model would not provide.

Spend some time playing around with different classes and seeing which part of the image the model looks at. Feel free to play around with other base images as well. Have fun!
