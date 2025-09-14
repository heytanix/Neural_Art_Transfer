# Neural Art Transfer

## Introduction
his project implements Neural Style Transfer, a powerful technique that uses deep learning to compose images in the style of another image. Based on the paper "[A Neural Algorithm of Artistic Style](https://arxiv.org/abs/1508.06576)" by Gatys et al., this project leverages a pre-trained convolutional neural network (CNN) to separate and recombine the content and style of arbitrary images.

## 🫡 You can help me by Donating
[![BuyMeACoffee](https://img.shields.io/badge/Buy%20Me%20a%20Coffee-ffdd00?style=for-the-badge&logo=buy-me-a-coffee&logoColor=black)](https://buymeacoffee.com/heytanix)

### Image Overview:
- **Content Image**: Any photograph or image that provides the subject and structure for the final output.
- **Style Image**: An image, typically an artwork, that provides the artistic style (e.g., textures, colors, and patterns) to be applied.
- **Generated Image**: The final output, which merges the content of the content image with the style of the style image.

**Objective**:
The goal of this project is to create an algorithm that takes two images—a content image and a style image—and generates a third image that retains the content of the first while adopting the artistic style of the second.

## Algorithms and Techniques Used
To achieve the style transfer effect, the following deep learning techniques and components were employed:

1.  **Pre-trained VGG19 Network**: A deep convolutional neural network used as a feature extractor. The network itself is not trained; only its pre-trained weights are used.
2.  **Content Loss**: A loss function that measures the difference in high-level content between the content image and the generated image.
3.  **Style Loss**: A loss function that uses a **Gram Matrix** to measure the difference in style (textures, patterns, colors) between the style image and the generated image across multiple layers.
4.  **Adam Optimizer**: An optimization algorithm used to iteratively update the pixels of the generated image to minimize the total loss.

## Project Workflow
### Step 1: Data Preprocessing
- Loaded the content and style images from disk.
- Preprocessed the images by resizing them to a consistent size and normalizing them with the mean and standard deviation required by the VGG19 model.

### Step 2: Model Definition
- Loaded the pre-trained, frozen VGG19 model from `torchvision`.
- Built a new model by dynamically inserting custom `ContentLoss` and `StyleLoss` layers after specific convolutional layers of the VGG network. These layers capture and compare feature maps during the forward pass.

### Step 3: Optimization and Image Generation
- Cloned the content image to serve as the starting point for the generated image.
- Defined an optimization loop that feeds the generated image into the custom model.
- Calculated the total loss (a weighted sum of content and style losses) and used backpropagation to compute gradients with respect to the generated image's pixels.
- The Adam optimizer then updated the image based on these gradients. This loop was run for a set number of steps.

### Step 4: Final Output
- After the optimization loop completed, the final generated tensor was post-processed.
- This involved reversing the normalization and converting the tensor back into a viewable image format.

### Conclusion
The project successfully demonstrates that by defining loss functions based on feature representations from a deep CNN, it is possible to separate and recombine the content and style of images.

- The **key to the technique** is the optimization process, which does not train a network but rather **trains an image** to simultaneously satisfy two constraints: matching the content of one image and the style of another. The final artistic outcome can be finely controlled by adjusting the weights of the content and style losses.

## Acknowledgments
- [A Neural Algorithm of Artistic Style by Gatys, Ecker, & Bethge](https://arxiv.org/abs/1508.06576)
- [PyTorch Official Style Transfer Tutorial](https://pytorch.org/tutorials/advanced/neural_style_tutorial.html)
- [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)
