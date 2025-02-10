---
list_title: GenAI | How Stable Diffusion Model Works
title: How Stable Diffusion Model Works
layout: post
mathjax: true
categories: ["GenAI", "Stable Diffusion"]
---

### Introduction

In the previous post, we explored the theory behind diffusion models. While the original diffusion model serves as more of a proof of concept, it highlights the immense potential of multi-step diffusion models compared to one-pass neural networks. However, it comes with a significant drawback: the pre-trained model operates in pixel space, which is computationally intensive. In 2022, researchers introduced Latent Diffusion Models, which effectively addressed the performance limitations of earlier diffusion models. <mark>This approach later became widely known as Stable Diffusion</mark>.

At its core, Stable Diffusion is a collection of models that work together to generate images. These components include:

- <strong>Tokenizer</strong>: Converts a text prompt into a sequence of tokens.
- <strong>Text Encoder</strong>: A specialized Transformer-based language model, specifically the text encoder from a CLIP model.
- <strong>Variational Autoencoder (VAE)</strong>: Encodes images into a latent space and reconstructs them back into images.
- <strong>UNet</strong>: The core of the denoising process. This architecture models the noise removal steps by taking inputs such as noise, time-step data, and a conditional signal (e.g., a text representation). It then predicts noise residuals, which guide the image reconstruction process.
This combination of components allows Stable Diffusion to efficiently generate high-quality images while significantly reducing computational costs.

### Latent Space

Stable Diffusion is a type of latent diffusion model, which means that instead of operating directly in pixel space, it works in a lower-dimensional, compressed representation called the latent space. Why Does Stable Diffusion Operate in the Latent Space?

- Computational Efficiency

    - **Reduced Dimensions**: Images in pixel space can be very high-dimensional (e.g., a 512×512 RGB image has 512 × 512 × 3 pixels). Operating in a latent space often reduces the dimensionality by a large factor (e.g., down to 64×64 or 32×32 with several channels), which means fewer computations are required.
    - **Faster Sampling**: The diffusion process, which involves many iterative steps, becomes much faster when each step is operating on a compressed representation.
    - **Memory Efficiency**: Lower-dimensional representations use significantly less memory. This allows for training and sampling on devices with more limited memory (like GPUs) and enables the model to work with larger batch sizes.

- Preserving Semantics:
    - The latent space is designed to capture the high-level, semantic features of an image (like shapes, object positions, and overall style) rather than every fine-grained pixel detail. This focus on semantics allows the diffusion process to operate on the essential content of the image.
    - When the model denoises or generates images in this space, it can later decode the latent representations into detailed images via the decoder, which restores the high-resolution details.

- Improved Quality and Robustness:

Working in latent space can sometimes make the denoising process more stable. The model learns to generate images in a space where the structure of the data is easier to model, and then the decoder is tasked with converting that structure back into the pixel space.
This separation of concerns can lead to better visual quality in the final images.

Both training and sampling processes happen in the latent space, as shown below

<img class="md-img-center" src="{{site.baseurl}}/assets/images/2025/01/sd-02-01.png">

An autoencoder (VAE) is typically used to learn this latent representation. The autoencoder consists of:

- Encoder: Compresses the high-dimensional image into a lower-dimensional latent code.
- Decoder: Reconstructs the original image from the latent code.

### Inference

<img class="md-img-center" src="{{site.baseurl}}/assets/images/2025/01/sd-02-02.png">

Note that the Stable Diffusion models not only support generating images via prompts, they also support image-guided generation. In the previous article, we start the inference process using a noise image that follows the Gaussian distribution. Here, if text is the only input to the model, we can directly create a noise tensor (e.g., `[1, 4, 64, 64]`) as the input latent vector.

- Stable Diffusion uses CLIP to generate an embedding vector, which will be fed into UNet, using the attention mechanism
- If the input contains an image as a guiding signal, we need to first encode the image to a latent vector and then `concat` it with the randomly generated noise tensor that has the same size as the image.

The inference process is similar to the training process. After a number of denoting steps, the latent decoder (VAE) converts the image from latent space to the pixel space.

### The Stable Diffusion XL model pipeline