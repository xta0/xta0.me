---
list_title: GenAI | Generate images with Stable Diffusion model
title: Generate images with Stable Diffusion model
layout: post
mathjax: true
categories: ["GenAI", "Stable Diffusion"]
---

### Introduction to the Stable Diffusion model

In previous [articles](https://xta0.me/2019/08/03/Learn-PyTorch-3.html), we have explored an image generation technique using the GAN network. In the world of generative models, utilizing text prompts to generate images has become a new trend. In Jan 2020, a paper titled "Denoising Diffusion Probabilities Models" introduced a diffusion-based probability model for image generation. <mark>This idea of diffusion inspired machine learning researchers to apply it to denoising and sampling process</mark>. In other words, <mark>we can start with a noisy image and gradually transforms an image with high-levels of noise into a clear version of the original image</mark>. Therefore, this generative model, is referred to as a denoising diffusion probability model.

The idea behind this approach is ingenious, For any given image, a limited number of normally distributed noise images are added to the original image, effectively transforming it into a fully noisy image. What if we train a model that can reverse this diffusion process? In this article, we're going to explore this process by building a small UNet based model that can generate small avatar images.

<img class="md-img-center" src="{{site.baseurl}}/assets/images/2025/01/sd-01.png">

Essentially, Stable Diffusion is a set of models that includes the following:

- <strong>Tokenizer</strong>: This tokenizes a text prompt into a sequence of tokens
- <strong>Text Encoder</strong>: A special Transformer Language model - specifically, the text encoder of a CLIP model.
- <strong>Variational Autoencoder(VAE)</strong>: This encodes images into latent space and decodes them back into images
- <strong>UNet</strong>: This is where the denoising process happens. The UNet architecture is employed to comprehend the steps involved in the nosing/denoising cycle. It accepts certain elements such as noise, time step data, and a conditional signal (for instance, a representation of a text description), and forecasts noise residuals that can be utilized in the denoising process.



## Resources

- [Denoising Diffusion Probabilities Models](https://arxiv.org/abs/2006.11239)
