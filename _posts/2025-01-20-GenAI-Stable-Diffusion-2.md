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

<img class="md-img-center" src="{{site.baseurl}}/assets/images/2025/01/sd-02-01.png">