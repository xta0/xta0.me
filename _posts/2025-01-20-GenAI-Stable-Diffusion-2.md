---
list_title: GenAI | How Stable Diffusion Model Works
title: How Stable Diffusion Model Works
layout: post
mathjax: true
categories: ["GenAI", "Stable Diffusion"]
---

Essentially, <mark>Stable Diffusion is a set of models</mark> that includes the following:

- <strong>Tokenizer</strong>: This tokenizes a text prompt into a sequence of tokens
- <strong>Text Encoder</strong>: A special Transformer Language model - specifically, the text encoder of a CLIP model.
- <strong>Variational Autoencoder(VAE)</strong>: This encodes images into latent space and decodes them back into images
- <strong>UNet</strong>: This is where the denoising process happens. The UNet architecture is employed to comprehend the steps involved in the nosing/denoising cycle. It accepts certain elements such as noise, time step data, and a conditional signal (for instance, a representation of a text description), and forecasts noise residuals that can be utilized in the denoising process.