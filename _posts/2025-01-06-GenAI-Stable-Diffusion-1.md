---
list_title: GenAI | Generate images with Stable Diffusion model
title: Generate images with Stable Diffusion model
layout: post
mathjax: true
categories: ["GenAI", "Stable Diffusion"]
---

### Introduction

In previous [articles](https://xta0.me/2019/08/03/Learn-PyTorch-3.html), we have explored an image generation technique using the GAN network. However, in the world of generative models, utilizing text prompts to generate images has become a new trend. In Jan 2020, a paper titled "Denoising Diffusion Probabilities Models" introduced a diffusion-based probability model for image generation. The term <strong>diffusion</strong> is borrowed from thermodynamics. The original meaning is the movement of particles from a region of high concentration to a region of low concentration.

This idea of diffusion inspired machine learning researchers to apply it to <mark>denoising and sampling process</mark>. In other words, <mark>we can start with a noisy image and gradually transforms an image with high-levels of noise into a clear version of the original image</mark>. Therefore, this generative model, is referred to as a denoising diffusion probability model.

<img class="md-img-center" src="{{site.baseurl}}/assets/images/2025/01/sd-03.png">

Essentially, <mark>Stable Diffusion is a set of models</mark> that includes the following:

- <strong>Tokenizer</strong>: This tokenizes a text prompt into a sequence of tokens
- <strong>Text Encoder</strong>: A special Transformer Language model - specifically, the text encoder of a CLIP model.
- <strong>Variational Autoencoder(VAE)</strong>: This encodes images into latent space and decodes them back into images
- <strong>UNet</strong>: This is where the denoising process happens. The UNet architecture is employed to comprehend the steps involved in the nosing/denoising cycle. It accepts certain elements such as noise, time step data, and a conditional signal (for instance, a representation of a text description), and forecasts noise residuals that can be utilized in the denoising process.

In this post, we're going walk through this process by building a small UNet based model that can generate pixel images. But before we dive deep into the model architecture, let's first take a look at the noising and denoising process. 

## The image to noise process

First, we need to normalize the pixels in the image so that their values are within the range `[0,1]`.
Next, we need to generate a noise image of the same size as the original image. Note that the noise should follow a Gaussian distribution (standard normal distribution).Then we mix the noise image and the original image channel by channel (R, G, B) using the following formula:

$$
\sqrt{\beta} \times \epsilon + \sqrt{1 - \beta} \times x
$$

Note that in the formula above, $\epsilon$ represents Gaussian noise, $x$ represents the pixel values of the image, and $\beta$ is a float number between [0,1]. The squares of $\sqrt{\beta}$ and $\sqrt{1 - \beta}$ sum to 1, satisfying the Pythagorean theorem. This means that as $\beta$ changes, the proportion of noise in the original image will also change. 

For example, as $\beta$ increases, the proportion of the original image gradually decreases:

<img class="md-img-center" src="{{site.baseurl}}/assets/images/2025/01/sd-04.png">

It is important to note that each step above relies on the result of the previous calculation. In other words, the noise-adding process is an iterative process, expressed as:

$$
x_t = \sqrt{\beta_t} \times \epsilon_t + \sqrt{1 - \beta_t} \times x_t
$$

where $\epsilon_t$ follows a standard normal distribution:

$$
\epsilon_t ~ N(0,1)
$$

Additionally, the value of $\beta_t$ keeps increasing at each step:

$$
0 < \beta_1 < \beta_2 < \beta_3 < \beta_{t-1} < \beta_t < 1 
$$

Let us define:

$$
\alpha_t = 1 - \beta_t
$$

Then the above formula can be rewritten as:

$$
x_t = \sqrt{1-\alpha_t} \times \epsilon_t + \sqrt{\alpha_t} \times x_{t-1}
$$

Next, we can consider whether it is possible to directly derive $x_t$ from $x_0$, which would eliminate the need for intermediate iterative steps (from $x_1$ to $x_{t-1}$). 

It turns out that we can achieve this using the **reparameterization** trick. By applying mathematical induction (the detailed derivation is omitted here), we can have the following equation:

$$
x_t = \sqrt{1 - a_t a_{t-1} a_{t-2} a_{t-3} \cdots a_2 a_1} \times \epsilon + \sqrt{a_t a_{t-1} a_{t-2} a_{t-3} \cdots a_2 a_1} \times x_0
$$

Here, $a_t a_{t-1} a_{t-2} a_{t-3} \cdots a_2 a_1$ is quite long, so we represent it as $\bar{\alpha}_t$. The equation above can then be further simplified as:

$$
x_t = \sqrt{1 - \bar{\alpha}_t} \times \epsilon + \sqrt{\bar{\alpha}_t} \times x_0 \\

\bar{\alpha}_t = a_t a_{t-1} a_{t-2} a_{t-3} \cdots a_2 a_1
$$

The following code simulates the above process

```python

```

## Resources

- [Denoising Diffusion Probabilities Models](https://arxiv.org/abs/2006.11239)
- [CLIP](https://arxiv.org/pdf/2103.00020)
