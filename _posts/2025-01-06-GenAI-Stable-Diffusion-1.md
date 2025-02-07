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

<img class="md-img-center" src="{{site.baseurl}}/assets/images/2025/01/sd-01.png">
<img class="md-img-center" src="{{site.baseurl}}/assets/images/2025/01/sd-02.png">

The idea behind this approach is ingenious, For any given image, a limited number of normally distributed noise images are added to the original image, effectively transforming it into a fully noisy image. What if we train a model that can reverse this diffusion process? 

Essentially, Stable Diffusion is a set of models that includes the following:

- <strong>Tokenizer</strong>: This tokenizes a text prompt into a sequence of tokens
- <strong>Text Encoder</strong>: A special Transformer Language model - specifically, the text encoder of a CLIP model.
- <strong>Variational Autoencoder(VAE)</strong>: This encodes images into latent space and decodes them back into images
- <strong>UNet</strong>: This is where the denoising process happens. The UNet architecture is employed to comprehend the steps involved in the nosing/denoising cycle. It accepts certain elements such as noise, time step data, and a conditional signal (for instance, a representation of a text description), and forecasts noise residuals that can be utilized in the denoising process.

In this post, we're going walk through this process by building a small UNet based model that can generate pixel images. But before we dive deep into the model architecture, let's first take a look at the noising and denoising process. 

## The image to noise process

- First, we need to normalize the pixels in the image so that their values are within the range `[0,1]`.
- Next, we need to generate a noise image of the same size as the original image. Note that the noise should follow a Gaussian distribution (standard normal distribution).
- Then we mix the noise image and the original image channel by channel (R, G, B) using the following formula:

$$
\sqrt{\beta} \times \epsilon + \sqrt{1 - \beta} \times x
$$

Note that in the formula above, $\epsilon$ represents Gaussian noise, $x$ represents the pixel values of the image, and $\beta$ is a float number between [0,1]. The squares of $\sqrt{\beta}$ and $\sqrt{1 - \beta}$ sum to 1, satisfying the Pythagorean theorem. This means that as $\beta$ changes, the proportion of noise in the original image will also change. As $\beta$ increases, the proportion of the original image gradually decreases.

<img class="md-img-center" src="{{site.baseurl}}/assets/images/2025/01/sd-04.png">

上图中，我们
The following code adds a Gaussian noise to an image and then visualizes the progression over a number of iterations. It performs a simulation of a forward diffusion process:

```python

```

## The Sampling Process

Before we dive deep into how to train the network, let's first discuss the sampling process, or what we do with the network after it's trained at inference time.

You first have a noise sample image, and you put it through the network. The model outputs a noise image. Then we subtract the predicted noise image with the input noise sample image To get something more like a sprite image.

<img class="md-img-center" src="{{site.baseurl}}/assets/images/2025/01/sd-03.png">

We do this over and over again until we get a high quality output (500 iterations in the above example).
The above process can be implemented using the following code


```python
# sample using standard algorithm
@torch.no_grad()
def sample_ddpm(n_sample, save_rate=20):
    # x_T ~ N(0, 1), sample initial noise
    samples = torch.randn(n_sample, 3, height, height).to(device)  

    # array to keep track of generated steps for plotting
    intermediate = [] 
    for i in range(timesteps, 0, -1):
        print(f'sampling timestep {i:3d}', end='\r')

        # reshape time tensor
        t = torch.tensor([i / timesteps])[:, None, None, None].to(device)

        # sample some random noise to inject back in. For i = 1, don't add back in noise
        z = torch.randn_like(samples) if i > 1 else 0
        # predict noise 
        eps = nn_model(samples, t)    e_(x_t,t)
        samples = denoise_add_noise(samples, i, eps, z)

    return samples
```



## Resources

- [Denoising Diffusion Probabilities Models](https://arxiv.org/abs/2006.11239)
- [CLIP](https://arxiv.org/pdf/2103.00020)