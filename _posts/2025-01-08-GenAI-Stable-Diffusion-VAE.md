---
list_title: Stable Diffusion | Theory Behind the Diffusion Models
title: Theory Behind the Diffusion Models
layout: post
mathjax: true
categories: ["GenAI", "Stable Diffusion"]
---

## Variational Autoencoder (VAE)

The VAE in Stable Diffusion doesn’t control the image generation process. Instead, it compresses images into a lower-dimensional latent representation before diffusion, and decompresses the final latent back into an image after the diffusion model has finished sampling.

### The Autoencoder Architecture

Before we dive into VAEs, it's important to first understand the architecture of Autoencoder. <mark>Autoencoder is a likelihood-based approach for image generation</mark>. The Autoencoder architecture consists of three main components: the encoder, the bottleneck(or latent space), and the decoder.

<img class="md-img-center" src="{{site.baseurl}}/assets/images/2025/01/sd-ae-02.png">

- The **encoder** compress the input data into a "latent-space" representation, which is a low-dimensional space that captures the essential features of the input data.
- The **bottleneck** layer is the smallest layer that holds the compressed representation
- The **decoder** reconstructs the input data from the compressed representation

Training an Autoencoder is a self-supervised process that focuses on minimizing the difference between the original data and its reconstructed version. <mark>The goal is to improve the decoder's ability to accurately reconstruct the original data from the compressed latent representation. At the same time, the encoder becomes better at compressing the data in a way of preserving critical information, ensuring that the original data can be effectively reconstructed.</mark>

If the model is well-trained, the encoder can be used separately to perform data dimensional reduction. It maps images with similar features (e.g., animals, trees) to nearby regions in latent space. For example, images of dogs will cluster closer to each other than to images of trees in the latent space.

### Latent Space

What is latent space, and why is it helpful in Autoencoder or Stable Diffusion?

- **Reduced Dimensions**: Images in pixel space can be very high-dimensional (e.g., a `512×512` RGB image has `512 × 512 × 3` pixels). Operating in a latent space often reduces the dimensionality by a large factor (e.g., down to `64×64` or `32×32` with several channels), which means fewer computations are required.
- **Faster Sampling**: The diffusion process, which involves many iterative steps, becomes much faster when each step is operating on a compressed representation.
- **Memory Efficiency**: Lower-dimensional representations use significantly less memory. This allows for training and sampling on devices with more limited memory (like GPUs) and enables the model to work with larger batch sizes.
- **Preserving Semantics**: <mark>The latent space is designed to capture the high-level, semantic features of an image</mark> (like shapes, object positions, and overall style) rather than every fine-grained pixel detail. <mark>This focus on semantics allows the diffusion process to operate on the essential content of the image</mark>.

### Variational Bayes

With the Autoencoder architecture, a natural idea for generating new images would be to randomly sample points from this latent space and run them through the decoder. Unfortunately, this approach won't produce meaningful images mainly because the latent space is unstructured and disorganized. For example, if you sample a random latent vector (say, from $N(0,1)$), there's no guarantee it maps to a valid image.

That is why most of modern implementation of Autoencoders regularize the latent space. VAE is the most famous type of regularized autoencoder. The latent space produced by VAE is structured and continuous, <mark>following a standard Gaussian distribution</mark>. This makes it easy to sample new points and interpolate smoothly between them

<img class="md-img-center" src="{{site.baseurl}}/assets/images/2025/01/sd-ae-04.png">

VAE was first introduced in 2013 in this paper named [Auto-Encoding Variational Bayes](https://arxiv.org/abs/1312.6114). The idea is that, in order to reconstruct an image from a latent vector, we need to calculate the conditional probability $p(x\|z)$, where $p(z)$ is a latent distribution that capture the core features of the images. To make this process computable, we assume $p(z)$ follows the standard Gaussian distribution: $p(z) = N(0, 1)$. This allows us to compute the likelyhood $p(x\|z)$. 

Then the question is how do we produce this Gaussian Distribution? That's where VAE kicks in, it learns the parameter $\mu$ and $\sigma$, which is an optimization process usually known as <mark>variational Bayes</mark>. Here is the process:

1. We train a deep encoder to estimate these parameters $\mu$ and $\sigma$ from the images
2. Then we use a decoder to reconstruct images from a sampled latent variable
3. Compare the reconstructed images with the original images with the following loss function:

<img class="md-img-center" src="{{site.baseurl}}/assets/images/2025/01/sd-ae-05.png">

The first part of the equation simply measures how well our model can reconstruct an image $x$ from its encoded latent variable $z$. The $log$ likelyhood term reduces to a simple $L_2$ reconstruction loss, also known as mean square error. It just measures the loss between the reconstructed image and the original image. 

The second part of the equation is the KL divergence that measures the distant between two probability distributions. In our case, it measures the distance between $p(z)$ and the normal distribution. While minimizing the loss, $p(z)$ will take shape of the normal distribution as well.

<img class="md-img-center" src="{{site.baseurl}}/assets/images/2025/01/sd-ae-03.png">

After the training finishes, the latent space should look very closely to the shape of a 2D normal distribution as shown in the image above.

### Use VAE to generate images

Once the model is trained, the encoder can encode the images to their latent representation. However, VAE does this a bit differently than the plain Autoencoder. Instead of mapping the input image directly to a single latent vector in the latent space, <mark>the VAE encoder converts the input into a group of vectors that follows the Gaussian distribution</mark>, namely $\mu$ and $\sigma$. From this latent distribution, we sample points at random, and the decoder converts these sampled points back into the input space. By sampling a random latent vector, we can create a wide variety of new images.

<div class="md-flex-h md-flex-no-wrap">
<div><img src="{{site.baseurl}}/assets/images/2025/01/sd-ae-06.png"></div>
<div class="md-margin-left-12"><img src="{{site.baseurl}}/assets/images/2025/01/sd-ae-07.png"></div>
</div>


In summary, VAE maintains a compact and smooth latent space, ensuring that <mark>most of the points (latent vectors) within the normal distribution in the latent space will lead to plausible samples</mark>. Appendix #2 demonstrates [how to use VAE in Stable Diffusion 1.5](#appendix-2-vae-in-stable-diffusion).


## Resources

- [How Diffusion Models Work](https://www.coursera.org/projects/how-diffusion-models-work-project)
- [Denoising Diffusion Probabilities Models](https://arxiv.org/abs/2006.11239)
- [Using Stable Diffusion with Python](https://www.amazon.com/Using-Stable-Diffusion-Python-Generation/dp/1835086373/)

