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

Stable Diffusion is a type of latent diffusion model, which means that instead of operating directly in pixel space, it works in a lower-dimensional, compressed representation called the latent space. But why does Stable Diffusion operate in the latent space?

- **Computational Efficiency**

    - **Reduced Dimensions**: Images in pixel space can be very high-dimensional (e.g., a 512×512 RGB image has 512 × 512 × 3 pixels). Operating in a latent space often reduces the dimensionality by a large factor (e.g., down to 64×64 or 32×32 with several channels), which means fewer computations are required.
    - **Faster Sampling**: The diffusion process, which involves many iterative steps, becomes much faster when each step is operating on a compressed representation.
    - **Memory Efficiency**: Lower-dimensional representations use significantly less memory. This allows for training and sampling on devices with more limited memory (like GPUs) and enables the model to work with larger batch sizes.

- **Preserving Semantics**:
    - The latent space is designed to capture the high-level, semantic features of an image (like shapes, object positions, and overall style) rather than every fine-grained pixel detail. This focus on semantics allows the diffusion process to operate on the essential content of the image.
    - When the model denoises or generates images in this space, it can later decode the latent representations into detailed images via the decoder, which restores the high-resolution details.

Both training and sampling processes happen in the latent space, as shown below

<img class="md-img-center" src="{{site.baseurl}}/assets/images/2025/01/sd-02-01.png">

An autoencoder (VAE) is typically used to learn this latent representation. The autoencoder consists of:

- Encoder: Compresses the high-dimensional image into a lower-dimensional latent code.
- Decoder: Reconstructs the original image from the latent code.

### The Inference Process

<img class="md-img-center" src="{{site.baseurl}}/assets/images/2025/01/sd-02-02.png">

Note that the Stable Diffusion models not only support generating images via prompts, they also support image-guided generation. In the previous article, we start the inference process using a noise image that follows the Gaussian distribution. Here, if text is the only input to the model, we can directly create a noise tensor (e.g., `[1, 4, 64, 64]`) as the input latent vector.

- Stable Diffusion uses CLIP to generate an embedding vector, which will be fed into UNet, using the attention mechanism
- If the input contains an image as a guiding signal. The image needs to be first encode to a latent vector and then `concat` with the randomly generated noise tensor.

The inference process is similar to the training process. After a number of denoting steps, the latent decoder (VAE) converts the image from latent space to the pixel space.

## The Stable Diffusion XL (SDXL) model pipeline

SDXL is a latent diffusion model that has the same overall architecture used in Stable Diffusion v1.5. The UNet backbone is three times larger, there are two text encoders in the SDXL base model, and a separate diffusion-based refinement model is included. The overall architecture is shown as follows:

<img class="md-img-center" src="{{site.baseurl}}/assets/images/2025/01/sd-02-03.png">

Note that the refiner module is optional. Now let's break down the components in detail.

### The VAE of the SDXL

The VAE used in SDXL is a retrained one, using the same autoencoder architecture but with an increased batch size (256 vs 9). Additionally, it tracks the weights with an exponential moving average. The new VAE outperforms the original model in all evaluated metrics. Here is the code for encoding and decoding an image using VAE:

```python
# encode the image from pixel space to latent space
vae_model = AutoencoderKL.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    subfolder = "vae",
    torch_dtype=torch.float32 # for mps, the fp16 does not work
).to("mps")

image_path = './mona.jpg' 
image = load_image(image_path)

image_processor = VaeImageProcessor()

"""
Converts the image to a normailzed torch tensor in range: [-1, 1]
"""
prep_image = image_processor.preprocess(image)
prep_image = prep_image.to("mps", dtype=torch.float32)
print(prep_image.shape) #[1, 3, 296, 296]
with torch.no_grad():
    image_latent = vae_model.encode(prep_image).latent_dist.sample()

print(image_latent.shape) #[1, 4, 37, 37]

# decode the latent
with torch.no_grad():
    decoded_image = vae_model.decode(image_latent, return_dict = False)[0]
    decoded_image = decoded_image.to("cpu")
   
pil_img = image_processor.postprocess(image = decoded_image, output_type="pil")
pil_img = pil_img[0]
plt.imshow(pil_img)
plt.title("")
plt.axis('off')
plt.show()
```
Note that a `[1, 3, 296, 296]` image is encoded into a smaller `[1, 4, 37, 37]` vector. Additionally, the `preprocess` will resize the image, such that `H/8` and `W/8` becomes an integer (296 / 8 = 37).

### The UNet of SDXL

UNet is the backbone of SDXL. The UNet backbone in SDXL is almost three times larger, which 2.6G billion trained parameters, while the SD v1.5 has only 860 million parameters. For SDXL, the minimum 15GB of VRAM is the commonly required; otherwise, we'll need to reduce the image resolution. 

Additionally, the SDXL integrates Transformer block within the UNet architecture, making it more expressive and capable of understanding complex text-image relationships. The U-Net still has CNNs, but each downsampled feature map passes through a Transformer-based attention module before being processed further.

> Check out [this gist](https://gist.github.com/xta0/70a8dcf4b0848ca12b2bda05ed47436a) to view the SDXL's unet architecture

### Text Encoders

One of the most significant changes in SDXL is the text encoder. SDXL uses two text encoders together, CLIP ViT-L and OpenCLIP Vit-bigG(aka openCLIP G/14). 

The OpenCLIP ViT-bigG model is the largest and the best OpenClip model trained on the LAION-2B dataset, a 100 TB dataset containing 2 billion images. While the OpenAI CLIP model generates a 768 dimensional embedding vector, OpenClip G14 outputs a 1,280-dimensional embedding. By concatenating the two embeddings(of the same length), a 2048-dimension embedding is output. This is much larger than previous 768-dimensional embedding from Stable Diffusion v1.5.

```python
import torch
from transformers import CLIPTokenizer, CLIPTextModel

prmpt = "a running dog"

clip_tokenizer = CLIPTokenizer.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    subfolder = "tokenizer",
    dtype = torch.float16
)

input_tokens = clip_tokenizer_1(
    prmpt,
    return_tensors = "pt"
)["input_ids"]

print(input_tokens_1) # [49406, 320, 2761, 7251, 49407]
```

We extract the token_ids from our prompt, resulting in a `[1, 5]` tensor. Note that `49406` and `49407` represent the beginning and ending symbols, respectively.

```python
clip_text_encoder = CLIPTextModel.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    subfolder = "text_encoder",
).to("mps")

# encode token ids to embeddings
with torch.no_grad():
    prompt_embed = clip_text_encode(
        input_tokens_1.to("mps")
    )[0]
print(prompt_embed.shape) #[1, 5, 768]
```
The code above produces an embedding vector with the shape `[1, 5, 768]`, which is expected because it transforms one-dimensional token IDs into 768-dimensional vectors. If we switch to the OpenCLIP ViT-bigG encoder, the resulting embedding vector will have the shape `[1, 5, 1280]`.

In real world, SDXL uses something called **pooled embeddings** from OpenCLIP ViT-bigG. Embedding pooling is the process of converting a sequence of tokens into one embedding vector. In other words, pooling embedding is a lossy compression of information.

Unlike the embedding in the above python example, which encodes each token into an embedding vector, a pooled embedding is one vector that represents the whole input text.

```python

# pooled embedding
clip_text_encoder_3 = CLIPTextModelWithProjection.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    subfolder = "text_encoder_2",
    torch_dtype = torch.float16
).to("mps")

# encode token ids to embeddings
with torch.no_grad():
    prompt_embed_3 = clip_text_encoder_3(
        input_tokens_1.to("mps")
    )[0]
print(prompt_embed_3.shape)
```
The encoder will produce a `[1, 1280]` embedding tensor, as <mark>the maximum token size for a pooled embedding is 77</mark>. In SDXL, the pooled embedding is provided to the UNet together with the token-level embedding from both CLIP and OpenCLIP encoders.

### The two-stage design

The refiner model is just another image-to-image model used to enhance an image by quality adding more details, especially during the last 10 steps. It may not be necessary if the base model can produce high quality images.



## Resource

- [SDXL: Improving Latent Diffusion Models for High-Resolution Image Synthesis](https://arxiv.org/abs/2307.01952)
- [Using Stable Diffusion with Python](https://www.amazon.com/Using-Stable-Diffusion-Python-Generation/dp/1835086373/)