---
list_title: GenAI | Fine-tuning Stable Diffusion Models
title: Fine-tunning Stable Diffusion Models
layout: post
mathjax: true
categories: ["GenAI", "Stable Diffusion"]
---

## LoRA Fine-tuning

[In previous articles](https://xta0.me/2024/11/24/GenAI-LLM-2.html), we briefly discussed LoRA as a method for fine-tuning LLMs. With LoRA, the original model remains unchanged and frozen, while the fine-tuned weight adjustments are stored separately in what is known as a LoRA file.

LoRA works by creating a small, low-rank model that is adapted for a specific concept. This small model can be merged with the main checkpoint model to generate images during the inference stage.

Let's use $W$ to represent the original UNet attention weights(`Q`, `K`, `V`), $\Delta W$ to denote the fine-tuned weights from LoRA, and $W'$ as the combined weights. The process of adding LoRA to a model can be expressed as:

$$
W' = W + \Delta W
$$

If we want to control the scale of LoRA weights, we can leverage a scale factor $\alpha$:

$$
W' = W + \alpha\Delta W
$$

The range of $\alpha$ can be from `0` to `1.0`. It should be fine if we set $\alpha$ slightly larger than `1.0`.

The reason why LoRA is so small is that $\Delta W$ can be represented by two small low-rank matrices $A$ and $B$, such that:

$$
\Delta W = AB^T
$$

Where $A$ is a `n x d` matrix, and $B$ is a `m x d` matrix. For example, if $\Delta W$ is a `6x8` matrix, there a total of 48 weight numbers. Now, in the LoRA file, the `6x8` matrix can be divided by simply two small matrices - a `6x2` matrix, `12` numbers in total, and another `2x8` matrix, making it `16` numbers. The total trained parameters have been reduced from `48` to `28`. This is why the LoRA file can be so small.

So, the overall idea of merging LoRA weights to the checkpoint model works like this:

1. Find the $A$ and $B$ weight matrices from the LoRA file
2. Match the LoRA module layer name to the model's module layer name so that we know which matrix to patch
3. Produce $\Delta W = AB^T$
4. Update the model weights

### LoRA in practice

To utilize LoRA, we can leverage the `load_lora_weights` method from `StableDiffusionPipeline`. The example below demonstrates how to apply two LoRA filters. The `adapter_weights` parameter determines the extent to which the LoRA model's "style" influences the output.

```python
# LoRA fine tuning

pipeline.to("mps")

pipeline.load_lora_weights(
    "andrewzhu/MoXinV1",
    weight_name = "MoXinV1.safetensors",
    adapter_name = "MoXinV1",
    cache_dir = cache_dir
)

pipeline.load_lora_weights(
    "andrewzhu/civitai-light-shadow-lora",
    weight_name = "light_and_shadow.safetensors",
    adapter_name = "light_and_shadow",
    cache_dir = cache_dir
)

pipeline.set_adapters(
    ["MoXinV1", "light_and_shadow"],
    adapter_weights = [0.5, 1.0]
)
```
The images below showcase a Chinese painting generated using the Stable Diffusion 1.5 model. The middle image enhances realism, making it look like an authentic Chinese painting. Meanwhile, the second LoRA model introduces more vibrant colors, transforming it into a different artistic style.

<div class="md-flex-h md-flex-no-wrap">
<div><img src="{{site.baseurl}}/assets/images/2025/01/sd-lora-base.png"></div>
<div class="md-margin-left-6"><img src="{{site.baseurl}}/assets/images/2025/01/sd-lora-1.png"></div>
<div class="md-margin-left-6"><img src="{{site.baseurl}}/assets/images/2025/01/sd-lora-2.png"></div>
</div>


## Textual Inversion

Textual Inversion(TI) is another way to fine tune the pretrained model. Unlike LoRA, <mark>TI is a technique to add new embedding space based on the trained data</mark>. Simply put, TI is a text embedding that matches the target image the best, such as its style, object, or face. The key is to find the new embedding that does not exist in the current text encoder.

### How does TI works

To train a TI model, you only need a minimal set of three to five images, resulting in a compact `pt` or `bin` file, typically just a few kilobytes in size. This makes TI a highly efficient method for incorporating new elements or styles into the image. 

<img class="md-img-center" src="{{site.baseurl}}/assets/images/2025/01/sd-03-01.png">

The goal of training is to find a new embedding represented by $v_*$. We use $S_*$ as the token string placeholder to represent the new concepts we wish to learn. We aim to find a single word embedding, such that sentences of the form "A photo of S*" will lead to the reconstruction of images from our small training set. This embedding is found through an optimization process shown in the above figure, which we refer to as <mark>"Textual Inversion"</mark>. 

Let's recall the loss function we use to train the stable diffusion model:

$$
L_{LDM} := \mathbb{E}_{z \sim \mathcal{E}(x), y, \epsilon \sim \mathcal{N}(0,1), t} 
\left[ \left\| \epsilon - \epsilon_{\theta}(z_t, t, c_{\theta}(y)) \right\|_2^2 \right],
$$


Once the new corresponding embedding vector is found, the training is done. The output of the training is usually a vector with 768 numbers. That is why TI file is tiny.




## Resources

- [An Image is Worth One Word: Personalizing Text-to-Image Generation using Textual Inversion](https://arxiv.org/abs/2208.01618)
