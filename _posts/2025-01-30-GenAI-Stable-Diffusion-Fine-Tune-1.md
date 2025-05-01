---
list_title: Stable Diffusion | Fine-Tuning | LoRA
title: LoRA Fine-Tuning
layout: post
mathjax: true
categories: ["GenAI", "LoRA", "Stable Diffusion"]
---

## How LoRA Works

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

### The benefits of LoRA

- **Reduced resource consumption**. Fine-tuning deep learning models typically requires substantial computational resources, which can be expensive and time-consuming. LoRA reduces the demand for resources while maintaining high performance.

- **Faster iterations**. LoRA enables rapid iterations, making it easier to experiment with different fine-tuning tasks and adapt models quickly.

- **Improved transfer learning**. LoRA enhances the effectiveness of transfer learning, as models with LoRA adapters can be fine-tuned with fewer data. This is particularly valuable in situations where labeled data are scarce.

### LoRA in practice


### LoRA in Stable Diffusion

To utilize LoRA, we can leverage the `StableDiffusionPipeline` method from [HuggingFace's diffuser package](https://huggingface.co/docs/diffusers/index). The example below demonstrates how to apply two LoRA filters. The `adapter_weights` parameter determines the extent to which the LoRA model's "style" influences the output.

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

## Resources

- [Using Stable Diffusion with Python](https://www.amazon.com/Using-Stable-Diffusion-Python-Generation/dp/1835086373/)