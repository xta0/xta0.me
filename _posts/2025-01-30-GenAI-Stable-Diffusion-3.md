---
list_title: GenAI | Use LoRA for fine tunning
title: Use LoRA for fine tunning
layout: post
mathjax: true
categories: ["GenAI", "Stable Diffusion"]
---

## How LoRA works at high-level

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