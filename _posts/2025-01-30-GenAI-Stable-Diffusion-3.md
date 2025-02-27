---
list_title: GenAI | Use LoRA for fine tunning
title: Use LoRA for fine tunning
layout: post
mathjax: true
categories: ["GenAI", "Stable Diffusion"]
---

## How LoRA works at high-level

[In previous articles](https://xta0.me/2024/11/24/GenAI-LLM-2.html), we have explored LoRA as a strategy for fine-tuning LLMs. With LoRA, the original checking is frozen without any modification, and the tuned weight changes are stored in an independent file, which is referred as LoRA file. 

LoRA works by creating a small, low-rank model that is adapted for a specific concept. This small model can be merged with the main checkpoint model to generate images during the inference stage.

Let's use $W$ to represent the original UNet attention weights(`Q`, `K`, `V`), $\delta W$ to denote the fine-tuned weights from LoRA, and $W'$ as the combined weights. The process of adding LoRA to a model can be expressed as:

$$
W' = W + \delta W
$$