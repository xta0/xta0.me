---
list_title: On-Device ML | Quantization in PyTorch
title: Quantization
layout: post
mathjax: true
categories: ["AI", "Deep Learning", "On-Device ML"]
---

## Introduction

Quantization is a widely used model compression technique that reduces the memory footprint and computational costs of deep learning models by storing numerical parameters in lower precision, such as 8-bit integers (`INT8`) instead of 32-bit floating points (`FP32`). This process enables faster inference, lower power consumption, and reduced storage requirements, making it particularly useful for deployment on edge devices, mobile phones, and embedded systems.

Quantization can be categorized based on its granularity, which determines how the model's parameters are quantized. One common approach is **linear quantization**, where values are mapped to discrete levels using a fixed scaling factor and zero point. This ensures that floating-point numbers are effectively represented in a lower precision format with minimal information loss.

<img class="md-img-center" src="{{site.baseurl}}/assets/images/2022/03/quant-1.png">

On top of the linear quantization, there are several options for determining the granularity of quantization, based on how much of the model you want to quantize at a time. These options include:

**Per-tensor quantization**: A single scaling factor and zero point are used for the entire tensor, making it simple and efficient but potentially less precise for models with varying parameter distributions across channels.

**Per-channel quantization**: Each channel (e.g., convolutional filter) is assigned a unique scaling factor and zero point, allowing for better accuracy by accounting for variations in parameter distributions across channels.

**Per-group quantization**: Parameters are divided into smaller groups, and each group has its own scaling factor and zero point. This method balances efficiency and precision, providing more flexibility than per-tensor quantization while reducing the computational overhead compared to per-channel quantization.

Each of these quantization methods serves different purposes depending on the model’s structure and deployment constraints. While per-tensor quantization is the fastest and simplest to implement, per-channel and per-group quantization generally offer higher accuracy by preserving more information about the original parameter distribution.

### Linear Quantization

Quantization refers to the process of mapping a large set to a smaller set of values. For example, in an 8-bit quantization, a float number(`torch.float32`) will be mapped to an 8-bit integer (`torch.int8`) in the memory.

<img class="md-img-center" src="{{site.baseurl}}/assets/images/2022/03/quant-2.png">