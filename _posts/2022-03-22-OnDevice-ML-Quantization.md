---
list_title: On-Device ML | Quantization in PyTorch
title: Quantization
layout: post
mathjax: true
categories: ["AI", "Deep Learning", "On-Device ML"]
---

## Introduction

Quantization refers to the process of mapping a large set to a smaller set of values. It is a widely used model compression technique that reduces the memory footprint and computational costs of deep learning models. For example, in an 8-bit quantization process, a 32-bit float number(`torch.float32`) will be mapped to an 8-bit integer (`torch.int8`) in the memory.

<img class="md-img-center" src="{{site.baseurl}}/assets/images/2022/03/quant-2.png" style="width: 80%;">

In a typical neural network, Quantization can be applied to

- **quantize the weights**: neural network parameters
- **quantize the activations**: values that propagate through the layers of the network

When quantization is applied after the model has been fully trained, it is referred to as **post-training quantization (PTQ)**.

Advantages of quantization includes:

- Smaller model
- Speed gains
    - Memory bandwidth
    - Faster operations
        - GEMM: matrix to matrix multiplication
        - GEMV: matrix to vector multiplication

### Linear Quantization

**Linear Quantization** maps values to discrete levels using a fixed scaling factor and zero point. This ensures that floating-point numbers are effectively represented in a lower precision format with minimal information loss.

<img class="md-img-center" src="{{site.baseurl}}/assets/images/2022/03/quant-1.png">

The linear mapping formula can be described as:

$$
r = s(q - z)
$$

where $r$ is the original value(e.g. `fp32`), and $q$ is the quantized value(e.g. `int8`)


