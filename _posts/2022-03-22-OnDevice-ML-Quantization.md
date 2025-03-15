---
list_title: On-Device ML | Quantization in PyTorch
title: Quantization
layout: post
mathjax: true
categories: ["AI", "Deep Learning", "On-Device ML"]
---

## Introduction

Quantization refers to the process of mapping a large set to a smaller set of values. It is a widely used model compression technique that reduces the memory footprint and computational costs of deep learning models. For example, in an 8-bit quantization process, a 32-bit float number(`torch.float32`) will be mapped to an 8-bit integer (`torch.int8`) in the memory.

<div style="display: block; width: 50%;">
<img class="md-img-center" src="{{site.baseurl}}/assets/images/2022/03/quant-2.png">
</div>

In a typical neural network, Quantization can be applied to quantize the **model weights** and the **activations**. When quantization is applied after the model has been fully trained, it is referred to as **post-training quantization (PTQ)**.

Advantages of quantization includes:

- Smaller model
- Speed gains
    - Memory bandwidth
    - Faster operations
        - GEMM: matrix to matrix multiplication
        - GEMV: matrix to vector multiplication

### Linear Quantization

**Linear Quantization** uses a linear mapping to map a higher precision range(e.g. float32) to a lower precision range(e.g. int8) using <mark>a fixed scaling factor and zero point</mark>. This ensures that floating-point numbers are effectively represented in a lower precision format with minimal information loss.

<div style="display: block; width: 50%;">
<img class="md-img-center" src="{{site.baseurl}}/assets/images/2022/03/quant-3.png">
</div>

The linear mapping formula can be described as:

$$
r = s(q - z)
$$

Where $r$ is the original value(e.g. `fp32`), $q$ is the quantized value(e.g. `int8`), $s$ is scale and $z$ is the zero point. For example, with $s=2$ and $z=0$, we get $r = 2(q-0) = 2q$. If $q = 10$, then we have $r = 2 \times 10 = 20$.

<div style="display: block; width: 50%;">
<img class="md-img-center" src="{{site.baseurl}}/assets/images/2022/03/quant-4.png">
</div>