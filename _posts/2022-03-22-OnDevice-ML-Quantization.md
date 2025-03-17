---
list_title: On-Device ML | Quantization in PyTorch
title: Quantization in PyTorch
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

## Linear Quantization

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

The formula for calculating $q$:

$$
q = int(round(r/s+z))
$$

PyTorch's implementation:

```python
import torch

def linear_q_with_scale_and_zero_point(
    tensor, scale, zero_point, dtype = torch.int8):
    scaled_and_shifted_tensor = tensor / scale + zero_point
    rounded_tensor = torch.round(scaled_and_shifted_tensor)

    q_min = torch.iinfo(dtype).min # -128
    q_max = torch.iinfo(dtype).max # 128
    q_tensor = rounded_tensor.clamp(q_min,q_max).to(dtype)
    
    return q_tensor

def linear_dequantization(quantized_tensor, scale, zero_point):
    return scale * (quantized_tensor.float() - zero_point)
```

Note that if we pick up a random `scale` and `zero_point` number, the quantized error could be really high. So how do we determine $s$ and $z$? If we look at the **extreme values** for $[r_{min}, r_{max}]$ and $[q_{min}, q_{max}]$, we should have:

$$
\begin{align*}
r_{\min} = s \left( q_{\min} - z \right) \\
r_{\max} = s \left( q_{\max} - z \right)
\end{align*}
$$

If we subtract the first equation from the second one, we get the `scale`:

$$
s = (r_{max} - r_{min}) / (q_{max} - q_{min})
$$

For the `zero point`, we need to round the value since it is a n-bit integer

$$
z = int(round(q_{min} - r_{min}/s))
$$

```python
def get_q_scale_and_zero_point(tensor, dtype=torch.int8):
    q_min, q_max = torch.iinfo(dtype).min, torch.iinfo(dtype).max
    r_min, r_max = tensor.min().item(), tensor.max().item()

    scale = (r_max - r_min) / (q_max - q_min)
    zero_point = q_min - (r_min / scale)

    # clip the zero_point to fall in [quantized_min, quantized_max]
    if zero_point < q_min:
        zero_point = q_min
    elif zero_point > q_max:
        zero_point = q_max
    else:
        # round and cast to int
        zero_point = int(round(zero_point))
    
    return scale, zero_point
```

To put everything together, we now have one single function to quantize a tensor:

```python
def linear_quantization(tensor, dtype=torch.int8):
    scale, zero_point = get_q_scale_and_zero_point(tensor, dtype=dtype)
    quantized_tensor = linear_q_with_scale_and_zero_point(
        tensor,
        scale, 
        zero_point, 
        dtype=dtype)
    
    return quantized_tensor, scale , zero_point
```

### Symmetric vs Asymmetric mode

There are two modes in linear quantization:

- Asymmetric: map $[r_{min}, r_{max}]$ to [$q_{min}$, $q_{max}$]. This is what we implemented in the previous section.
- Symmetric: map $[-r_{max}, r_{max}]$ to [$-q_{max}$, $q_{max}$], where $r_{max} = max(\|tensor(r)\|)$

In the symmetric mode, we don't need to use the zero point(`z = 0`). This is because the floating-point range and the quantized range are symmetric with respect to zero.

<img class="md-img-left" src="{{site.baseurl}}/assets/images/2022/03/quant-5.png">

Hence, we can simplify the equation to 

$$
\begin{array}{l}
q = int(round(r/s)) \\
s =  r_{\max} / {q_{\max}}
\end{array}
$$

PyTorch's implementation for calculating `scale`:

```python
def get_q_scale_symmetric(tensor, dtype=torch.int8):
    r_max = tensor.abs().max().item()
    q_max = torch.iinfo(dtype).max

    # return the scale
    return r_max/q_max
```

Once we have the `scale` value calculated, we can reuse the function in the asymmetric mode to get the quantized tensor. 

```python
def linear_quantization(tensor, dtype=torch.int8):
    scale = get_q_scale_symmetric(tensor, dtype=dtype)
    quantized_tensor = linear_q_with_scale_and_zero_point(
        tensor,
        scale, 
        0,  # zero_point = 0
        dtype=dtype)
    
    return quantized_tensor, scale
```

The trade-off between symmetric and asymmetric mode of quantization are:

- **Utilization of quantized range**:
    - When using asymmetric quantization, the quantized range is fully utilized.
    - When symmetric mode, if the float range is biased towards one side, this will result in a quantized range where a part of the range is dedicated to values that we'll never see. (e.g., ReLU where the output is positive).
- **Simplicity**: Symmetric mode is much simpler compared to asymmetric mode.
- **Memory**: We don’t store the zero-point for symmetric quantization.
