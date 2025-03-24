---
list_title: On-Device ML | Quantization in PyTorch | Part 2
title: Build a custom quantizer
layout: post
mathjax: true
categories: ["AI", "Deep Learning", "Quantization", "On-Device ML"]
---

## Build an 8-bit quantizer

Our goal is to implement a `W8A16LinearLayer` class that stores 8-bit weights along with their corresponding scales. The `W8A16` means the weights are in 8-bit and the activation is in 16-bit half precision float number. Once defined, we will replace all instances of `torch.nn.Linear` with our custom W8A16LinearLayer. To get started, let’s take a look at the forward pass

```python
def w8_a16_forward(weight, input, scales, bias=None):
    """
    input size: 1x16
    weight size: 32x16 (output_features, input_features)
    scale size: 1x32
    bisa size: 1x32
    """
    # cast the weights from 8-bit to 16-bit
    casted_weights = weight.to(input.dtype)
    # symmetric quantization, z = 0
    output = F.linear(input, casted_weights) * scales
    # add the bias
    if bias is not None:
        output = output + bias
    return output
```
In above example, before calling the linear function, we need to first convert the int8 weights to the same `dtype` as the input tensor. In this case, the input tensor is a `[1, 16]` half precision(`fp16`) float tensor. After the linear function, the output tensor is a `[1, 32]` half precision tensor as well.

Next, we are going to implement the `init` method for our custom quantizer class:

```python
class W8A16LinearLayer(nn.Module):
    def __init__(self, 
                in_features,
                out_features, 
                bias=True, 
                dtype=torch.float32):
        super().__init__()

        self.register_buffer("int8_weights",
            torch.randint(
                -128, 127, 
                (out_features, in_features), 
                dtype=torch.int8
            )
        )
        
        self.register_buffer("scales", 
            torch.randn((out_features), dtype=dtype)
        )
        
        if bias:
            self.register_buffer("bias", 
                torch.randn((1, out_features), dtype=dtype)
            )
        else:
            self.bias = None

    def forward(self, input):
        return w8_a16_forward(
            self.int8_weights, 
            input, 
            self.scales, 
            self.bias)
```

Note that we use `register_buffer` to store the int8 weight tensor instead of `nn.Parameter`. This is because `nn.Parameter` would mark the weight as trainable, which isn't what we want. By default, `register_buffer` persists the weights into model's `state_dict`.

Now, we are ready to implement the `quantize` method:

```python
def quantize(self, weights):
    # w_fp32: [m, n]
    w_fp32 = weights.clone().to(torch.float32)
    # returns the max value per row
    # scales: [m]
    scales = w_fp32.abs().max(dim=-1).values / 127
    scales = scales.to(weights.dtype)
    # scales: [m, 1]
    scales = scales.unsqueeze(1)

    # apply per-channel linear quantization
    w_fp32 = torch.round(weights/scales).to(torch.int8)

    self.int8_weights = w_fp32
    self.scales = scales
```

Note that we first upscale the weights to `fp32` for stability purpose. Then we calculate the scale for each row of the weight tensor. This is due to per-channel quantization. Finally, we use the formula from the previous post to quantize the weight tensor.


## Replace module with Quantized Layers

Now that we have our custom quantizer, we are going to iterate over all linear modules in the original model and replace those with our quantized linear layer modules. 

```python
def replace_linear_with_target(module, 
                            target_class,  # replacement
                            module_name_to_exclude):
    for name, child in module.named_children():
        if isinstance(child, nn.Linear) and not \
        any([x == name for x in module_name_to_exclude]):
            old_bias = child.bias
            new_module = target_class(child.in_features, 
                                    child.out_features, 
                                    old_bias is not None, 
                                    child.weight.dtype)
            setattr(module, name, new_module)
            if old_bias is not None:
                getattr(module, name).bias = old_bias
        else:
            # Recursively call the function for nested modules
            replace_linear_with_target(
                child, target_class, module_name_to_exclude)
```

To test it, we create a dummy model with multiple linear layers

```python
class DummyModel(torch.nn.Module):
  def __init__(self):
    super().__init__()
    self.emb = torch.nn.Embedding(1, 1)
    # Try with bias
    self.linear_1 = nn.Linear(1, 1)
    # Try without bias
    self.linear_2 = nn.Linear(1, 1, bias=False)
    # Lm prediction head
    self.lm_head = nn.Linear(1, 1, bias=False)

model = DummyModel()
replace_linear_with_target(model, W8A16LinearLayer, ["lm_head"])
print(model)
```
If we look at the printed result, only the first two linear layers will be replaced by our custom linear:

```python
DummyModel(
  (emb): Embedding(1, 1)
  (linear_1): W8A16LinearLayer()
  (linear_2): W8A16LinearLayer()
  (lm_head): Linear(in_features=1, out_features=1, bias=False)
)
```
One downside of this approach is that in order to quantize the model, we have to load the model's entire weights into the memory and cast them to `fp32`. If we have a large model, this could consume a significant amount of memory.

## Weights Packing

In certain models, quantized weights can be represented using just 4 bits—or even 2 bits—making 8-bit storage unnecessarily wasteful. In this section, we’ll explore how to store and load int8 tensors using only 2 or 4 bits to save memory and space.

Consider the tensor below that stores 4 values that can be represented in 2-bit precision, but stored in 8-bit

```python
tensor = torch.tensor([1, 0, 3, 2], dtype = torch.uint8)
```
In memory, the tensor is represented as binaries:

```
00000001, 00000000, 00000011, 00000010
```
Note that the leading 6 `0`s are unnecessary, we can pack all these tensors into a single 8-bit value

```
# 10 11 00 01 (read from right to left, 1 0 3 2)
packad_tensor = torch.tensor([177], dtype = torch.uint8)
```

Obviously, storing the weights in a lower bit representation could save memory and space. The downside is

- The unpacked weight tensors need to be a shape with a multiple of 8 // nbits
- The weights need to be unpacked before performing an inference operation

Let's take a look at the PyTorch implementation

```python
def pack_weights(uint8tensor, bits):
    if uint8tensor.shape[0] * bits % 8 != 0:
        raise ValueError(f"The input shape needs to be a mutiple \
        of {8 / bits} - got {uint8tensor.shape[0]}")

    num_values = uint8tensor.shape[0] * bits // 8
    num_steps = 8 // bits
    unpacked_idx = 0
    # [0000 0000]
    packed_tensor = torch.zeros((num_values), dtype=torch.uint8)

    # 1. For every number in the uint8 tensor, shift left two times * j
    # 2. XOR with the packed_tensor
    for i in range(num_values):
        for j in range(num_steps):
            packed_tensor[i] |= uint8tensor[unpacked_idx] << (bits * j)
            unpacked_idx += 1
    return packed_tensor

def unpack_weights(uint8tensor, bits):
    num_values = uint8tensor.shape[0] * 8 // bits
    num_steps = 8 // bits
    unpacked_tensor = torch.zeros((num_values), dtype=torch.uint8)
    unpacked_idx = 0

    mask = 2 ** bits - 1
    for i in range(uint8tensor.shape[0]):
        for j in range(num_steps):
            unpacked_tensor[unpacked_idx] |= uint8tensor[i] >> (bits * j)
            unpacked_idx += 1

    unpacked_tensor &= mask
    return unpacked_tensor
```

## Recent SOTA quantization methods

- [LLM.int8 (only 8-bit) – Aug 2022 – *Dettmers et al.*](https://arxiv.org/abs/2208.07339)
- [GPTQ – Oct 2022 – *Frantar et al.*](https://arxiv.org/abs/2210.17323)
- [SmoothQuant – Nov 2022 – *Xiao et al.*](https://arxiv.org/abs/2211.10438)
- [QLoRA (only 4-bit) – May 2023 – *Dettmers et al.*](https://arxiv.org/abs/2305.14314)
- [AWQ – Jun 2023 – *Lin et al.*](https://arxiv.org/abs/2306.00978)
- [QuIP# (promising results for 2-bit) – Jul 2023 – *Tseng et al.*](https://arxiv.org/abs/2307.12345)
- [HQQ (promising results for 2-bit) – Nov 2023 – *Badri et al.*](https://arxiv.org/abs/2311.09876)
- [AQLM (promising results for 2-bit) – Feb 2024 – *Egiazarian et al.*](https://arxiv.org/abs/2402.01234)


## Resource

- [Quantization in Depth](https://learn.deeplearning.ai/courses/quantization-in-depth)

