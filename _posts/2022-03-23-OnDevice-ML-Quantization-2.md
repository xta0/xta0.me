---
list_title: On-Device ML | Quantization in PyTorch | Part 2
title: Build a custom quantizer
layout: post
mathjax: true
categories: ["AI", "Deep Learning", "Quantization", "On-Device ML"]
---

## Build an 8-bit quantizer

Our goal is to create a `W8A16LinearLayer` class to store 8-bit weights and scales. Then replace all `torch.nn.Linear` with our custom `W8A16LinearLayer`.