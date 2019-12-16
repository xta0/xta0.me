---
list_title: 学一点PyTorch
title: 学一点PyTorch
layout: post
mathjax: true
categories: ["AI", "Machine Learning","Deep Learning"]
---

PyTorch是Facebook开源的一套Deep Learning的框架，它的API都是基于Python的，因此对Researcher非常友好。我对PyTorch的理解是它是具有自动求导功能的Numpy，当然PyTorch比Numpy肯定要强大的多。由于PyTorch目前仍在快速的迭代中，并且有着愈演愈烈的趋势，我们今天也来凑凑热闹，学一点PyTorch。

### Linear Regression问题

我们使用的例子是一个很简单的[线性回归模型]()，假设我们有一组观测数据如下

```python
t_y = [0.5, 14.0, 15.0, 28.0, 11.0, 8.0, 3.0, -4.0, 6.0, 13.0, 21.0]
t_x = [35.7, 55.9, 58.2, 81.9, 56.3, 48.9, 33.9, 21.8, 48.4, 60.4, 68.4]
```
我们的目标是通过PyTorch帮我们学习得到$\omega$和$b$，使下面等式成立

$$
t_y = \omega \times t_x + b
$$

实际上就是一个对离散点进行线性拟合的问题。

我们首先来创建两个tensor

```python
import torch

t_y = torch.tensor(t_y)
t_x = torch.tensor(t_x)
```