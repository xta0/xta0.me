---
list_title: 深度学习 | Neural Networks Overview
title: Neural Networks Overview
layout: post
mathjax: true
categories: ["AI", "Machine Learning","Deep Learning"]
---

### Notations

- $x^{(i)}$：表示第$i$组训练样本
- $x^{(i)}_j$：表示第$i$组训练样本的第$j$个feature
- $a^{[l]}$：表示第$l$层神经网络
- $a^{[l]}_i$: 表示第$l$层神经网络的第$i$个节点
- $a^{[l](m)}_i$：表示第$m$个训练样本的第$l$层神经网络的第$i$个节点

遵循上述的Notation，一个两层的神经网络可用下图描述

<img src="{{site.baseurl}}/assets/images/2018/01/dp-w3-1.png" class="md-img-center">

将上述式子用向量表示，则对于给定的输入$x$，有

$$
z^{[1]} = W^{[1]}x + b^{[1]} \\
a^{[1]} = \sigma(z^{[1]}) \\ 
z^{[2]} = W^{[2]}x + b^{[2]} \\
a^{[2]} = \sigma(z^{[2]}) 
$$

- $z^{[1]}$是`4x1`
- $W^{[1]}$是`4x3`
- $x$是`3x1`
- $b^{[1]}$是`4x1`
- $a^{[1]}$是`4x1`
- $z^{[2]}$是`1x1`
- $W^{[2]}$是`1x4`
- $a^{[1]}$是`1x1`
- $b^{[2]}$是`1x1`

上述神经网络只有一个组训练集，如果将训练集扩展到多组($x^{(1)}$,$x^{(2)}$,...,$x^{(m)}$，则有下面式子成立

<img src="{{site.baseurl}}/assets/images/2018/01/dp-w3-2.png" class="md-img-center">
