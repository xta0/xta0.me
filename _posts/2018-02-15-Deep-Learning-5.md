---
list_title: 笔记 | 深度学习 | Optimization Algorithms
title: Optimization Algorithms
layout: post
mathjax: true
categories: ["AI", "Machine Learning","Deep Learning"]
---

## Mini-batch gradient descent

假设一组训练数据$X$有$m$个样本，$X = [x^{(1)},x^{(2)},x^{(3)},...,x^{(m)}]$，每个$x^{[i]}$有n个feature，则$X$矩阵为$(n, m)$，相应的，$Y=[y^{(1)},y^{(2)},y^{(3)},...,y^{(m)}]$。对于m组数据，在training时，我们可以用vectorization的方式代替for循环，这样可以提高训练速度。

但是如果$m$非常大，比如5个million，那么将所有数据一起training，效率将非常低。实际上，在back prop的时候，我们不需要更新整个数据的weights和bias。我们可以将training set分成多个batch，比如将5个million切分成5000个小的batch，每个batch包含1000个训练数据。

使用Mini-batch gradient descent会影响training，表现为cost函数不会一直下降，而是不断变化，如下图所示

<img src="{{site.baseurl}}/assets/images/2018/02/dp-ht-07.png">

我们用 $X^{{t}}$表示一个batch的训练数据，则

- 当batch size为`m`的时候，称为**Batch Gradient Descent**，此时$(X^{{1}}, Y^{{1}}) = (X, Y)$
- 当batch size为`1`的时候，称为**Stochastic Gradient Descent**，此时$(X^{{1}}, Y^{{1}}) = (x^{(1)}, y^{(1)}), ..., (X^{{t}}, Y^{{t}}) = (x^{(t)}, y^{(t)})$

注意，如果使用SGD，梯度下降的过程将极为noise，并不会一直沿着梯度下降最大的方向前进，也不会收敛于某个值，而是在某个区域内不断变化。

在实际应用中，batch size往往在`(1, m)`中选取。三种方式的梯度下降过程如下图所示


总的来说

1. 如果训练样本很小(`m<2000`)，使用Batch Gradient Descent
2. 如果使用Mini Batch, `m`可选取2的阶乘，比如64, 128, 256, 512, 1024
3. 注意单个batch所产生的运算量(forward和backward)是否能fit in当前的内存中

## Gradient descent with momentum

Exponentially weighted averages是移动平均的意思，计算方式如下

$$
v_t = \Betav_{t-1} + (1-\Beta)\Theta_t
$$

$\Beta$在0和1之间，当$\Beta$越大，曲线平滑，平均的样本数越多，反之曲线波动大，平均样本越少，如下图所示，分别是$\Beta$为0.98，0.9和0.5时的曲线



## Adam optimization algorithm

## Resources 

- [An overview of gradient descent optimization algorithms](https://ruder.io/optimizing-gradient-descent/)