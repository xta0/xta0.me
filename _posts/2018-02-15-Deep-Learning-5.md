---
list_title:   Deep Learning | Hyperparameter Tuning and Optimization Algorithms
title: Hyperparameter Tuning and Optimization Algorithms
layout: post
mathjax: true
categories: ["AI", "Machine Learning","Deep Learning"]
---

## Mini-batch Gradient Descent

假设一组训练数据$X$有$m$个样本，$X = [x^{(1)},x^{(2)},x^{(3)},...,x^{(m)}]$，每个$x^{[i]}$有n个feature，则$X$矩阵为$(n, m)$，相应的，$Y=[y^{(1)},y^{(2)},y^{(3)},...,y^{(m)}]$。对于m组数据，在training时，我们可以用vectorization的方式代替for循环，这样可以提高训练速度。

但是如果$m$非常大，比如5个million，那么将所有数据一起training，效率将非常低。实际上，在back prop的时候，我们不需要更新整个数据的weights和bias。我们可以将training set分成多个batch，比如将5个million切分成5000个小的batch，每个batch包含1000个训练数据。

使用Mini-batch gradient descent会影响training，表现为cost函数不会一直下降，而是不断变化，如下图所示

<img src="{{site.baseurl}}/assets/images/2018/02/dl-ht-08.png">

我们用 $X^{\{t\}}$ 表示一个batch的训练数据，则

- 当batch size为`m`的时候，称为**Batch Gradient Descent**，此时$(X^{{1}}, Y^{{1}}) = (X, Y)$
- 当batch size为`1`的时候，称为**Stochastic Gradient Descent**，此时$(X^{{1}}, Y^{{1}}) = (x^{(1)}, y^{(1)}), ..., (X^{\{t\}}, Y^{\{t\}}) = (x^{(t)}, y^{(t)})$

注意，如果使用SGD，梯度下降的过程将极为noise，并不会一直沿着梯度下降最大的方向前进，也不会收敛于某个值，而是在某个区域内不断变化。

在实际应用中，batch size往往在`(1, m)`中选取。三种方式的梯度下降过程如下图所示

<img src="{{site.baseurl}}/assets/images/2018/02/dl-ht-09-1.png">
<img src="{{site.baseurl}}/assets/images/2018/02/dl-ht-09-2.png">

小结一下

1. 如果训练样本很小(`m<2000`)，使用Batch Gradient Descent
2. 如果使用Mini Batch, `m`可选取2的阶乘，比如`64`, `128`, `256`, `512`, `1024`
3. 注意单个batch所产生的运算量(forward和backward)是否能被加载到当前的内存中

## Gradient Descent with momentum

Exponentially weighted averages是移动平均的意思，计算方式如下

$$
v_t = \beta v_{t-1} + (1-\beta) \theta_t
$$

$\beta$在0和1之间，当$\beta$越大，曲线平滑，平均的样本数越多，反之曲线波动大，平均样本越少，如下图所示，分别是$\beta$为0.98，0.9和0.5时的曲线



## Adam optimization algorithm

## Resources 

- [An overview of gradient descent optimization algorithms](https://ruder.io/optimizing-gradient-descent/)