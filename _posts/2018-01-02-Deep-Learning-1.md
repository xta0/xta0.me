---
updated: '2017-11-30'
list_title: 深度学习 | Logistic Regression as a Neural Network
title: Logistic Regression as a Neural Network
layout: post
mathjax: true
categories: ["AI", "Machine Learning","Deep Learning"]
---

### Logistic Regression Recap

前面机器学习的课程中，我们曾[介绍过Logistic Regression的概念](https://xta0.me/2017/09/27/Machine-Learning-3.html)，它主要用来解决分类的问题，尤其是True和False的分类。我们可以将逻辑回归模型理解为一种简单的神经网络，`F(x) = {0 or 1}`，如果我们将重点放在线性的逻辑回模型上，则数学表达为

$$
\hat{y} = \sigma(w^Tx + b)
$$

其中$\hat{y}$, $x$, $w$ 和 $b$ 皆为矩阵，$\sigma$为sigmoid函数，定义为$\sigma(z) = \frac{1}{1+e^{-z}}$。我们假设每一组训练样本用$x$向量表示，则$x$是nx1的（注意是n行1列），那么假设一共有m组训练样本，那么$x$矩阵表示为

$$
x= 
\begin{bmatrix}
. . . . . . . \\
. . . . . . . \\
x^{(1)} x^{(2)} x^{(3)} ... x^{(m)} \\
. . . . . . . \\
. . . . . . . \\
\end{bmatrix}
$$

### Vectorization 

对于`z`的计算，我们可以使用向量化的方式，使用向量化的好处是对于矩阵运算可以显著的提升计算效率，numpy提供了方便的API可以取代for循环而进行矩阵的数值运算

$$
z^{(1)} = W^Tx^{(1)}+b
$$

## Cost function

## Computing Graph

## Backpropagation
