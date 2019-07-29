---
updated: '2017-11-30'
list_title: Neural Networks and Deep Learning | 深度学习 
title: 深度学习
layout: post
mathjax: true
categories: ["AI", "Machine Learning","Deep Learning"]
---

## Logistic Regression Recap

前面机器学习的课程中，我们曾[介绍过Logistic Regression的概念](https://xta0.me/2017/09/27/Machine-Learning-3.html)，它主要用来解决分类的问题，尤其是True和False的分类。我们可以将逻辑回归模型理解为一种简单的神经网络，`F(x) = {0 or 1}`
- Sigmoid Function []
- Loss function

$$
\large \ell = \frac{1}{2n}\sum_i^n{\left(y_i - \hat{y}_i\right)^2}
$$

- Cost function
- Backpropagation


### Vectorization 

对于`z`的计算，我们可以使用向量化的方式，使用向量化的好处是对于矩阵运算可以显著的提升计算效率，numpy提供了


$$
z^{(1)} = W^Tx^{(1)}+b
$$