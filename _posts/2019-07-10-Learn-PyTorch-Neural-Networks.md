---
list_title: 学一点 PyTorch | Learn PyTorch | 神经网络 | Neural Networks with PyTorch
title: PyTorch实现神经网络
layout: post
mathjax: true
categories: ["PyTorch", "Machine Learning","Deep Learning"]
---

### 从线性回归到神经网络

上一篇文章中我们用PyTorch实现了一个线性回归的model，这篇文章我们将用神经网络来代替线性回归，重新训练我们的模型。虽然我们只有一个feature和11个训练样本，使用神经网络不免有些OverKill了，但神经网络的一个有趣之处是我们不知道我们最后拟合出的模型到底是什么样的，所以我们