---
list_title: 笔记 | 深度学习 | Hperparameters
title: Hperparameters
layout: post
mathjax: true
categories: ["AI", "Machine Learning","Deep Learning"]
---

- Bias and Variance

如果我们的train set上面的error很低，但是在Dev set上的error很高，说明我们的模型出现over fitting，这种情况下我们说模型的**Variance**很高。如果二者错误率接近，且都很高，这是我们称为**high bias**，这是我们的模型的问题是under fitting。

解决high bias可以引入更多的hidden layer