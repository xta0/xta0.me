---
list_title: 笔记 | 深度学习 | Hperparameters
title: Hperparameters
layout: post
mathjax: true
categories: ["AI", "Machine Learning","Deep Learning"]
---

### Bias and Variance

如果我们在train data上面的error很低，但是在Dev set上的error很高，说明我们的模型出现over fitting，这种情况下我们说模型的**Variance**很高。如果二者错误率接近，且都很高，这是我们称为**high bias**，此时我们的模型的问题是under fitting。解决high bias可以引入更多的hidden layer来增加training的时间，或者使用一些的优化方法，后面会提到。如果要解决over fitting的问题，我们则需要更多的数据或者使用Regularization。

### Regularization

我们还是用Logistic Regression来举例。在LR中，Cost Function定义为

$$
J(w,b) = \frac{1}{m}\sum_{i=1}^{m}L(\hat{y}^{(i)}, y^{(i)})
$$

为了解决Overfitting，我们可以在上面式子的末尾增加一个Regularization项