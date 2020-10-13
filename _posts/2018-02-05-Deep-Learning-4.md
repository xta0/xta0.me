---
list_title: 笔记 | 深度学习 | Hperparameters Tuning
title: Hperparameters
layout: post
mathjax: true
categories: ["AI", "Machine Learning","Deep Learning"]
---

## Bias and Variance

如果我们在train data上面的error很低，但是在Dev set上的error很高，说明我们的模型出现over fitting，这种情况下我们说模型的**Variance**很高。如果二者错误率接近，且都很高，这是我们称为**high bias**，此时我们的模型的问题是under fitting。解决high bias可以引入更多的hidden layer来增加training的时间，或者使用一些的优化方法，后面会提到。如果要解决over fitting的问题，我们则需要更多的数据或者使用Regularization。

## Regularization

我们还是用Logistic Regression来举例。在LR中，Cost Function定义为

$$
J(w,b) = \frac{1}{m}\sum_{i=1}^{m}L(\hat{y}^{(i)}, y^{(i)})
$$

为了解决Overfitting，我们可以在上面式子的末尾增加一个Regularization项

$$
J(w,b) = \frac{1}{m}\sum_{i=1}^{m}L(\hat{y}^{(i)}, y^{(i)}) + \frac{\lambda}{2m}||{w}||{^2}
$$

上述公式末尾的二范数也称作L2 Regularization，其中$\omega$的二范数定义为

$$
||{w}||{^2} = \sum_{j=1}^{n_x}\omega^{2}_{j} = \omega^T\omega
$$

除了使用L2范数外，也有些model使用L1范数，即

$$
\frac{\lambda}{2m}||\omega||_1
$$。

如果使用L1范数，得到的$\omega$矩阵会较为稀疏（0很多），不常用。

对于一般的Neural Network，Cost Function定义为

$$
J(\omega^{[1]}, b^{[1]},...,\omega^{[l]}, b^{[l]}) = \frac{1}{m}\sum_{i=1}^{m}L(\hat{y^{(i)}}, y^{(i)}) + \frac{\lambda}{2m}||{\omega}||{^2}
$$

其中对于某$l$层的L2范数计算方法为

$$
||\omega^{[l]}||^{(2)} = \sum_{i=1}^{n^{l}}\sum_{j=1}^{n^{(l-1)}}(\omega_{i,j}^{[l]})^2
$$

其中$i$表示row，$n^{l}$表示当前层有多少neuron(输出)，$j$表示column，$n^{(l-1)}$为前一层的输入有多少个neuron。简单理解，上面的L2范数就是对权重矩阵中的每个元素平方后求和。

新增的Regularization项同样会对反响求导产生影响，我们同样需要增加改Regularization项对$\omega$的导数

$$
d\omega = (from backprop) + \frac{\lambda}{m}\omega 
$$

Gradient Descent的公式不变

$$
\omega^{[l]} := \omega^{[l]} - \alpha d\omega^{[l]} 
$$



