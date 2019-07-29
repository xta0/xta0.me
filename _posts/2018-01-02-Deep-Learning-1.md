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

其中$\hat{y}$, $x$, $w$ 和 $b$ 皆为矩阵，$\sigma$为sigmoid函数，定义为$\sigma(z) = \frac{1}{1+e^{-z}}$。我们假设第$i$组训练样本用向量$x^{(i)}$表示，则$x^{(i)}$是$nx1$的（注意是n行1列），那么假设一共有$m$组训练样本，那么$x$矩阵表示为

$$
x= 
\begin{bmatrix}
. & . & . & . & . & . & . \\
. & . & . & . & . & . & . \\
x^{(1)} & x^{(2)} & x^{(3)} & . & . & . & x^{(m)} \\
. & . & . & . & . & . & . \\
. & . & . & . & . & . & . \\
\end{bmatrix}
$$

类似的$w$是一个$n x 1$的向量，则$w^Tx$是 $1xm$, 对应的常数项$b$也是$1xm$的矩阵，另

$$
z^{(i)} = w^Tx^{(i)} + b \\
a^{(i)} = \sigma(z^{(i)}) 
$$

则$\hat{y}$可以表示为

$$
\hat{y} = 
\begin{bmatrix}
z^{(1)} & z^{(2)} & . & . & . &z^{(m)}  
\end{bmatrix}
= w^{T}x + 
\begin{bmatrix}
b_1 & b_2 & . & . & . &b_n
\end{bmatrix}
$$

因此预测结果$\hat{y}$为$1 \times m$的向量

### Cost function

对于某组训练集可知其Loss Function为

$$
L(\hat(y),y) = - (y\log{\hat{y}}) + (1-y)\log{(1-\hat{y})} 
$$

我们将所有训练样本一起计算，则可以得到Cost function

$$
J(w,b) = \frac{1}{m}\sum_{i=1}^{m}L(\hat{y}^{(i)}, y^{(i)}) = -\frac{1}{m}\sum_{i=1}^{m}[(y^{(i)}\log{\hat{y}^{(i)}}) + (1-y^{(i)})\log{(1-\hat{y}^{(i)})} ]
$$

### Gradient Descent

有了Cost funciton之后，我们就可以使用梯度下降来求解$w$和$b$，使$J(w,b)$最小，梯度下降的计算方式如下

$$
Repeat \{ \\
    w  := w - \alpha\frac{dJ(w)}{dw} \\
\}
$$

其中$\alpha$为Learning Rate,用来控制梯度下降的幅度，

> 在后面的Python代码中，使用`dw`表示 $\frac{dJ(w)}{dw}$


### Neural Network

## Computing Graph

## Backpropagation

## Vectorization 

对于上面$z$的计算，涉及到矩阵相乘和相加，通常的计算方式是使用两层for循环，根据矩阵乘法和加法的定义完成计算。但这种实现方式效率不高，我们可以使用向量化的方式，使用向量化的好处是对于矩阵运算可以显著的提升计算效率，numpy提供了方便的API可以取代for循环而进行矩阵的数值运算

```python
z = np.dot(w.T,x)+b
```
