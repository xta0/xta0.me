---
updated: '2017-11-30'
list_title: 深度学习 | Logistic Regression as a Neural Network
title: Logistic Regression as a Neural Network
layout: post
mathjax: true
categories: ["AI", "Machine Learning","Deep Learning"]
---

> 部分图片截取自课程视频[Nerual Networks and Deep Learning](https://www.coursera.org/learn/neural-networks-deep-learning)

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

有了Cost funciton之后，我们就可以使用梯度下降来求解$w$和$b$，使$J(w,b)$最小。梯度下降的计算方式如下

$$
w := w - \alpha\frac{dJ(w,b)}{dw} \\
b := b - \alpha\frac{dJ(w,b)}{db} 
$$

上述式子通过不断的对$w$和$b$进行求偏导，最终使其收敛为一个稳定的值，其中$\alpha$为Learning Rate,用来控制梯度下降的幅度。在后面的Python代码中，使用`dw`表示 $\frac{dJ(w,b)}{dw}$，`db`表示$\frac{dJ(w,b)}{db}$，以此类推。

虽然我们有了上面的算式，但如何有效的计算它是我们接下来要讨论的问题，这里我们介绍一种使用**Computation Graph**的思路，所谓的Computation Graph的概念，基本思想是将每一步运算都用一个节点表示，然后将这些节点串联起来得到一个Graph，举例来说，假设有一个函数为

$$
J(a,b,c) = 3(a+bc)
$$

我们另

- $u = bc$
- $v = a+u$
- $J = 3v$

则该算式的Computation Graph可以表示如下

<img src="{{site.baseurl}}/assets/images/2018/01/dp-w2-1.png" class="md-img-center">

接下来我们要思考如何对Graph中的每一项进行求导，这将是后面计算神经网络backpropagation的基础。显然如果有微积分基础的话，这并不难

- $\frac{dJ}{dv} = 3$
- $\frac{dJ}{du} = \frac{dJ}{dv} \times \frac{dv}{du} = 3 \times 1 = 3$
- $\frac{dJ}{da} = \frac{dJ}{dv} \times \frac{dv}{da} = 3 \times 1 = 3$
- $\frac{dJ}{db} = \frac{dJ}{dv} \times \frac{dv}{du} \times \frac{du}{db} = 3 \times 1 \times c = 3c$
- $\frac{dJ}{dc} = \frac{dJ}{dv} \times \frac{dv}{du} \times \frac{du}{dc} = 3 \times 1 \times b = 3b$

在接下来的代码中，我们需要表示上面的每个导数值，其表示方式为

$$
\frac{dFinalOutputVar}{dvar}
$$

这种写法太过冗余，因此，如果想表示$\frac{dJ}{da}$，在代码中可直接写成`da`，其余同理，计算这些变量到数值的过程，可类比于神经网络的backpropagation

<img src="{{site.baseurl}}/assets/images/2018/01/dp-w2-2.png" class="md-img-center">

理解了Computation Graph，我们回到算梯度下降上来，假设我们只有两组训练集$x^{(1)}$和$x^{(2)}$，我们可以随机给出两个$w$矩阵，$w^{(1)}$和$w^{(2)}$，以及一个$b$矩阵，则逻辑回归model的Graph如下

<img src="{{site.baseurl}}/assets/images/2018/01/dp-w2-3.png" class="md-img-center">

我们最终目标是求解$w^{(1)}$和$w^{(2)}$和$b$，使Loss函数的值最小，根据梯度下降的公式，

$$
w^{(1)} := w^{(1)} - \alpha\frac{dL(a^{(1)},y^{(i)})}{dw^{(1)}} \\
w^{(2)} := w^{(2)} - \alpha\frac{dL(a^{(2)},y^{(i)})}{dw^{(2)}} \\
b := b - \alpha\frac{dL(a,b)}{db} 
$$

接下来利用前面提到的求偏导的方式，一步步反向计算得到 $w^({1})$ 和 $w^({2})$的最终值，如下图所示

> 注意这里的$w^({1})$, $w^({2})$, $b$ 均为矩阵

<img src="{{site.baseurl}}/assets/images/2018/01/dp-w2-4.png" class="md-img-center">

- $da = \frac {dL(a,y)} {da} = - \frac{y}{a} + \frac{1-y}{1-a}$
- $dz = \frac {dL(a,y)} {da} = \frac {dL(a,y)}{da} \times \frac {da}{dz} = a-y$
- $dw_1 = \frac {dL(a,y)} {dw_1} = x_1 \times dz$ 
- $dw_2 = \frac {dL(a,y)} {dw_2} = x_2 \times dz$ 
- $db = \frac {dL(a,y)} {db} = dz$ 

因此上述梯度下降公式，最终可以表示为

$$
w^{(1)} := w^{(1)} - \alphadw^{(1)} \\
w^{(2)} := w^{(2)} - \alphadw^{(2)} \\
b := b - \alphadb
$$


## Vectorization 

对于上面$z$的计算，涉及到矩阵相乘和相加，通常的计算方式是使用两层for循环，根据矩阵乘法和加法的定义完成计算。但这种实现方式效率不高，我们可以使用向量化的方式，使用向量化的好处是对于矩阵运算可以显著的提升计算效率，numpy提供了方便的API可以取代for循环而进行矩阵的数值运算

```python
z = np.dot(w.T,x)+b
```
