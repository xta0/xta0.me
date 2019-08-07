---
list_title: 深度学习 | Deep-Layer Neural Networks
title: Deep-Layer Neural Networks
layout: post
mathjax: true
categories: ["AI", "Machine Learning","Deep Learning"]
---

> 本文部分图片截取自课程视频[Nerual Networks and Deep Learning](https://www.coursera.org/learn/neural-networks-deep-learning)

### Notations

- $x^{(i)}$：表示第$i$组训练样本
- $x^{(i)}_j$：表示第$i$组训练样本的第$j$个feature
- $a^{[l]}$：表示第$l$层神经网络
- $a^{[l]}_i$: 表示第$l$层神经网络的第$i$个节点
- $a^{[l] (m)}_i$：表示第$m$个训练样本的第$l$层神经网络的第$i$个节点

遵循上述的Notation，一个只有一组训练样本的$(x_1, x_2, x_3)$的两层神经网络可用下图描述

<img src="{{site.baseurl}}/assets/images/2018/01/dp-w3-1.png" class="md-img-center">

将上述式子用向量表示，则对于给定的输入$x$，有

$$
\begin{align*}
& z^{[1]} = W^{[1]}x + b^{[1]} \\
& a^{[1]} = \sigma(z^{[1]}) \\ 
& z^{[2]} = W^{[2]}a^{[1]} + b^{[2]} \\
& a^{[2]} = \sigma(z^{[2]}) 
\end{align*}
$$

其中，$z^{[1]}$是`4x1`，$W^{[1]}$是`4x3`，$x$是`3x1`，$b^{[1]}$是`4x1`，$a^{[1]}$是`4x1`，$z^{[2]}$是`1x1`，$W^{[2]}$是`1x4`，$a^{[1]}$是`1x1`，$b^{[2]}$是`1x1`

### Forward Propagation

上述神经网络只有一个组训练集，如果将训练集扩展到多组($x^{(1)}$,$x^{(2)}$,...,$x^{(m)})$，则我们需要一个`for`循环来实现每组样本的神经网络计算，然后对它们进行求和

$$
\begin{align*}
& for\ i=1\ to\ m\ \{ \\
& \qquad z^{[1] (i)} = W^{[1] (i)}x^{(i)} + b^{[1]} \\
& \qquad a^{[1] (i)} = \sigma(z^{[1] (i)}) \\ 
& \qquad z^{[2] (i)} = W^{[2]}a^{[1] (i)} + b^{[2]} \\
& \qquad a^{[2] (i)} = \sigma(z^{[2] (i)}) \\
& \}
\end{align*}
$$

结合前面文章可知，我们可以用向量化计算来取代`for`循环，另

$$
X= 
\begin{bmatrix}
. & . & . & . & . \\
| & | & . & . & | \\
x^{(1)} & x^{(2)} & . & . & x^{(m)} \\
| & | & . & . & | \\
. & . & . & . & . \\
\end{bmatrix}
\quad
W^{[1]} = 
\begin{bmatrix}
. & - & -w^{[1]}- & - & . & \\
. & - & -w^{[1]}- & - & . & \\
. & - & -w^{[1]}- & - & . & \\
. & - & -w^{[1]}- & - & . & \\
\end{bmatrix}
\\
A^{[1]} = 
\begin{bmatrix}
. & . & . & . & .  \\
| & | & . & . & | \\
a^{[1] (1)} & a^{[1] (2)} & . & . & a^{[1] (m)} \\
| & | & . & . & | \\
. & . & . & . & .  \\
\end{bmatrix}
\quad
b^{[1]} = 
\begin{bmatrix}
. & . & . & . & .  \\
| & | & . & . & | \\
b^{[1] (1)} & b^{[1] (2)} & . & . & b^{[1] (m)} \\
| & | & . & . & | \\
. & . & . & . & .  \\
\end{bmatrix}
\\
W^{[2]} = 
\begin{bmatrix}
w^{[2]}_1 & w^{[2]}_2  & w^{[2]}_3 & w^{[2]_4}\\
\end{bmatrix}
\\
b^{[2]} = 
\begin{bmatrix}
b^{[2] (1)} & b^{[2] (2)} & ... & b^{[2] (m)}  \\
\end{bmatrix}
\\
A^{[2]} = 
\begin{bmatrix}
a^{[2] (1)} & a^{[2] (2)} & ... & a^{[2] (m)}  \\
\end{bmatrix}
$$

则上述两层神经网络的向量化表示为

$$
\begin{align*}
& Z^{[1]} = W^{[1]}X + b^{[1]} \\
& A^{[1]} = \sigma(Z^{[1]}) \\ 
& Z^{[2]} = W^{[2]}A^{[1]} + b^{[2]} \\
& A^{[2]} = \sigma(Z^{[2]}) 
\end{align*}
$$

其中，$X$是`3xm`, $W^{[1]}$依旧是`4x3`, $A^{[i]}$是`4xm`，$b^{[1]}$也是`4xm`， $W^{[2]}$是`3x1`，$A^{[2]}$是`1xm`, $b^{[2]}$是`1xm`的。由此可以看出，训练样本增加并不影响$W^{[1]}$的维度

### Activation Functions

如果神经网路的某个Layer要求输出结果在`[0,1]`之间，那么选取$\sigma(x) = \frac{1}{1+e^{-x}}$作为Activation函数，此外，则可以使用**Rectified Linear Unit**函数：

$$
ReLU(z) = g(z) = max(0,z)
$$

<img src="{{site.baseurl}}/assets/images/2018/01/dp-w3-3.png" class="md-img-center" width="60%">

### Back Propagation

上述神经网络的Cost函数和前文一样

$$
J(W^{[1]}, b^{[1]}, W^{[2]}, b^{[2]}) = \frac {1}{m} \sum_{i=1}^mL(\hat{y}, y) = - \frac{1}{m} \sum\limits_{i = 1}^{m} \large{(} \small y^{(i)}\log\left(A^{[2] (i)}\right) + (1-y^{(i)})\log\left(1- A^{[2] (i)}\right) \large{)} \small\tag{13}
$$

其中$Y$为`1xm`的行向量 $Y = [y^{[1]},y^{[2]},...,y^{[m]}]$。对上述式子进行求导，可以得出下面结论(推导过程省略)

- $dZ^{[2]} = A^{[2]} - Y$ 
- $dW^{[2]} = \frac{1}{m}dZ^{[2]}A^{[1]^{T}}$ 
- $db^{[2]} = \frac{1}{m}np.sum(dz^{[2]}, axis=1, keepdims=True)$
- $dz^{[1]} = W^{[2]^{T}}dZ^{[2]} * g^{[1]'}(Z^{[1]}) \quad (element-wise \ product)$
- $dW^{[1]} = \frac{1}{m}dZ^{[1]}X^{T}$ 
- $db^{[1]} = \frac{1}{m}np.sum(dz^{[1]}, axis=1, keepdims=True)$

其中$g^{[1]^{'}}(Z^{[1]})$取决于Activation函数的选取，如果使用$tanh$，则$g^{[1]'}(Z^{[1]}) = 1-A^{[1]^2}$

### Gradient Descent

有了$dW^{[2]}$,$dW^{[1]}$,$db^{[2]}$,$db^{[2]}$，我们变可以使用梯度下降来update $W^{[1]}, b^{[1]}, W^{[2]}, b^{[2]}$了，公式和前面一样

$$
\theta = \theta - \alpha \frac{\partial J }{ \partial \theta }
$$

其中对$\alpha$的取值需要注意，不同learning rate的选取对梯度下降收敛的速度有着重要的影响，如下图

<img src="{{site.baseurl}}/assets/images/2018/01/dp-w3-4.gif" class="md-img-center">