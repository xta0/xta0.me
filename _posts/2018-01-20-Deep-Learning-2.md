---
list_title: 深度学习 | Neural Networks Overview
title: Neural Networks Overview
layout: post
mathjax: true
categories: ["AI", "Machine Learning","Deep Learning"]
---

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

注意，这里面$W^{[1] (i)}$是`4x3`二维矩阵，结合前面文章可知，我们可以用向量化计算来取代`for`循环，另

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
. & - & w^{[1] (1)} & - & . & \\
. & - & w^{[1] (2)} & - & . & \\
. & - & w^{[1] (3)} & - & . & \\
. & - & w^{[1] (4)} & - & . & \\
\end{bmatrix}
\\
A^{[1]} = 
\begin{bmatrix}
. & . & . & . & .  \\
| & | & | & . & | \\
a^{[1] (1)} & a^{[1] (2)} & . & . & a^{[1] (m)} \\
| & | & | & . & | \\
. & . & . & . & .  \\
\end{bmatrix}
\quad
b^{[1]} = 
\begin{bmatrix}
. & . & . & . & .  \\
| & | & | & . & | \\
b^{[1] (1)} & b^{[1] (2)} & . & . & b^{[1] (m)} \\
| & | & | & . & | \\
. & . & . & . & .  \\
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

其中，$X$是`3xm`, $W^{[1]}$是`4x3`的, 则$A^{[i]}$是`4xm`的，因此$b^{[1]}$也是`4xm`的

### Activation Functions

如果神经网路的某个Layer要求输出结果在`[0,1]`之间，那么选取$\sigma(x) = \frac{1}{1+e^{-x}}$作为Activation函数，此外，则可以使用**Rectified Linear Unit**函数：

$$
ReLU(z) = max(0,z)
$$

<img src="{{site.baseurl}}/assets/images/2018/01/dp-w3-3.png" class="md-img-center" width="60%">

