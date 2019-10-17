---
list_title: 深度学习 | Convolutional Neural Networks
title: Convolutional Neural Networks
layout: post
mathjax: true
categories: ["AI", "Machine Learning","Deep Learning"]
---

> 文中部分图片截取自课程视频[Nerual Networks and Deep Learning](https://www.coursera.org/learn/neural-networks-deep-learning)

CNN是深度学习中一种处理图像的神经网络，可以用来做图像的分类，segmentation，目标检测等等。其工作原理可简单概括为通过神经网络的各层layer提取图片特征，然后针对这些特征做一系列的运算从而达到我们想要的结果。

### Edge Detection

图像的边缘检测实际上是对图像中的各个点做卷积运算，离散的卷积运算在时域（空域）上可以理解为是一种加权求和的过程，在频域上可以理解为一种滤波器。例如一副36个像素的灰度图片，我想想要检测它的竖直边缘，可以用一个3x3的kernel滑过图片的每个像素点，如下图所示

<div class="md-flex-h md-flex-no-wrap md-margin-bottom-12">
<div><img src="{{site.baseurl}}/assets/images/2018/01/dl-cnn-1-1.png" width="80%"></div>
<div class="md-margin-left-12"><img src="{{site.baseurl}}/assets/images/2018/01/dl-cnn-1-2.png" width="80%"></div>
</div>

图中蓝色区域的脚标值构成一个kernel，如上图中的kernel为

$$
\begin{bmatrix}
1 & 0 & -1  \\
1 & 0 & -1  \\
1 & 0 & -1  \\
\end{bmatrix}
$$

对于滑过的蓝色区域，对图中的像素和kernel矩阵进行element-wise的乘积并求和，以作图为例，滤波后的像素点为的值为

```shell
3x1 + 1x1 + 2x1 + 0x0 + 5x0 + 7x0 + 1x(-1) + 8x(-1) + 2x(-1) = -5
```

这样当kernel滑过整张图片后，会得到一个4x4的矩阵，包含滤波后的像素值。

<img src="{{site.baseurl}}/assets/images/2018/01/dl-cnn-1-3.png" width="40%">

图中可见滤波后图像的大小为kernel在水平和竖直方向上所滑过的次数。我们假设图片的大小是`nxn`的，kernel的大小是`fxf`的，那么输出图片的大小为

$$
n-f+1 \times n-f+1
$$

对于kernel的选取并不固定，如果熟悉图像处理算法，可知边缘检测的算子有很多种，比如Sobel，拉普拉斯等。但是这些算子的值都是根据经验来确定的，比如Sobel的算子的值为

$$
\begin{bmatrix}
1 & 0 & 1  \\
2 & 0 & -3  \\
1 & 0 & -1  \\
\end{bmatrix}
$$

CNN要做的事情就是将这些经验值，通过神经网络的训练得到一个更加精确的值。

### Padding

从上面边缘检测的过程中可以看出，每当图片完成一次卷积运算后，它的大小会变小；另外，对于图片中边缘的像素点，只会经过一次的卷积运算，而对于位于图中中心位置的点，则可能经过多次的卷积运算，因此卷积运算对于图片边缘的点并不能有效的运用起来。

为了解决这两个问题，我们可以给输入的图片增加一圈padding，以上面6x6的图片为例，如果我们在图片周围各加一个像素的padding，那么6x6的图片，将会变成8x8，经过卷积运算后的图片尺寸依旧是 8-3+1 = 6x6。

我们另$p$为padding的像素数，则卷积后的图片尺寸为

$$
n+2p-f+1 \times n+2p-f+1 
$$

在CNN中，我们称Valid为没有padding的卷积运算，称Same为有padding的卷积运算，卷积后的尺寸和原图片相同，此时要求 

$$
p = \frac{f-1}{2}
$$

在计算机视觉中，`f`通常为奇数

### Strided Convolutions

在前面例子中，kernel滑过图片时的步长为1，我们也可以改变kernel滑动的步长，如下图中，kernel滑动的步长为2

<div class="md-flex-h md-flex-no-wrap md-margin-bottom-12">
<div><img src="{{site.baseurl}}/assets/images/2018/01/dl-cnn-1-4.png"></div>
<div class="md-margin-left-12"><img src="{{site.baseurl}}/assets/images/2018/01/dl-cnn-1-5.png" ></div>
</div>

上图中，一个7x7的图片和一个3x3的kernel，按照步长为2进行卷积运算，得到的图片大小为3x3。计算方法为

$$
\lfloor{\frac{n+2p-f}{stride} + 1}\rfloor \times \lfloor{\frac{n+2p-f}{stride} + 1}\rfloor
$$

### Convolutions Over Volume 

在前面的例子中，我们介绍了二维灰度图片的卷积运算，我们可以将原理推广到三维的RGB图片上，对于RGB图片的卷积运算，我们可以用下图表示

<img class="md-img-center" src="{{site.baseurl}}/assets/images/2018/01/dl-cnn-1-6.png" width="80%">

此时我们的输入图片变成了6x6x3的矩阵，表示有三张RGB的二维图片，内存中的排列方式为

```shell
r r r ... r
r r r ... r
... ... ...
r r r ... r
g g g ... g
g g g ... g
... ... ...
g g g ... g
b b b ... b
b b b ... b
... ... ...
b b b ... b
```

值得注意的是，对于图片矩阵，6x6x3的含义和3x6x6的含义并不相同，后者表示的矩阵为

```shell
rgb rgb rgb ... rgb
rgb rgb rgb ... rgb
... ... ... ... rgb
rgb rgb rgb ... rgb
```

同样对于卷积核，也是一个3x3x3的矩阵，对应RGB三个通道。整个卷积运算的过程为RGB三通道图片分别和对应的卷积核进行卷积操作，将结果填充到一个4x4的矩阵中。

注意到，上图中我们只考虑了一种情况，即一张RGB图片和一个kernel进行卷积得到一个4x4的矩阵，这个kernel可以是竖直边缘检测的kernel，那么得到的4x4矩阵则是图片的竖直边缘特征。如果我们要同时提取图片的竖直和水平边缘则需要让图片和另一个kernel进行卷积，得到另一个4x4的矩阵，那么最终的结果将是一个4x4x2的矩阵，如下图所示

<img class="md-img-center" src="{{site.baseurl}}/assets/images/2018/01/dl-cnn-1-8.png">

小结一下，对于图片的卷积操作，一张$n \times n \times n_c$的图片和一个$f \times f \times n_c$的kernel做卷积得到的输出为

$$
n-f+1 \times n-f+1 \times n_c^{'}
$$

其中$n_c^{'}$为kernel的个数

<img src="{{site.baseurl}}/assets/images/2018/01/dl-cnn-1-9.png">

如上图中是一个一层的CNN，其中hidden layer包含两个4x4的节点$a_i$。回忆前面神经网络的概念，对于单层的神经网络，输出可用下面式子表示

$$
z^{[1]} = W^{[1]}a^{[0]} + b^{[1]} \\
a^{[0]} = g(z^{[1]}) \\
$$


对应到CNN中，$a^{[0]}$是我们输入的图片，大小为6x6x3，两个kernel类比于$W^[1]$矩阵，则$W^{[1]}a^{[0]}$类比于$a^{[0]} * W^{[1]}$得到的输出为一个4x4的矩阵，接下来让该矩阵中的各项加上bias，即$b1$，$b2$，再非线性函数$Relu$对其求值，则可得到$a^{[1]}$

如果layer $l$是一个convolution layer，另

- $f^{[l]}$ = filter size 
- $p^{[l]}$ = padding
- $s^{[l]}$ = stride
- $n_C^{[l]}$ = number of filters

则layer ${l}$的input的size为

$$
n_H^{[l-1]} \times  n_W^{[l-1]} \times n_C^{[l-1]}
$$

output的size为

$$
n_H^{[l]} \times n_W^{[l]} \times n_C^{[l]}
$$

其中，$n_H^{[l]}$ 和 $n_W^{[l]}$的size计算公式前面曾提到过

$$
n_H^{[l]} = \lfloor{\frac{n^{[l-1]}+2p^{[l]}-f^{[l]}}{s^{[l]}} + 1}\rfloor
$$

每个Fileter的size为 

$$
f^{[l]} \times f^{[l]} \times n_C^{[l-1]}
$$

Hidden Layer中每个Activation unit $A^{[l]}$的size为，其中m为batch数量

$$
 m \times n_H^{[l]} \times n_W^{[l]} \times n_C^{[l]}
$$

Weights $W^{[l]}$的size为

$$
f^{[l]} \times f^{[l]} \times n_C^{[l-1]} \times n_C^{[l]}
$$

Bias $b^{[l]}$的size为 $n_C^{[l]}$

### Pooling layer

Pooling是用来对输入矩阵进行优化的一种方法。举例来说，下图是对一个4x4的矩阵进行max pooling，得到一个2x2的矩阵

<img src="{{site.baseurl}}/assets/images/2018/01/dl-cnn-1-10.png">

上图中，max-pooling的两个参数 - $stride$为2，$f$为2，对于多维矩阵同理

<img src="{{site.baseurl}}/assets/images/2018/01/dl-cnn-1-11.png">

注意，Pooling用到的这两个参数是经验值，并非通过backprop求得。

总结一下，如果Pooling layer的输入为 $n_H \times n_W \times n_C$，则输出的size为

$$
\lfloor{\frac{n_H-f}{stride} + 1}\rfloor \times \lfloor{\frac{n_W-f}{stride} + 1}\rfloor \times n_C
$$

### A Convolutional Network Example

一般来说，一个卷积神经网络有下面几种layer

- Convolution
- Pooling
- Fully connected

如下图是一个LeNet-5的卷积神经网路

<img src="{{site.baseurl}}/assets/images/2018/01/dl-cnn-1-12.png">

该神经网络后面几层为Fully connected layer。对于卷积神经网络一个比较常见的pattern是conv layer后面追加pooling layer，并且最后几层为FC，然后是一个softmax做分类。