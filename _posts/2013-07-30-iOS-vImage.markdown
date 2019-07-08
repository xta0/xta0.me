---
list_title: iOS中的vImage | vImage in iOS
title: iOS中的vImage
layout: post
categories: [iOS]
mathjax: true
---

最近大家都在探讨iOS7的blur effect，用的最多的基本上是apple在<a href="https://developer.apple.com/downloads/index.action?name=WWDC%202013">WWDC2013中的demo</a>。其实现思路是：高斯模糊+一层白色的mask，再调整一下饱和度。出于性能考虑，demo采用了Accelerate framework中的<a href="https://developer.apple.com/library/mac/documentation/performance/Conceptual/vImage/Introduction/Introduction.html">vImage</a>，vImage提供了模版运算的api，可以直接通过openGL做向量运算，因此demo采用了<a href="http://www.w3.org/TR/SVG/filters.html#feGaussianBlurElement">三次均值滤波去逼近高斯滤波的效果</a>，误差有3%。但却能大大提高计算效率。

那为什么处理图片要用要用<a href="http://zh.wikipedia.org/wiki/%E5%8D%B7%E7%A7%AF">卷积运算</a>，其实也不是所有的图像处理都需要卷积运算，只有需要做邻域运算才会用到模版，比如图像平滑，图像锐化等。那卷积是什么概念呢？

从时域的角度讲（wiki百科上有很详细的解释），卷积运算是一种“加权平均”的方法。从频域的角度讲，卷积可以说是一种滤波。假设我们有一幅图像，高频信息很多（噪声，突变，边缘等），我们想对它进行去噪，就可以对这幅图片进行低通滤波，让只有低频信息的像素得以保留。进行低通滤波，就是在空域上将图像和某个系统做卷积运算，所谓的某个系统也叫做卷积核（kernel）。由于平面图像是二维的，像素点是离散的，因此，卷积运算是一种离散二维卷积运算:

$$
g(s,t) = \sum_{m1}^{m1+m2-1}\sum_{n1}^{n1+n2-1}f(m,n)h(s-m+1,t-n+1)
$$

将这个式子展开就是所谓的模版运算。

Apple的vImage，最大的优势就是提供卷积运算和向量运算的api，对于滤波这种图像处理算法在计算性能上能得到大大的提升，但是它也有局限性，只能做一些简单的线性滤波，复杂一点的的非线性滤波（如中值滤波），vImage就不支持了（或者是我没找对用法），因此，效果好一些的滤镜vImage也做不了，下面我们用vImage的api做两个简单的模版运算：

（1）均值滤波

（2）拉普拉斯锐化


### 均值滤波

均值滤波，也叫邻域平均是图像平滑去噪算法中最简单的一种，它将原图中的一个像素值和它周围临近的N个像素值相加，然后求得平均值，最为新图中该点的像素值，其数学公式为：

$$ g(i,j) = \sum f(i,j)/N $$

N是kernel中包含的像素个数。

例如3x3的kernel为：

$$ 
\frac{1}{9}\begin{bmatrix}
1 &1 &1 \\ 
1 &1 &1 \\ 
1 &1 &1 
\end{bmatrix}
$$

vImage对于这种权重相同的kernel提供了一个专门的api叫做vImageBoxConvolve_ARGB8888：

```c
err = vImageBoxConvolve_ARGB8888(   effectInBuffer, 
                                    effectOutBuffer, 
                                    NULL, 
                                    0, 0, 9, 9, 
                                    bgColor, 
                                    kvImageEdgeExtend);
```

原图为256x256的ARGB，结果为：

<div class="md-flex-h">
<div><img src="{{site.baseurl}}/assets/images/2013/11/lena_ave_o.png"></div>
<div class="md-margin-left-12"><img src="{{site.baseurl}}/assets/images/2013/11/lena-5x5.png" ></div>
<div class="md-margin-left-12"><img src="{{site.baseurl}}/assets/images/2013/11/lena-9x9.png" ></div>
</div>


第一幅图为原图，第二幅为使用5x5模版的平滑结果，第三幅为使用9x9模版的平滑结果。显然这种方式在去掉了噪声的同时，也模糊了边缘。

### 拉普拉斯锐化

和上面的平滑不同，锐化是消除图片中的低频分量，保留高频分量，是一种高通滤波器。从数学上讲，平滑是一种平均运算或积分运算，那么和它相反，锐化就是一种微分运算，而微分运算又可以说是梯度运算，表征了新号的变化快慢，因此可以用来检测图像的边缘信息。一般简单的锐化用梯度锐化就可以了，拉普拉斯锐化用的是拉普拉斯算子：

$$
\Delta f = \nabla^2 f = \frac{\partial^2f}{\partial x^2} + \frac{\partial^2 f}{\partial y^2}
$$

拉普拉斯算子是一种二阶微分算子，我们通常用它过零点来确定像素的图片点,其离散形式为：

$$
\nabla^2 f = [f(x+1,y) + f(x-1,y)+f(x,y+1)+f(x,y-1)]-4f(x,y)
$$

由于拉普拉斯算子的模版权重不同，因此需要使用vImage的api：vImageConvolve_ARGB8888，需要显示提供模版：

```c
int16_t kernel[9] = {1,1,1,1,-8,1,1,1,1};
err = vImageConvolveWithBias_ARGB8888(  effectInBuffer,
                                        effectOutBuffer,
                                        NULL, 
                                        0, 0, kernel, 
                                        3, 3, 1, 128, 
                                        bgColor, 
                                        kvImageEdgeExtend);
``` 

Bias的意思是如果计算过程中如果有溢出（比如出现负数或者大于255），那么会自动纠正。结果如下：


<div class="md-flex-h">
<div><img src="{{site.baseurl}}/assets/images/2013/11/lena_ave_o.png"></div>
<div class="md-margin-left-12"><img src="{{site.baseurl}}/assets/images/2013/11/lena-laplas-3x3.png"></div>
</div>

可以看到通过使用高通滤波器，我们成功的检测除了图像的边缘，但是大部分低频信号也被滤除了，因此图像只保留了边缘信息

