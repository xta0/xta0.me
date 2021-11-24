---
list_title: 笔记 | 深度学习 | Semantic Segmentation
title: Semantic Segmentation
layout: post
mathjax: true
categories: ["AI", "Machine Learning","Deep Learning"]
---

Sementic Segmentation 可以对图片中像素点进行分类，比如下面例子中，我们将car标记为1，building标记为2，road标记为3。model的输出为一个segmentation map。

<img class="md-img-center" src="{{site.baseurl}}/assets/images/2018/05/dl-cnn-unet-0.png">

Unet是目前一个比较流行的semantic segmentation model，它结构是一个U shape，从左到右，feature map的depth逐渐增加，spatial size逐渐减小；从右到左，feature map的depth逐渐减小，spatial size逐渐增加。

<img class="md-img-center" src="{{site.baseurl}}/assets/images/2018/05/dl-cnn-unet-1.png">

### Transpose Convolution

其中Unet的右半边需要用到Transpose Conv，其运算过程如下

<img class="md-img-center" src="{{site.baseurl}}/assets/images/2018/05/dl-cnn-unet-2.png">

上面例子中的输入是一个`(2x2)`的tensor，conv kernel是`(3x3)`，输出是一个`(4x4)`的tensor。其中，padding = 1, stride = 2。

使用Transpose conv的一个好处是kernel是train出来的，因此比直接用upsample得到的segmentation map更准确。缺点是会增加model的size。

### Unet Architecture

Unet结构中一个特别的地方在于skip connection。这个connection实际上是一个element-wise的add操作，它将左边的feature map直接加到右边对应的feature map上

<img class="md-img-center" src="{{site.baseurl}}/assets/images/2018/05/dl-cnn-unet-3.png">

对于右边的feature map，经过transpose conv之后，它里面包含high-level，spatial，contextual inforamtion，但是缺少low-level，detail的information。而左边与之对应的feature map则恰好包含这些信息。