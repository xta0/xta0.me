---
list_title: 深度学习 | Object Detection | 目标检测
title: Object Detection
layout: post
mathjax: true
categories: ["AI", "Machine Learning","Deep Learning"]
---

目标检测指的是通过训练神经网络可以识别出图片中的物体，并用矩形将其框出。它包含两部分，一部分是对图像内容的识别，也就是分类问题，另一个是标记目标在图像中的位置，也就是定位的问题。因此，神经网络的输出也包含了更多的信息，除了包含物体的类别外，还需要包含矩形框的位置信息。

### Sliding Windonw Detection

一种容易想到的目标检测方式是使用滑动窗口，我们用一个矩形窗口依次滑过图片中的每个区域，每个窗口通过一个已经训练好CNN网络（比如Resnet，GoogLnet等）进行图片识别，如下图所示

<img class="md-img-center" src="{{site.baseurl}}/assets/images/2018/04/dl-cnn-3-7.png">

这种方式的问题在于计算量太大，对于每个窗口都需要单独计算。举例来说，上图中的窗口大小为一个14\*14\*3，现在这个窗口向左，右和右下各滑动一次，假如用Resnet需要计算4次，得到4个结果

<img class="md-img-center" src="{{site.baseurl}}/assets/images/2018/04/dl-cnn-3-5.png">

在参考文献[1]中提到了减少计算量一个方案是将原来神经网路中的FC层变成卷积层，如下图所示

<img class="md-img-center" src="{{site.baseurl}}/assets/images/2018/04/dl-cnn-3-6.png">

还是上图中的滑动窗口，滑动步长为2，则4次滑动运算只需要一次即可完成

<img class="md-img-center" src="{{site.baseurl}}/assets/images/2018/04/dl-cnn-3-8.png">

此时得到结果是一个2\*2\*4的矩阵，包含4组计算结果






### YOLO


## Resources

- [OverFeat: Integrated Recognition, Localization and Detection using Convolutional Networks](https://arxiv.org/abs/1312.6229)
- [YOLO](https://pjreddie.com/darknet/yolo/)
- [You Only Look Once: Unified, Real-Time Object Detection](https://arxiv.org/abs/1506.02640)