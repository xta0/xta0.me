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

这种方式的问题在于计算量太大，对于每个窗口都需要单独计算。举例来说，上图中的窗口大小为一个14\*14\*3，现在这个窗口向左，右和右下各滑动一次，步长为2，选取Resnet作为图像识别网路，则需要按照上图的方式计算4次，得到4个结果

<img class="md-img-center" src="{{site.baseurl}}/assets/images/2018/04/dl-cnn-3-5.png">

在参考文献[1]中提到了减少计算量一个方案是将原来神经网路中的FC层变成卷积层，如下图所示

<img class="md-img-center" src="{{site.baseurl}}/assets/images/2018/04/dl-cnn-3-6.png">

还是上图中的滑动窗口，滑动步长为2，则4次滑动运算只需要一次即可完成

<img class="md-img-center" src="{{site.baseurl}}/assets/images/2018/04/dl-cnn-3-8.png">

此时得到结果是一个2\*2\*4的矩阵，包含了当前位置，右边，下边和右下共4组计算结果。实际上这种方式是将上面4次单独计算合并成了一次。推而广之，假如我们有一个更大的的滑动窗口(28\*28\*3)，一次运算，我们可以得到64组运算结果

<img class="md-img-center" src="{{site.baseurl}}/assets/images/2018/04/dl-cnn-3-4.png">

再进一步扩大，我们我们可以让一个图片等同于一个滑动窗口，只进行一次运算即可

<img class="md-img-center" src="{{site.baseurl}}/assets/images/2018/04/dl-cnn-3-9.png">

### YOLO

上述算法虽然解决了计算效率问题，但是没有解决对目标矩形的定位问题，例如，上面算法中，我们完全有可能碰到这种情况，即没有任何一个窗口能完全覆盖检测目标，如下图所示

<img class="md-img-center" src="{{site.baseurl}}/assets/images/2018/04/dl-cnn-3-3.png">

解决这个问题，参考文献[2]，即YOLO算法提供一个不错的思路。YOLO将一图图片分割成$n$*$n$的几个小区域，如下图中$n=3$，即9个格子

<img class="md-img-center" src="{{site.baseurl}}/assets/images/2018/04/dl-cnn-3-10.png">

在数据标注的时候，我们待检测目标的中心赋予某一个box，比如上图中的黄点和绿点。然后对该box用下面的一个向量表示

$$
y = [ p_c, b_x, b_y, b_h, b_w, c_1, c_2,c_3 ]
$$

其中，$p_c$表示该box中是否有待检测的目标，如果有$p_c$为1，否则为0，$(b_x, b_y, b_h, b_w)$表示目标矩形，$(b_x,b_y)$表示目标的中心点，$(b_h, b_w)$表示矩形框的高和宽，最后$c_1, c_2,c_3$表示目标类别。例如上图中黄色矩形为

$$
y = [1.0, 0.3, 0.4, 0.9, 0.5, 0, 1, 0]
$$

其中每个box的左上角为(0,0)，右下角为(1,1)，$b_h$和$b_w$为相对于该box的百分比


## Resources

- [OverFeat: Integrated Recognition, Localization and Detection using Convolutional Networks](https://arxiv.org/abs/1312.6229)
- [YOLO](https://pjreddie.com/darknet/yolo/)
- [You Only Look Once: Unified, Real-Time Object Detection](https://arxiv.org/abs/1506.02640)