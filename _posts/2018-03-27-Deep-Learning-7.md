---
list_title: 深度学习 | Classic CNN Models | 几种经典的CNN Model
title: Convolutional Neural Networks
layout: post
mathjax: true
categories: ["AI", "Machine Learning","Deep Learning"]
---

### LeNet-5

<img src="{{site.baseurl}}/assets/images/2018/03/dl-cnn-2-lenet5.png">

LetNet-5是Y.LeCun在1998年在的paper中提出的，被用来识别手写数字。这篇paper堪称经典，其结构前面已经文章中已经介绍过了，这里就不再重复。LeNet-5并不算复杂，只有60k个参数，相对今天的神经网络动辄几千万，甚至上亿个参数来说，还是比较初级的。但这个神经网络为后来的CNN奠定了雏形，并提供了不少最佳实践，比如随着网络的深入，输出图片的size越来越小，channel到是越来越多；conv layer后面经常跟随pool layer，然后跟随FC等。其基本的参数如下

layer  | type | filter shape
-------| ---- | ---- 
1  | input | 32x32x1
2  | conv2d | (5x5x1)x6
3  | max pool |  2*2
4  | conv2d | (5x5x6)x16
5  | max pool |  2*2
6  | conv2d | 400x120
7  | FC | 120x84
8  | FC | 84x10

### AlexNet

<img src="{{site.baseurl}}/assets/images/2018/03/dl-cnn-2-alexnet.png">

AlexNet是Alex Krizhevsky和他的导师Geoffrey Hinton在2012年的paper中提出的，并获得了当年的ImageNet冠军，识别成功率比第二名高出近10个百分点。相比于LeNet，AlexNet层次更深，有8层卷积层，参数更是达到了6千多万个。但它的独到之处还在于下面几点

1. 使用ReLu作为激活函数。虽然这个选择在现在看来很平常，但当时是一个很大胆的创新，它解决了梯度消失的问题，使收敛加快，从而提高了训练速度
2. 使用多GPU进行训练，解决了GPU间通信的问题
3. 介绍一种归一化方式 - Local Response Normalization，后面被很多学者证明并不是很有效

AlexNet的参数如下

layer  | type | filter shape
-------| ---- | ---- 
1  | input | 224x224x3
2  | conv1 / Relu | (11x11x3)x96
3  | LRN | 5
4  | max pool |  3x3
5  | conv2 / Relu | (5x5x96)x256
6  | LRN | 5
7  | max pool |  3x3
8  | conv3 / Relu | (3x3x256)x384
9  | conv4 / Relu | (3x3x384)x384
10  | conv5 / Relu | (3x3x384)x256
11  | max pool |  3x3
12  | FC | (6x6x256)x4096
13  | Dropout | 
14  | FC | 4096 x 4096
15  | Dropout | 
16  | FC | 4096 x 1000

在AlexNet提出后，人们逐渐意识到使用CNN进行图像识别确实是一条可行的方法

### VGG / VGG 16

AlexNet发表后，业界对这个模型做了很多改进的工作，使得其在ImageNet上的表现不断提升。VGG是其中一个比较重要的改进，并在2014年的比赛中获得了第一名。



### Resources

- [LetNet5 - Gradient-based learning applied to document recognition](https://ieeexplore.ieee.org/document/726791)
- [AlexNet – ImageNet Classification with Deep Convolutional Neural Networks](https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks)