---
list_title: 深度学习 | Classic CNN Models | 几种经典的CNN Model
title: Convolutional Neural Networks
layout: post
mathjax: true
categories: ["AI", "Machine Learning","Deep Learning"]
---

### LeNet-5

<img src="{{site.baseurl}}/assets/images/2018/03/dl-cnn-2-lenet5.png">

LetNet-5是Y.LeCun在1998年在的paper中提出的，被用来识别手写数字。这篇paper堪称经典，其结构前面已经文章中已经介绍过了，这里就不再重复。LeNet-5并不算复杂，只有60k个参数，相对今天的神经网络动辄几千万，甚至上亿个参数来说，还是比较初级的。但这个神经网络为后来的CNN奠定了雏形，并提供了不少最佳实践，比如随着网络的深入，输出图片的size越来越小，channel越来越深；conv layer后面跟随pooling layer，然后跟随FC等。其基本的参数如下

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

AlexNet是Alex Krizhevsky和他的导师Geoffrey Hinton在2012年的paper中提出的，并获得了当年的ImageNet冠军，识别成功率比第二名高出近10个百分点。相比于LeNet，AlexNet层次更深，有8层卷积层 + 3层FC层，参数更是达到了6千多万个。但它的独到之处还在于下面几点

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

<img src="{{site.baseurl}}/assets/images/2018/03/dl-cnn-2-vgg16.png">

AlexNet发表后，业界对这个模型做了很多改进的工作，使得其在ImageNet上的表现不断提升。VGG是其中一个比较重要的改进，它在2014年的比赛中获得了第一名。VGG的最主要贡献在于通过加深网络层次，来得到更好的模型泛化能力,是一个真正意义上Deep Network。

在AlexNet之后，学术界一致认为，从理论上看，神经网络应该是层数越多，泛化能力越强。但实际中，随着网络的不断加深，往往会出现梯度消失和过拟合的情况出现，这使得加深网络十分困难。VGG通过一系列手段解决了这个问题，具体来说

1. 对所有的卷积层统一使用3x3，步长为1，相同padding的kernel
2. 对所有的pooling层，统一使用使用2x2，步长为2的Max Pooling
3. 放弃了AlexNet中引入的LRN技术，原因是这个技巧并没有带来效果的提升

上图中是一个16层的VGG网络，看起来非常规则，因此也得到了一批researcher的热爱，但是该结构一共有大约138million个参数，训练起来很有挑战

### ResNet

前面曾提到过，随着网络的不断加深，会出现梯度“消失”或者梯度“爆炸”的情况。VGG虽然解决了一些问题，但研究人员也发现了一个现象，当将VGG不断加深，直到50多层后，模型的性能不但没有提升，反而开始下降，也就是准确度变差了，网络好像遇到了瓶颈。为了解决这个瓶颈，ResNet提出了残差网络（Residual Network）的概念，从而可以将模型规模从几层，十几层或几十层一直推到上百层的结构，且错误率只有VGG或GoogleNet的一半。这篇论文也获得了2016年CVPR的最佳论文，在发表后获得了超过1万2千次的引用。

前面提到RetNet建立在Residual Block的概念之上，接下来我们就来看看它是怎么解决问题的。

<img src="{{site.baseurl}}/assets/images/2018/03/dl-cnn-2-resnet-1.png" width="80%">

假设我们有一个两层的FC网路如上图所示，按照之前介绍的求法，则有下面一些式子

$$
z^{[l+1]} = W^{[l+1]}a^{[l]} + b^{[l+1]} \\
a^{[l+1]} = g(z^{[l+1]}) \\
z^{[l+2]} = W^{[l+2]}a^{[l+1]} + b^{[l+2]} \\
a^{[l+2]} = g(z^{[l+2]}) \\
$$

也就是说，如果想要得到$a^[l+2]$，必须要经历上面4部求解过程。而Residual Network则直接将$a[l]$作为Residual Block加入到了网络的末尾，如下图所示

<img src="{{site.baseurl}}/assets/images/2018/03/dl-cnn-2-resnet-2.png">

则$a^{[l+2]}$变成了

$$
a^{[l+1]} = g(z^{[l+2]}+a^{[l]})
$$

推而广之，如果我们有一个下图中的"Plain Network"，我们可以将下面的layer两两形成一个Residual Block，进而组成了一个Residual Network

<img src="{{site.baseurl}}/assets/images/2018/03/dl-cnn-2-resnet-3.png">

那为什么ResNet






### Resources

- [LetNet5 - Gradient-based learning applied to document recognition](https://ieeexplore.ieee.org/document/726791)
- [AlexNet – ImageNet Classification with Deep Convolutional Neural Networks](https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks)
- [VGG - Very Deep Convolutional Networks for Large-Scale Image Recognition](https://arxiv.org/pdf/1409.1556.pdf)
- [ResNet - Deep Residual Learning for Image Recognition]()