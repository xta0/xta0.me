---
list_title: 深度学习 | Neural Style Transfer
title: Neural Style Transfer
layout: post
mathjax: true
categories: ["AI", "Machine Learning","Deep Learning"]
---

## Neural Style Transfer

所谓Netural Style Transfer是指将一幅图片的style信息提取出来并应用到另一幅图片的content上，从而合成一副同时具备两幅图片特征的新图片。如下图所示

<img class="md-img-center" src="{{site.baseurl}}/assets/images/2018/04/dl-cnn-4-cat-st.png">

接下来的问题便是

1. 如何提取content image中的content
2. 如何提取style image中的style
3. 如何提取出的content和style重新组合起来

### 提取content

参考文献[1]的论文中提到，使用卷积神经网络来帮我们提取content。这里我们要先重新回顾一下前面的内容，以图像分类为例，一个卷积神经网络可以分为两部分，前一部分是由conv2d和pooling组成的卷积层，其作用是提取图片中的特征。后一部分是由FC层和Softmax组成的分类层，其作用是将图片的特征flatten成一维向量并映射到某个具体的类别上。

而对于提取特征来说，我们并不需要后面的FC层，只需要保留前面的卷积层即可。论文中使用VGG19，如下图所示

<img class="md-img-center" src="{{site.baseurl}}/assets/images/2018/04/dl-cnn-4-vgg19.png" width="80%">

虽然我们保留了卷积层，但我们还要知道图片通过每个卷积层之后的输出，也就是说各卷积核到底在提取图片的哪些特征，阅读参考文献[2,3]可知，随着卷积网络的加深，卷积层提取的特征粒度将越来越大，比如前几层的卷积层可识别图片的边缘，颜色等，随着网络的加深，后面几层则可以识别人脸，身体等大型特征。

<img class="md-img-center" src="{{site.baseurl}}/assets/images/2018/04/dl-cnn-4-features.png" width="90%">

接下来我们要做的便是根据这几层的输出来重建一张新的图片，新的图片需要具备原图的重要特征。我们首先创建一张空白图片（或者全是噪声的图片）用$T_c$表示，我们将原图通过某个卷积层后的输出结果用$C_c$表示，接下来我们计算两者的差值，使用下面的式子

$$
L_{content} = \frac{1}{2}\sum(T_c-C_c)^2
$$

接下来我们便可以用梯度下降法求导，并最终确定$T_c$的值。在论文中，作者重点观察了`conv1_2`, `conv2_2`, `conv3_2`, `conv4_2`和`conv5_2`这几层的输出，发现这几层可以很好的重建原图的特征。

## Resources

1. [Image Style Transfer Using Convolutional Neural Networks](https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Gatys_Image_Style_Transfer_CVPR_2016_paper.pdf)
2. [Visualizing and Understanding Convolutional Networks](https://arxiv.org/abs/1311.2901)
3. [Visualizing and Understanding Deep Neural Networks by Matt Zeiler](https://www.youtube.com/watch?v=ghEmQSxT6tw)
4. Florian Schroff, Dmitry Kalenichenko, James Philbin (2015). [FaceNet: A Unified Embedding for Face Recognition and Clustering](https://arxiv.org/pdf/1503.03832.pdf)
5. Yaniv Taigman, Ming Yang, Marc'Aurelio Ranzato, Lior Wolf (2014). [DeepFace: Closing the gap to human-level performance in face verification](https://research.fb.com/wp-content/uploads/2016/11/deepface-closing-the-gap-to-human-level-performance-in-face-verification.pdf) 
6. The pretrained model we use is inspired by Victor Sy Wang's implementation and was loaded using his code: https://github.com/iwantooxxoox/Keras-OpenFace.
7. Our implementation also took a lot of inspiration from the official FaceNet github repository: https://github.com/davidsandberg/facenet 