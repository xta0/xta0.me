---
list_title: 深度学习 | Face Verification and Neural Style Transfer
title: Face Verification | Neural Style Transfer
layout: post
mathjax: true
categories: ["AI", "Machine Learning","Deep Learning"]
---

## Neural Style Transfer

所谓Netural Style Transfer是指将一幅图片的style信息提取出来并应用到另一幅图片的content上，从而合成一副同时具备两幅图片特征的新图片。如下图所示

<img src="{{site.baseurl}}/assets/images/2018/01/dl-cnn4-cat-st.png">

这里我们要重新回顾一下卷积神经网络的作用，以图像分类为例，一个卷积神经网络可以分为


人脸识别的一个挑战是如何在只有一张人脸图片的情况下，识别出当前的人是否是图片中的人。也就是所谓的one Shot Learning，其本质找到两张图片的差异，使这个差值越小越好

$$
d(img1, img2) = degree of difference between images
$$

有了上面的式子，我们的目的便是构建一个CNN网络来生成一个上述模型

## Resources

1. [Image Style Transfer Using Convolutional Neural Networks](https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Gatys_Image_Style_Transfer_CVPR_2016_paper.pdf)

2. Florian Schroff, Dmitry Kalenichenko, James Philbin (2015). [FaceNet: A Unified Embedding for Face Recognition and Clustering](https://arxiv.org/pdf/1503.03832.pdf)
3. Yaniv Taigman, Ming Yang, Marc'Aurelio Ranzato, Lior Wolf (2014). [DeepFace: Closing the gap to human-level performance in face verification](https://research.fb.com/wp-content/uploads/2016/11/deepface-closing-the-gap-to-human-level-performance-in-face-verification.pdf) 
4. The pretrained model we use is inspired by Victor Sy Wang's implementation and was loaded using his code: https://github.com/iwantooxxoox/Keras-OpenFace.
5. Our implementation also took a lot of inspiration from the official FaceNet github repository: https://github.com/davidsandberg/facenet 