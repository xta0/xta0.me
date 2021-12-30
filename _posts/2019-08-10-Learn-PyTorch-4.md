---
list_title: PyTorch | Cycle GAN
title: Cycle GAN
layout: post
mathjax: true
categories: ["PyTorch", "Machine Learning","Deep Learning"]
---

## Pix2Pix

在了解Cycle GAN之前，我们先要了解下Pix2Pix GAN。Pix2Pix GAN解决的问题是image mapping，即我们需要让Generator将一张图片`x`映射成另一张图片`y`。这就需要我们的training data是一个组pair images。其中，$x_i$是Generator的输入，$y_i$是Discriminator的输入

<img class="md-img-center" src="{{site.baseurl}}/assets/images/2019/08/gan_08.png">

Paper使用Unet作为Generator的architecture。Input先经过一个encoder变成小的feature maps，再经过decoder将尺寸复原，最后通过Discriminator进行classification。

<img class="md-img-center" src="{{site.baseurl}}/assets/images/2019/08/gan_09.png">

对于Discriminator，


和前面介绍的GAN不同的是，Generator的输入不是一个small random的vector，而是一张图片。Discriminator的输入是一组pair

## Resources

- [Pix2Pix paper](https://arxiv.org/pdf/1611.07004.pdf)
- [DC GAN paper](https://arxiv.org/pdf/1511.06434.pdf)
- [BatchNorm Paper](https://arxiv.org/pdf/1502.03167.pdf)
- [Udacity Deep Learning](https://classroom.udacity.com/nanodegrees/nd101)


