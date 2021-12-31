---
list_title: PyTorch | Cycle GAN
title: Cycle GAN
layout: post
mathjax: true
categories: ["PyTorch", "Machine Learning","Deep Learning"]
---

## Pix2Pix

在了解Cycle GAN之前，我们先要了解下Pix2Pix GAN。Pix2Pix GAN解决的问题是image mapping，即Generator将一张图片`x`映射成另一张图片`y`。这就需要我们的training data是一个组pair images。其中，$x_i$是Generator的输入，$y_i$是Discriminator的输入

<img class="md-img-center" src="{{site.baseurl}}/assets/images/2019/08/gan_08.png">

Paper使用Unet作为Generator的architecture。Input先经过一个encoder变成小的feature maps，再经过decoder将尺寸复原，最后通过Discriminator进行classification。

<img class="md-img-center" src="{{site.baseurl}}/assets/images/2019/08/gan_09.png">

上面的结构有一个问题是，Discriminator如何判断Generator的输出是否是fake。举例来说，理想情况下，$G(x_1)=y_1$，但是如果出现$G(x_2)=y_1$的情况，Discriminator也会将其判定为true。因此，Discriminator的inputs是一对pair，它的outputs是这对pair是否是match。如下图所示

<div class="md-flex-h">
<div><img src="{{site.baseurl}}/assets/images/2019/08/gan_11.png"></div>
<div class="md-margin-left-12"><img src="{{site.baseurl}}/assets/images/2019/08/gan_12.png" ></div>
</div>




## Resources

- [Pix2Pix paper](https://arxiv.org/pdf/1611.07004.pdf)
- [DC GAN paper](https://arxiv.org/pdf/1511.06434.pdf)
- [BatchNorm Paper](https://arxiv.org/pdf/1502.03167.pdf)
- [Udacity Deep Learning](https://classroom.udacity.com/nanodegrees/nd101)


