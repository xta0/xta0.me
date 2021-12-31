---
list_title: PyTorch | Cycle GAN
title: Cycle GAN
layout: post
mathjax: true
categories: ["PyTorch", "Machine Learning","Deep Learning"]
---

## Cycle GAN from High level

在了解Cycle GAN之前，我们先要了解下Pix2Pix GAN。Pix2Pix GAN解决的问题是image mapping，即Generator将一张图片`x`映射成另一张图片`y`。这就需要我们的training data是一个组pair images。其中，$x_i$是Generator的输入，$y_i$是ground true。我们的目标是训练Generator，使$G(x_i) = y_i$。

Paper使用Unet作为Generator的architecture。Input先经过一个encoder变成小的feature maps，再经过decoder将尺寸复原，最后通过Discriminator进行classification。

<img class="md-img-center" src="{{site.baseurl}}/assets/images/2019/08/gan_09.png">

上面的结构有一个问题是，Discriminator如何判断Generator的输出是否是fake。举例来说，理想情况下，$G(x_1)=y_1$，但是如果出现$G(x_1)=y_2$的情况，由于$y_2$是real image，Discriminator也会将其判定为true。因此，Discriminator的输入应该是一对pair，它的outputs是这对pair是否是match。如下图所示

<div class="md-flex-h md-flex-no-wrap">
<div><img src="{{site.baseurl}}/assets/images/2019/08/gan_11.png"></div>
<div class="md-margin-left-12"><img src="{{site.baseurl}}/assets/images/2019/08/gan_12.png" ></div>
</div>

实际应用中，这种一对一的pair data是很难得到的。比较容易得到的是两组image set - $X$和$Y$，比如一组马的图片和一组斑马的图片。此时我们需要找到一个映射使 $G(X) = Y$，即对于$X$中的每个image，都有$G(x) = y$。此时不是一对一的映射，而是多对多的映射，如下图所示

<div class="md-margin-left-12"><img src="{{site.baseurl}}/assets/images/2019/08/gan_13.png" ></div>

如果仔细思考，这里会有一个问题，既然是多对多，就会有多个$x$映射到同一个$y$的可能，为了避免这种情况，我们还要建立一个反向约束，即$G(Y) = X$，此时有

$$
G_{YtoX}(G_{XtoY(x)}) \approx x
$$

既然需要reverse mapping，我们就需要两组GAN，一组完成$G(X)=Y$，另一组完成$G(Y)=X$。因此我们就有两组Adversarial Loss $L_X$和$L_Y$。此外，为了满足上面的式子，我们还需要引入一个Cycle Consistency Loss来确保生成出的图片被revert回去后可以和原图近似。

<div class="md-margin-left-12"><img src="{{site.baseurl}}/assets/images/2019/08/gan_14.png" ></div>

Cycle Consistency Loss同样也有两组，分别为forward consistency loss 和 backward consistency loss，分别对应$x$和$y$。Cycle GAN的总的loss为

$$
L_Y + L_X + \lambdaL_{cyc}
$$


## 实现Cycle GAN

## Resources

- [Pix2Pix paper](https://arxiv.org/pdf/1611.07004.pdf)
- [Cycle GAN paper](https://arxiv.org/pdf/1703.10593.pdf)
- [Augmented CycleGAN](https://arxiv.org/abs/1802.10151)
- [StarGAN](https://github.com/yunjey/StarGAN)
- [Udacity Deep Learning](https://classroom.udacity.com/nanodegrees/nd101)


