---
list_title: PyTorch | Cycle GAN
title: Cycle GAN
layout: post
mathjax: true
categories: ["PyTorch", "Machine Learning","Deep Learning"]
---

### Cycle GAN

在了解Cycle GAN之前，我们先要了解下Pix2Pix GAN。Pix2Pix GAN解决的问题是image mapping，即Generator将一张图片`x`映射成另一张图片`y`。这就需要我们的training data是pair images。其中，$x_i$是Generator的输入，$y_i$是ground true。我们的目标是训练Generator，使$G(x_i) = y_i$。

Paper使用Unet作为Generator的architecture。Input先经过一个encoder变成小的feature maps，再经过decoder将尺寸复原。

<img class="md-img-center" src="{{site.baseurl}}/assets/images/2019/08/gan_09.png">

decoder输出的图片再通过Discriminator进行classification。Discriminator的结构和前一篇文章中的DC GAN类似

<img class="md-img-center" src="{{site.baseurl}}/assets/images/2019/08/gan_10.png">

上面Generator结构有一个问题是，Discriminator如何判断Generator的输出是否是fake。举例来说，理想情况下，$G(x_1)=y_1$，但是如果出现$G(x_1)=y_2$的情况，由于$y_2$是real image，Discriminator也会将其判定为true。因此，Discriminator的输入应该是一对pair，它的outputs是这对pair是否是match。如下图所示

<div class="md-flex-h md-flex-no-wrap">
<div><img src="{{site.baseurl}}/assets/images/2019/08/gan_11.png"></div>
<div class="md-margin-left-12"><img src="{{site.baseurl}}/assets/images/2019/08/gan_12.png" ></div>
</div>

实际应用中，这种一对一的pair data是很难得到的。比较容易得到的是两组image set - $X$和$Y$，比如一组马的图片和一组斑马的图片。此时我们需要找到一个映射使 $G(X) = Y$，即对于$X$中的每个image，都有$G(x) = y$。此时不是一对一的映射，而是多对多的映射，是**unsupervised learning**，如下图所示

<div class="md-margin-left-12"><img src="{{site.baseurl}}/assets/images/2019/08/gan_13.png" ></div>

如果仔细思考，这里会有一个问题，既然是多对多，就会有多个$x$映射到同一个$y$的可能，为了避免这种情况，我们还要建立一个反向约束，即$G(Y) = X$，此时有

$$
G_{YtoX}(G_{XtoY(x)}) \approx x
$$

既然需要reverse mapping，我们就需要两组GAN，一组完成$G(X)=Y$，另一组完成$G(Y)=X$。因此我们就有两组Adversarial Loss $L_X$和$L_Y$。此外，为了满足上面的式子，我们还需要引入一个Cycle Consistency Loss来确保生成出的图片被revert回去后可以和原图近似。

<div class="md-margin-left-12"><img src="{{site.baseurl}}/assets/images/2019/08/gan_14.png" ></div>

Cycle Consistency Loss同样也有两组，分别为forward consistency loss 和 backward consistency loss，分别对应$x$和$y$。

$$
total loss = L_Y + L_X + \lambda L_{cyc}
$$

### Discriminator

Cycle GAN需要两个Discriminator和Generator。Discriminator的结构前文DC GAN相似，区别在于Input不需要经过Fully Connected Layer。我们另input的size为`[1, 3, 128, 128]`

```python
class Discriminator(nn.Module):
    def __init__(self, conv_dim=64):
        super(Discriminator, self).__init__()
        # Should accept an RGB image as input and output a single value
        self.conv_dim = conv_dim
        self.conv1 = conv(3, conv_dim, 4, batch_norm=False) #(64, 64, 64)
        self.conv2 = conv(conv_dim, conv_dim * 2, 4) # (32, 32, 128)
        self.conv3 = conv(conv_dim * 2, conv_dim * 4, 4) # (16, 16, 256)
        self.conv4 = conv(conv_dim * 4, conv_dim * 8, 4) # (8, 8, 512)
        self.conv5 = conv(conv_dim * 8, 1, 4, stride=1, batch_norm=False)

    def forward(self, x):
        # define feedforward behavior
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = self.conv5(x)
        return x
```

### Generator

Generator的结构有一点特殊，我们需要在encoder和decoder之间引入skip connection，即Residual Block。其目的是解决梯度消失的问题，一个Residual block定义如下

<div class="md-margin-left-12"><img src="{{site.baseurl}}/assets/images/2019/08/gan_15.png" ></div>

通常一个block有两个conv组成，每个conv使用`3x3`的kernel，且input和output的channel相同，因此当`x`通过两个`conv`的layer之后，得到的`y`的shape不会发生任何变化，因此可以和`x`进行elementwise相加。

```python
class ResidualBlock(nn.Module):
    """Defines a residual block.
       This adds an input x to a convolutional layer (applied to x) with the same size input and output.
       These blocks allow a model to learn an effective transformation from one domain to another.
    """
    def __init__(self, conv_dim):
        super(ResidualBlock, self).__init__()
        self.conv1 = conv(conv_dim, conv_dim, kernel_size=3, stride=1, padding=1, batch_norm=True)
        self.conv2 = conv(conv_dim, conv_dim, kernel_size=3, stride=1, padding=1, batch_norm=True)
        
    def forward(self, x):
        y = F.relu(self.conv1(x))
        y = x + self.conv2(y)
        return y
```

接下来我们根据前面提到的Generator的结构创建model，paper中使用6个residual block，input size同样为`[1,3,128,128]`

```python
class CycleGenerator(nn.Module):
    def __init__(self, conv_dim=64, n_res_blocks=6):
        super(CycleGenerator, self).__init__()
        # 1. Define the encoder part of the generator
        self.conv1 = conv(3, conv_dim, 4)
        self.conv2 = conv(conv_dim, conv_dim*2, 4)
        self.conv3 = conv(conv_dim*2, conv_dim*4, 4)
        # 2. Define the resnet part of the generator
        res = []
        for _ in range(n_res_blocks):
            res.append(ResidualBlock(conv_dim * 4))
        self.res_blocks = nn.Sequential(*res)
        # 3. Define the decoder part of the generator
        self.deconv1 = deconv(conv_dim*4, conv_dim*2, 4)
        self.deconv2 = deconv(conv_dim*2, conv_dim, 4)
        self.deconv3 = deconv(conv_dim, 3, 4, batch_norm=False)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.res_blocks(x)
        x = F.relu(self.deconv1(x))
        x = F.relu(self.deconv2(x))
        x = F.tanh(self.deconv3(x))
        return x
```
## Discriminator and Generator Losses

前面提到，Cycle GAN的loss由两部分组成，一部分是Adversarial loss，前面文章已经讨论过，但是Cycle GAN的Discriminator不能使用Cross Entropy loss，原因paper中提到会有梯度消失的问题，因此，这里需要使用mean square loss. Cycle GAN loss的另一分部分是Cycle Consistency loss，这个loss比较简单，只是比较生成的图片和原图片每个像素点的差异即可。各种loss计算如下

```python
def real_mse_loss(D_out):
    # how close is the produced output from being "real"?
    return torch.mean((D_out-1)**2)

def fake_mse_loss(D_out):
    # how close is the produced output from being "false"?
    return torch.mean(D_out**2)

def cycle_consistency_loss(real_im, reconstructed_im, lambda_weight):
    # calculate reconstruction loss 
    # as absolute value difference between the real and reconstructed images
    reconstr_loss = torch.mean(torch.abs(real_im - reconstructed_im))
    # return weighted loss
    return lambda_weight*reconstr_loss   
```

## 训练Discriminator

训练Discriminator的步骤和前文的DC GAN基本类似，不同的是Cycle GAN要train两个Discriminator用来判断X和Y。可以follow下面步骤

```python
##   First: D_X, real and fake loss components   ##
# Train with real images
d_x_optimizer.zero_grad()
# 1. Compute the discriminator losses on real images
out_x = D_X(images_X)
D_X_real_loss = real_mse_loss(out_x)
# Train with fake images
# 2. Generate fake images that look like domain X based on real images in domain Y
fake_X = G_YtoX(images_Y)
# 3. Compute the fake loss for D_X
out_x = D_X(fake_X)
D_X_fake_loss = fake_mse_loss(out_x)
# 4. Compute the total loss and perform backprop
d_x_loss = D_X_real_loss + D_X_fake_loss
d_x_loss.backward()
d_x_optimizer.step()

##   Second: D_Y, real and fake loss components   ##
# Train with real images
d_y_optimizer.zero_grad()
# 1. Compute the discriminator losses on real images
out_y = D_Y(images_Y)
D_Y_real_loss = real_mse_loss(out_y)
# Train with fake images
# 2. Generate fake images that look like domain Y based on real images in domain X
fake_Y = G_XtoY(images_X)
# 3. Compute the fake loss for D_Y
out_y = D_Y(fake_Y)
D_Y_fake_loss = fake_mse_loss(out_y)
# 4. Compute the total loss and perform backprop
d_y_loss = D_Y_real_loss + D_Y_fake_loss
d_y_loss.backward()
d_y_optimizer.step()
```






## Resources

- [Pix2Pix paper](https://arxiv.org/pdf/1611.07004.pdf)
- [Cycle GAN paper](https://arxiv.org/pdf/1703.10593.pdf)
- [LSGAN](https://arxiv.org/pdf/1611.04076.pdf)
- [Augmented CycleGAN](https://arxiv.org/abs/1802.10151)
- [StarGAN](https://github.com/yunjey/StarGAN)
- [Why Skip Connection is important in Image Segmentation](https://arxiv.org/abs/1608.04117)
- [Udacity Deep Learning](https://classroom.udacity.com/nanodegrees/nd101)


