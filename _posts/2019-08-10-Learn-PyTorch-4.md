---
list_title: PyTorch | Pixel2Pixel and Cycle GAN
title: PyTorch实现GAN
layout: post
mathjax: true
categories: ["PyTorch", "Machine Learning","Deep Learning"]
---

GAN model由两部分构成，Generator和Discriminator。 Generator的输入是一个random的tensor，它通过training不断校正自己的输出，直到输出的图片接近于真实的图片。Discriminattor是一个普通的image classifier，用来判断输入的图片是真的还是假的。整个GAN model训练的过程可以理解为一个competition，Generator不断训练自己生成接近真实的图片，Discriminator不断训练自己检测假的图片，直到迫使Generator生成出接近真实的图片。这个过程很类似于博弈论中寻找的纳什均衡点。

### The MNIST GAN Model

我们还是先从最简单的MNIST dataset开始。我们训练一个GAN model用来生成数字，整个Architecture如下所示

<img class="md-img-center" src="{{site.baseurl}}/assets/images/2019/08/gan_01.png">

由于MNIST数据量比较小，我们可以使用FC layer作为hidden layer即可。对于GAN model有几个比较特殊的地方

- 我们需要使用`Leaky ReLU`作为activation函数。因为`Leaky ReLU`可以确保gradient可以flow through整个network。这个对于GAN model很重要，因为Generator在train的时候需要Discriminator的gradient信息。
- Generator的输出通常是用`tanh`，将tensor限制在`(-1, 1)`，这是由于Generator的output通常是Discriminator的input。对于Discriminator，它输入的MNIST图片也需要映射到`(-1, 1)`

综上，Generator和Discriminator的model结构并不复杂，定义如下

```python
class Discriminator(nn.Module):
    def __init__(self, input_size, hidden_dim, output_size):
        super(Discriminator, self).__init__()
        # define hidden linear layers
        self.fc1 = nn.Linear(input_size, hidden_dim*4)
        self.fc2 = nn.Linear(hidden_dim*4, hidden_dim*2)
        self.fc3 = nn.Linear(hidden_dim*2, hidden_dim)
        # final fully-connected layer
        self.fc4 = nn.Linear(hidden_dim, output_size)
        # dropout layer 
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        # flatten image
        x = x.view(-1, 28*28)
        # all hidden layers
        x = F.leaky_relu(self.fc1(x), 0.2) # (input, negative_slope=0.2)
        x = self.dropout(x)
        x = F.leaky_relu(self.fc2(x), 0.2)
        x = self.dropout(x)
        x = F.leaky_relu(self.fc3(x), 0.2)
        x = self.dropout(x)
        # final layer
        out = self.fc4(x)
        return out

class Generator(nn.Module):

    def __init__(self, input_size, hidden_dim, output_size):
        super(Generator, self).__init__()
        # define hidden linear layers
        self.fc1 = nn.Linear(input_size, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim*2)
        self.fc3 = nn.Linear(hidden_dim*2, hidden_dim*4)
        # final fully-connected layer
        # restore the size to 28*28
        self.fc4 = nn.Linear(hidden_dim*4, output_size)
        # dropout layer 
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        # all hidden layers
        x = F.leaky_relu(self.fc1(x), 0.2) # (input, negative_slope=0.2)
        x = self.dropout(x)
        x = F.leaky_relu(self.fc2(x), 0.2)
        x = self.dropout(x)
        x = F.leaky_relu(self.fc3(x), 0.2)
        x = self.dropout(x)
        # final layer with tanh applied
        out = F.tanh(self.fc4(x))
        return out
```
Discriminator和Generator的HyperParameter定义如下

```
Discriminator(
  (fc1): Linear(in_features=784, out_features=128, bias=True)
  (fc2): Linear(in_features=128, out_features=64, bias=True)
  (fc3): Linear(in_features=64, out_features=32, bias=True)
  (fc4): Linear(in_features=32, out_features=1, bias=True)
  (dropout): Dropout(p=0.3, inplace=False)
)

Generator(
  (fc1): Linear(in_features=100, out_features=32, bias=True)
  (fc2): Linear(in_features=32, out_features=64, bias=True)
  (fc3): Linear(in_features=64, out_features=128, bias=True)
  (fc4): Linear(in_features=128, out_features=784, bias=True)
  (dropout): Dropout(p=0.3, inplace=False)
)
```

### Train Discriminator

在Training时，我们要同时计算来自Generator和Discriminator的两个loss。其中Discriminator比较直观，因为它的输出是0或者1，代表出图片是真还是假，因此它是一个binary classifier。其loss函数可以用binary cross-entropy loss。 这里有一个比较tricky的地方，我们需要用`BCEWithLogitsLoss`而不是sigmoid + BCELoss

```python
prob = F.sigmoid(logits) #logits是最后一个FC的输出
loss != nn.BCELoss(prob, labels)
loss =  nn.BCEWithLogitsLoss(prob, labels*0.9)
```

> [BCEWithLogitsLoss](https://pytorch.org/docs/stable/generated/torch.nn.BCEWithLogitsLoss.html#torch.nn.BCEWithLogitsLoss) combines a sigmoid activation function and and binary cross entropy loss in one function.

对于Discriminator来说，它有两个inputs，一个是real image，一个是fake image。因此它也有两个loss函数，`real_loss`和`fake_loss`。对于real image，groud truth label是`1`，即`D(real_image)=1`，这里为了让model更generalize，我们另`y=0.9`。对于输入是fake image，我们希望`D(fake_image)=0`，因此`y`是`0`。总的loss为`real_loss + fake_loss`。定义loss函数如下

``` python
def real_loss(D_out, smooth=False):
    batch_size = D_out.size(0)
    # label smoothing
    if smooth:
        # smooth, real labels = 0.9
        labels = torch.ones(batch_size)*0.9
    else:
        labels = torch.ones(batch_size) # real labels = 1
        
    # numerically stable loss
    criterion = nn.BCEWithLogitsLoss()
    # calculate loss
    loss = criterion(D_out.squeeze(), labels)
    return loss

def fake_loss(D_out):
    batch_size = D_out.size(0)
    labels = torch.zeros(batch_size) # fake labels = 0
    criterion = nn.BCEWithLogitsLoss()
    # calculate loss
    loss = criterion(D_out.squeeze(), labels)
    return loss
```
训练Discriminator，可以follow下面步骤

```python
d_optimizer.zero_grad()
# 1. Compute the discriminator loss on real, training images
D_real = D(real_images)
d_real_loss = real_loss(D_real, smooth=True)
# 2. Generate fake images
z = np.random.uniform(-1, 1, size=(batch_size, z_size))
z = torch.from_numpy(z).float()
fake_images = G(z)
# 3. Compute the discriminator losses on fake images        
D_fake = D(fake_images)
d_fake_loss = fake_loss(D_fake)
#4.  add up loss and perform backprop
d_loss = d_real_loss + d_fake_loss
#5. back prop + optimization
d_loss.backward()
d_optimizer.step()
```

### Train Generator

对于Generator来说，它的输出`y`是一个fake image，我们将它作为Discriminator的输入。由于Generator的目标是`D(fake_image)=1`，我们需要让Discriminator认为来自Generator的输入是一个real image，因此我们需要用`real_loss`

训练Generator，可以follow下面的步骤

```python
g_optimizer.zero_grad()        
# 1. Generate fake images
z = np.random.uniform(-1, 1, size=(batch_size, z_size))
z = torch.from_numpy(z).float()
fake_images = G(z)

# 2. Compute the discriminator losses on fake images 
# using flipped labels!
D_fake = D(fake_images)
g_loss = real_loss(D_fake) # use real loss to flip labels

# 3. perform backprop
g_loss.backward()
g_optimizer.step()
```
### Training Results

我们选取optimizer为`Adam`, learning rate为`0.002`, `num_epochs = 100` 两个model的training loss的变化如下图

<img class="md-img-center" src="{{site.baseurl}}/assets/images/2019/08/gan_03.png">

我们也可以观察在训练中Generator输出的变化情况。我们可以在每个epoch之后输出一张Generator的图片

<img class="md-img-center" src="{{site.baseurl}}/assets/images/2019/08/gan_04.png">

可见，对于任意一个random tensor，Generator可以生成一个类似手写的数字。当然MNIST是个非常简单的情况，对于复杂一点的图片，我们需要用Convolution layer来代替FC layer。

## Deep Convolutional GAN

DC GAN和上面的MNIST model工作原理基本相同，不同的是DC GAN使用conv layer作为hidden layer。Paper中还列出了architecture设计的几个关键点

1. Replace any pooling layers with strided convolutions (discriminator) and fractional-strided
convolutions (generator).
2. Use batchnorm in both the generator and the discriminator.
3. Remove fully connected hidden layers for deeper architectures.
4. Use ReLU activation in generator for all layers except for the output, which uses Tanh.
5. Use LeakyReLU activation in the discriminator for all layers.

### Discriminator

Discriminator的结构和上面的MNIST model类似，它由若干个conv layer构成。和普通的classification model不同的是，它用stride=2的conv来取代pooling。

<img class="md-img-center" src="{{site.baseurl}}/assets/images/2019/08/gan_05.png">

需要注意的是，除了第一个conv layer外，后面的每个conv layer都需要追加BatchNorm操作来帮助training更好的converge。

```python
# helper conv function
def conv(in_channels, out_channels, kernel_size, stride=2, padding=1, batch_norm=True):
    """Creates a convolutional layer, with optional batch normalization.
    """
    layers = []
    conv_layer = nn.Conv2d(in_channels, out_channels, 
                           kernel_size, stride, padding, bias=False)
    # append conv layer
    layers.append(conv_layer)
    if batch_norm:
        # append batchnorm layer
        layers.append(nn.BatchNorm2d(out_channels))
    # using Sequential container
    return nn.Sequential(*layers)

class Discriminator(nn.Module):
    def __init__(self, conv_dim=32):
        super(Discriminator, self).__init__()
        self.conv_dim = conv_dim
        self.conv1 = conv(3, conv_dim, 4)
        self.conv2 = conv(conv_dim, conv_dim*2, 4)                
        self.conv3 = conv(conv_dim*2, conv_dim*4, 4)
        self.fc = (conv_dim*4*4*4, 1)
        
    def forward(self, x):
        # complete forward function
        x = F.leaky_relu(self.conv1(x), 0.2)
        x = F.leaky_relu(self.conv2(x), 0.2)
        x = F.leaky_relu(self.conv3(x), 0.2)
        x = x.view(-1, self.conv_dim*4 * 4 * 4)
        x = self.fc(x)
        return x
```
conv layer的depth可以从32开始，后面逐层double (64, 128, etc).

### Generator

Generator的input也是一个noise vector，它使用transpose conv(stride=2)来增加feature map的spatial size。同样的，我们需要使用对conv layer追加BatchNorm

<img class="md-img-center" src="{{site.baseurl}}/assets/images/2019/08/gan_05.png">

```python
# helper deconv function
def deconv(in_channels, out_channels, kernel_size, stride=2, padding=1, batch_norm=True):
    """Creates a transposed-convolutional layer, with optional batch normalization.
    """
    layers = []
    transposed_conv_layer = nn.ConvTranspose2d(in_channels, out_channels, 
                           kernel_size, stride, padding, bias=False)
    # append conv layer
    layers.append(transposed_conv_layer)
    if batch_norm:
        # append batchnorm layer
        layers.append(nn.BatchNorm2d(out_channels))
    # using Sequential container
    return nn.Sequential(*layers)
        

class Generator(nn.Module):
    def __init__(self, z_size, conv_dim=32):
        super(Generator, self).__init__()
        self.conv_dim = conv_dim
        # complete init function
        self.fc = nn.Linear(z_size, conv_dim*4*4*4)
        self.deconv1 = deconv(conv_dim*4, conv_dim*2, 4)
        self.deconv2 = deconv(conv_dim*2, conv_dim, 4)
        self.deconv3 = deconv(conv_dim*4, 3, 4, batch_norm = False)
        
    def forward(self, x):
        x = self.fc(x)
        x = x.view(-1, self.conv_dim*4, 4, 4)
        x = F.relu(self.deconv1(x), 0.2)
        x = F.relu(self.deconv2(x), 0.2)
        x = F.relu(self.deconv3(x), 0.2)
        x = F.tanh(x)
        
        return x
```
## 小结


## Resources

- [GAN paper]()
- [DC GAN paper](https://arxiv.org/pdf/1511.06434.pdf)
- [BatchNorm Paper](https://arxiv.org/pdf/1502.03167.pdf)
- [Udacity Deep Learning](https://classroom.udacity.com/nanodegrees/nd101)


