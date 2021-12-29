---
list_title: PyTorch | Implement GAN Using PyTorch
title: PyTorch实现GAN
layout: post
mathjax: true
categories: ["PyTorch", "Machine Learning","Deep Learning"]
---

这篇文章我们用PyTorch实现一个GAN。GAN model由两部分构成，Generator和Discriminator。 Generator的输入是一个random的tensor，它的作用是通过training不断校正自己的输出，直到Disscriminator无法区分它的输出是真的图片还是假的图片。Discriminattor是一个普通的image classifier，用来判断输入的image是真的还是假的。GAN traning的过程可以理解为一个competition，Generator不断训练自己生成接近真实的图片，Discriminator不断训练自己检测假的图片，直到Generator生成出真的图片。这个过程很类似于博弈论中寻找的纳什均衡点。

### Tips for training GAN

1. 使用`leak relu`作为activation函数。leaky relu可以取保gradient可以flow through整个network。这个对于GAN model很重要，因为Generator在train的时候需要Discriminator的gradient信息

2. Generator的输出通常是用`tanh`


### MNIST GAN

我们还是先从最简单的MNIST dataset开始。我们用GAN来生成手写的数字，整个Architecture如下所示

<img class="md-img-center" src="{{site.baseurl}}/assets/images/2019/08/gan-01.png">

由于MNIST数据量比较小，我们可以使用FC layer作为hidden layer即可。 我们需要使用`leak relu`作为activation函数， 因为leaky relu可以确保gradient可以flow through整个network。这个对于GAN model很重要，因为Generator在train的时候需要Discriminator的gradient信息。Generator的输出通常是用`tanh`。

在Training时，我们要同时计算来自Generator和Discriminator的两个loss。其中Discriminator比较直观，因为它只输出图片是真还是假，因此它就是一个binary classifier。loss函数可以用cross-entropy loss。对于Discriminator来说，这里有一个比较tricky的地方，即

```python
prob = F.sigmoid(logits) #logits是最后一个FC的输出
loss != nn.BCELoss(prob, labels)
loss =  nn.BCEWithLogitsLoss(prob, labels*0.9)
```
对于Generator来说，loss函数和Discriminator一样，只要label flip一下即可 (0变成1)

```python
d_loss =  nn.BCEWithLogitsLoss(prob, labels*0.9)
g_loss =  nn.BCEWithLogitsLoss(prob, flipped_labels)
```
Optimizer我们使用Adam，接下来我们先定义这两个model

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




### DC GAN

DC GAN将hidden layer替换成conv layer。对于Generator，它的input是一个很小的feature map，conv layer使用transpose conv来不断增大feature map的spatial size，


