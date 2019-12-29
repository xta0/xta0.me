---
list_title: 学一点 PyTorch | Learn PyTorch | 神经网络 | Neural Networks with PyTorch
title: PyTorch实现神经网络
layout: post
mathjax: true
categories: ["PyTorch", "Machine Learning","Deep Learning"]
---

上一篇文章中我们用PyTorch实现了一个线性回归的模型，这篇文章我们将用神经网络来重新训练我们的模型。虽然我们只有一个feature和极为少量的训练样本，使用神经网络不免有些OverKill了，但使用神经网络的一个有趣之处是我们不知道它最后会帮我们拟合出的什么样的模型。好奇心永远是最好的学习动力，我们下面会用PyTorch的API搭建两个简单的神经网络来重新拟合上一篇文章中的模型，最后我们会做一个全FC网络做数字识别。

### 一个神经元的神经网络

PyTorch中神经网络相关的layer称为module，封装在`torch.nn`中，由于我们的模型是线性的，我们可以用`nn.Linear`这个module，此外由于我们只有一个feature，加上我的输出也是一个值，因此我们的神经网络实际上只有一个神经元，输入是一个tensor，输出也是一个tensor。


```python
import torch
import torch.nn as nn

t_x = [35.7, 55.9, 58.2, 81.9, 56.3, 48.9, 33.9, 21.8, 48.4, 60.4, 68.4]
t_x = torch.tensor(t_x).unsqueeze(1) #convert t_x to [11x1]
t_xn = t_x*0.1

linear_model = nn.Linear(1,1)
output = linear_model(t_xn)
print(output)
```
上述代码中我们创建了一个`linear_model`，这个model只有一个神经元，输入和输出只有一个tensor。接着我们创建了一个`11x1`的input tensor。由于我们的model没有经过训练，因此输出为一堆无意义的tensor。默认情况下`nn.Linear`包含bias，而weight值被初始化为一个随机数

```python
print("weight: ",linear_model.weight) #tensor([[-0.1335]], requires_grad=True)
print("bias: ",linear_model.bias) #tensor([-0.4349], requires_grad=True)
```
接下来我们参考上一篇文章来训练我们的模型

```python
optimizer = optim.SGD(linear_model.parameters(), lr=1e-2)
def train_loop(epochs, learning_rate, loss_fn,x, y):
    for epoch in range(1, epochs + 1):    
        optimizer.zero_grad()
        t_p = linear_model(x)
        loss = loss_fn(y, t_p)
        loss.backward()
        optimizer.step()
        print(f'Epoch: {epoch}, Loss: {float(loss)}')
```
上述代码中和前一节大同小异，有下面几点值得注意

1. 待训练参数$\omega$和$b$保存在`linear_model.parameters()`中
2. 由于params保存在了model中，因此PyTorch知道如何update这些参数，不再需要我们手动编写梯度下降的代码
2. loss函数使用系统自带的`nn.MSELoss`对应上一节的L2 loss函数

由于我们的`linear_module`在数学上就是在计算$y = \omega x + b$，因此我们可以预测训练结果和前文是一致的

```python
train_loop(3000, 1e-2, nn.MSELoss(),t_xn, t_y)
print("params:", list(linear_model.parameters()))
#tensor([[5.3491]], requires_grad=True), Parameter containing:
#tensor([-17.1995], requires_grad=True)]
```
训练结果符合我们预期，感觉这个例子没什么意义，但它告诉我们可以使用神经网络来训练线性回归模型。在下一节我们将继续改进在这个简单的神经网络，从而构建出一个非线性模型。

### 非线性模型

为了更精准的拟合数据，我们可以采用非线性模型，比如高阶的线性回归，但与其手动的定义模型，我们还是用神经网络帮我们寻找这个模型。不同的是这次我们要对数据做一些非线性变换，具体来说是引入一个hidder layer和activation函数。我们新的model结构如下

```python
seq_model = nn.Sequential(
    nn.Linear(1,13),
    nn.Tanh(),
    nn.Linear(13,1))

print(seq_model)
# Sequential(
#   (0): Linear(in_features=1, out_features=13, bias=True)
#   (1): Tanh()
#   (2): Linear(in_features=13, out_features=1, bias=True)
# )
```
上述代码中我们引入了一个1*13的hidden layer，然后跟了一个`Tanh()`的非线性变换作为activation，最后的output layer又把结果变成`1x1`tensor，整个model的结构如下图所示

<img class="md-img-center" src="{{site.baseurl}}/assets/images/2019/07/pytorch-1.png">

在训练我们的模型之前，我们先来分析下有多少个待学习参数。显然对于hidden layer我们有13个$\omega$，13个$b$，对于output layer，我们有一个$\omega$和一个$b$。我们也可以用下面代码来验证

```python
for name,param in seq_model.named_parameters():
    print(name, param.shape)

# 0.weight torch.Size([13, 1])
# 0.bias torch.Size([13])
# 2.weight torch.Size([1, 13])
# 2.bias torch.Size([1])
```
上述代码会打印出整个network中待学习的参数。接下来我们用同样的代码训练我们的model

```python
optimizer = optim.SGD(seq_model.parameters(), lr=1e-3)
def train_loop(epochs, learning_rate, loss_fn,x, y):
    for epoch in range(1, epochs + 1):    
        optimizer.zero_grad()
        t_p = seq_model(x)
        loss = loss_fn(y, t_p)
        loss.backward()
        optimizer.step()
        print(f'Epoch: {epoch}, Loss: {float(loss)}')

train_loop(5000, 1e-3, nn.MSELoss(),t_xn, t_y)
```
5000次迭代后，loss收敛在1.950253963470459，接下来我们来可视化一下我们model，并和上一篇的线性的模型做个比较

<div class="md-flex-h md-flex-no-wrap md-margin-bottom-12">
<div><img src="{{site.baseurl}}/assets/images/2019/06/pytorch-lr-1.png"></div>
<div class="md-margin-left-12"><img src="{{site.baseurl}}/assets/images/2019/07/pytorch-2.png"></div>
</div>

上图中实心的点为我们的原始数据，绿色的曲线是神经网络拟合出的曲线，标记为x的点为预测值。

> 由于样本点少，暂不考虑过拟合的问题。

小结一下，这一节我们用PyTorch构建了一个两层的神经网络，训练了一个非线性模型，解决了一个简单的回归问题。但上述网络还是太过简单，在下面一节中我们将构建一个稍微复杂一点的网络解决数字识别问题。

### MNIST

这一节我们要设计一个神经网络解决识别数字问题，实际上这是一个很经典的问题了，我们用的训练集为著名的MNIST，如下图所示

<div><img src="{{site.baseurl}}/assets/images/2019/06/pytorch-1-2.png"></div>

上图中每个数字图片都是一个灰度图，我们的目标便是构建一个神经网络对上图中每个图片都能识别其中的数字。首先我们要将数据集下载下来

```python
# Define a transform to normalize the data
transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.5,), (0.5,))])
# Download and load the training data
trainset = datasets.FashionMNIST('./F_MNIST_data/', download=True, train=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

# Download and load the test data
testset = datasets.FashionMNIST('./F_MNIST_data/', download=True, train=False, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=True)
```
上述`trainloader`中包含了训练用的图片文件和标注，由于图片是我们神经网络的输入，因此我们需要看一下trainloader中图片的size，遍历trainloader我们可以使用下面方法

```python
for images, lable in trainloader:
    print(images.shape) #torch.Size([64, 1, 28, 28])
    print(labels.shape) #torch.Size([64])
```
可以看到images的格式是按照NCWH排列，表示有64组图片，每张图只有一个channel，长宽均为28。

了解了输入tensor的size之后，我们便可以着手设计模型了。首先这是一个分类问题，因此我们的最后一层可以用softmax做分类，前面我们可以用三层FC做hiddnen layer，如下

```python
FC (784,128)
ReLU()
FC (128,64)
ReLU()
FC (64,10)
Softmax()
```
和前面不同的是，这次的输出是一个分类问题，因此我们的loss函数要选取不同的，对于Softmax我们可以用`nn.CrossEntropyLoss()`。但根据[文档](https://pytorch.org/docs/stable/nn.html#torch.nn.CrossEntropyLoss)

> This criterion combines `nn.LogSoftmax()` and `nn.NLLLoss()` in one single class. The input is expected to contain scores for each class.

因此我们需要将FC的输出直接传给`CrossEntropyLoss()`,而不是softmax的输出。实际应用中，我们更希望将两者分开，因此这里我们使用`nn.NLLLoss()`。确定了loss函数后，我们可以测试下我们的model

```python
model = nn.Sequential(nn.Linear(784,128),
                      nn.ReLU(),
                      nn.Linear(128,64),
                      nn.ReLU(),
                      nn.Linear(64,10),
                      nn.LogSoftmax()
)

loss_fn = nn.NLLLoss()
images,labels = next(iter(trainloader))
input = images.view(images.shape[0],-1) #[64 x 784]
output = model(input)
loss = loss_fn(output, labels)
print(loss) #tensor(2.3290, grad_fn=<NllLossBackward>)
```





