---
list_title: 学一点 PyTorch | Learn PyTorch | 神经网络 | Neural Networks with PyTorch
title: PyTorch实现神经网络
layout: post
mathjax: true
categories: ["PyTorch", "Machine Learning","Deep Learning"]
---

上一篇文章中我们用PyTorch实现了一个线性回归的模型，这篇文章我们将用神经网络来代替线性回归，重新训练我们的模型。虽然我们只有一个feature和少量个训练样本，使用神经网络不免有些OverKill了，但使用神经网络的一个有趣之处是我们不知道它最后会帮我们拟合出的什么样的模型。好奇心永远是最好的学习动力，我们下面会用PyTorch的API搭建两个简单的神经网络来拟合出上一篇文章中的模型，最后我们会做一个全FC网络做数字识别。

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
上述代码中我们创建了一个`linear_model`，这个model只有一个神经元，输入和输出均为一个tensor。接着我们创建了一个`11x1`的input tensor。由于我们的model没有经过训练，因此输出为一堆无意义的tensor。默认情况下`nn.Linear`包含bias，weight被初始化为一个随机数

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
2. 由于params保存在了model中，因此PyTorch知道如何update这些参数，而不需要我们手动的进行梯度下降
2. loss函数使用系统自带的`nn.MSELoss`对应上一节的L2 loss函数

由于我们的`linear_module`在数学上就是在计算$y = \omega x + b$，因此我们可以预测训练结果和前文是一致的

```python
train_loop(3000, 1e-2, nn.MSELoss(),t_xn, t_y)
print("params:", list(linear_model.parameters()))
#tensor([[5.3491]], requires_grad=True), Parameter containing:
#tensor([-17.1995], requires_grad=True)]
```
训练结果符合我们预期，这说明我们可以使用神经网络来训练线性回归模型，在下一节我们将看到如何使用神经网络构建一个非线性回归的神经网络

### 非线性模型

为了更精准的拟合数据，我们实际上还可以采用非线性模型，比如高阶的线性回归，但与其手动的定义model，我们还是用神经网络帮我们寻找这个模型。不同的是这次我们要对数据做一些非线性变换

### MNIST

这一节我们用全FC层构建一个神经网络来识别数字，

