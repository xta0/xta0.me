---
list_title: 学一点 PyTorch | Play with PyTorch
title: 学一点 PyTorch
layout: post
mathjax: true
categories: ["AI", "Machine Learning","Deep Learning"]
---

PyTorch是Facebook开源的一套Deep Learning的框架，它的API都是基于Python的，因此对Researcher非常友好。我对PyTorch的理解是它是具有自动求导功能的Numpy，当然PyTorch比Numpy肯定要强大的多。由于PyTorch目前仍在快速的迭代中，并且有着愈演愈烈的趋势，我们今天也来凑凑热闹，学一点PyTorch。

### Linear Regression

我们使用的例子是一个很简单的[线性回归模型]()，假设我们有一组观测数据如下

```python
t_y = [0.5, 14.0, 15.0, 28.0, 11.0, 8.0, 3.0, -4.0, 6.0, 13.0, 21.0]
t_x = [35.7, 55.9, 58.2, 81.9, 56.3, 48.9, 33.9, 21.8, 48.4, 60.4, 68.4]
```
我们的目标是通过模型帮我们学习得到$\omega$和$b$，使下面等式成立

$$
t_y = \omega \times t_x + b
$$

实际上就是对上述的离散点进行线性拟合。我们首先来创建两个tensor

```python
import torch

t_y = torch.tensor(t_y)
t_x = torch.tensor(t_x)
```
接下来我们来定义我们的model和Loss函数

```python
def model(t_x,w,b):
    return w*t_x + b

def loss_fn(t_y, t_p):
    squared_diff = (t_y-t_p)**2
    return squared_diff.mean()
```
我们可以先随机生成一组数据，跑一下

```python
w = torch.randn(1)
b = torch.zeros(1)
t_p = model(t_x, w, b)
loss = loss_fn(t_y,t_p) //tensor(2478.7595)
```
可以看到loss非常大，这样在我们的意料之中，因为我们还没有对model进行训练，接下来我们要做的就是想办法来训练我们的model。由前面机器学习的文章可知，我们需要找到$\omega$，使loss函数的值最小，因此我们需要首先建立loss函数和$\omega$的关系，然后用梯度下降法找到使loss函数的收敛的极小值点，即可得到我们想要的$\omega$的值。$b$的计算同理。以$\omega$为例，回忆梯度下降的公式为

$$
w := w - \alpha \times \frac {dL}{d\omega}
$$

其中$\alpha$为Learning Rate，loss函数对$\omega$的导数，我们可以使用链式求导法则来得到

$$
\frac {dL}{d\omega} = \frac {dL}{d\hat y} \times \frac {d\hat y}{d\omega}
$$

对应到代码中，我们需要定义一个求导函数`grad_fn`

```python
def dloss_fn(t_y, t_p):
    return 2*(t_y-t_p)

def dmodel_dw(t_x,w,b):
    return t_x

def dmodel_db(t_x,w,b):
    return 1.0

def grad_fn(t_x,w,b,t_y,t_p):
    dw = dloss_fn(t_y,t_p) * dmodel_dw(t_x,w,b)
    db = dloss_fn(t_y,t_p) * dmodel_db(t_x,w,b)
    return torch.tensor([dw.mean(), db.mean()]) 
```

有了梯度函数后，我们便可以使用梯度下降接法来训练我们的model了

```python
def train(learning_rate,w,b,x,y):
    t_p = model(x, w, b)
    loss = loss_fn(y, t_p)
    print("loss: ",loss)
    w = w - learning_rate * grad_fn(t_x,w,b,t_y,t_p)[0]
    b = b - learning_rate * grad_fn(t_x,w,b,t_y,t_p)[1]
    t_p = model(x, w, b)
    loss = loss_fn(y, t_p)
    print("loss: ",loss)
    return (w,b)
```
上面我们先执行了一次forward pass，得到了loss，然后进行了一次backward pass，得到了新的$\omega$和$b$，接着我们又进行了一次forward，又得到了一个新的loss值。我们可以猜想，loss值应该会变小，我们训练一次观察一下结果

```python
train(learning_rate=1e-2, 
w=torch.tensor(1.0), 
b=torch.tensor(0.0), 
x=t_x, 
y=t_y)

loss:  1763.8846435546875
loss:  5802484.5
```
我们发现loss的值没有变小，反而变大了。回忆前面机器学习的文章，出现这种情况，有几种可能，比如`learning_rate`的值过大，或者没有对输入数据进行normalization。我们先试着改变步长

```python
train(learning_rate=1e-4, 
w=torch.tensor(1.0), 
b=torch.tensor(0.0), 
x=t_x, 
y=t_y)
#----------------------------
loss:  1763.8846435546875
loss:  323.0905456542969
```
我们发现loss的值变小了，符合我们的预期。接着我们尝试对输入数据进行normalization

```python
t_xn = t_x * 0.1
train(learning_rate=1e-2, 
w=torch.tensor(1.0), 
b=torch.tensor(0.0), 
x=t_xn, 
y=t_y)
#----------------------------
loss:  80.36434173583984
loss:  37.57491683959961
```
> 这里我们可以使用更复杂的normalization方法，这里为了简单起见，直接领输入数据乘以0.1

通过一系列操作，loss已经可以收敛了，这说明我们的梯度下降可以正常工作，接下来我们便可以正式训练了，我们对上面代码稍微重构一下

```python
def train_loop(epochs, learning_rate, params, x, y):
    for epoch in range(1, epochs + 1):
        w,b = params
        t_p = model(x, w, b)
        loss = loss_fn(y, t_p)
        grad = grad_fn(x,w,b,y,t_p)
        params = params - learning_rate * grad
        print(f'Epoch: {epoch}, Loss: {float(loss)}')
    return params

param = train_loop(epochs = 5000, 
learning_rate = 1e-2, 
params = torch.tensor([1.0,0.0]), 
x = t_x, 
y = t_y)
print("w,b",float(param[0]), float(param[1]))
#----------------------------
Epoch: 1, Loss:  80.36434173583984
Epoch: 2, Loss:  37.57491683959961
...
Epoch: 4999, Loss: 2.927647352218628
Epoch: 5000, Loss: 2.927647590637207
#----------------------------
w,b 5.367083549499512 -17.301189422607422
```
我们循环了5000次，loss收敛在`2.927647`左右，不再下降，此时我们可以认为得到的$\omega$和$b$是我们最终想要的，为了更直观的理解，我们将得到模型画出来

<img src="{{site.baseurl}}/assets/images/2019/06/pytorch-lr-1.png">

### PyTorch's autograd

上述代码并没有什么特别的地方，我们手动的实现了对$\omega$和$\b$的求导，但由于上面的model太过简单，因此难度不大。但是对于复杂的model，比如CNN的model，如果用手动求导的方式则会非常复杂，且容易出错。正如我前面所说，PyTorch强大的地方在于不论model多复杂，只要它满足可微分的条件，PyTorch便可以自动帮我们完成求导的计算，即所谓的**autograd**。

简单来说，对所有的Model，我们都可以用一个[Computational Graph](https://xta0.me/2018/01/02/Deep-Learning-1.html)来表示，Graph中的每个节点代表一个运算函数

<img src="{{site.baseurl}}/assets/images/2018/01/dp-w2-1.png">

如上图中的`a`,`b`,`c`为叶子节点，在执行forward pass的时候，PyTorch会记将非叶子节点上的函数，保存在该节点的tensor中，这样当做backward pass的时候便可以很方便的使用链式求导快速计算出叶子节点的导数值。

```python
a = torch.tensor(3.0, requires_grad=True)
b = torch.tensor(5.0, requires_grad=True)
c = torch.tensor(8.0, requires_grad=True)
```
这里我们定义了`a`,`b`,`c`三个tensor，并且告诉PyTorch需要对它们进行求导，接下来我们建立Computation Graph

```python
u = b*c
v = a+u
j = 3*v
```

接下来我们可以验证下每个节点上都有什么信息

```python
>>> u
tensor(40., grad_fn=<MulBackward0>)
>>> j
tensor(129., grad_fn=<MulBackward0>)
>>> v
tensor(43., grad_fn=<AddBackward0>)
```
可以看到每个节点上都有`grad_fn`，用来做autograd。上面的图中，j是最终节点，我们可以让j做backward，则它会触发u和v做反向求导，而求导的结果会存放在a,b,c上

```python
j.backward()
>>> a.grad
tensor(3.)
>>> b.grad
tensor(24.)
>>> c.grad
tensor(15.)
>>> u.grad #none
>>> v.grad #none
>>> j.grad #none
```
上面结果可知，只有leaf节点才会累积求导的结果，中间节点不会保存任何中间结果。接下来我们可以用PyTorch的autograd API重写上一节的训练代码

```python
params = torch.tensor([1.0,0.0], requires_grad=True)

```
接下来我们不需要自己定义求导函数，而是直接可以使用PyTorch的`backward()`来获得params的导数值

```python
# use autograd
def train_loop(epochs, learning_rate, params, x, y):
    for epoch in range(1, epochs + 1):
        if params.grad is not None:
            params.grad_zero()
        w,b = params
        t_p = model(x, w, b)
        loss = loss_fn(y, t_p)
        loss.backward()
        params = (params - learning_rate * params.grad).detach().requires_grad_()
        print(f'Epoch: {epoch}, Loss: {float(loss)}')
    return params


params = torch.tensor([1.0,0.0], requires_grad=True)
param = train_loop(epochs = 5000, 
learning_rate = 1e-2, 
params = params,
x = t_xn, 
y = t_y)
print("w,b",float(param[0]), float(param[1]))
```

## Resoures

- [AUTOGRAD: AUTOMATIC DIFFERENTIATION](https://pytorch.org/tutorials/beginner/blitz/autograd_tutorial.html#sphx-glr-beginner-blitz-autograd-tutorial-py)
