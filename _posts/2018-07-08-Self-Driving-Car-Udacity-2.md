---
updated: "2018-08-10"
layout: post
list_title: 自动驾驶入门 | Autonomous Driving | 卡尔曼滤波 | Kalman Filter
title: 卡尔曼滤波
categories: [AI,Autonomous-Driving]
mathjax: true
---

<img class="md-img-center" src="{{site.baseurl}}/assets/images/2018/07/ad-location.png">

> Course Notes from Udacity Self-Driving Car Nanodegree Program

### Prerequisite

- [Beyesian Model](https://xta0.me/2018/07/06/Self-Driving-Car-Udacity-1.html) 
- [Probability Distribution](https://zh.wikipedia.org/wiki/%E6%A6%82%E7%8E%87%E5%88%86%E5%B8%83)
- [Python Numpy](https://xta0.me/2017/05/10/Data-Science-Tools.html)
- [Kalman Filter](https://zh.wikipedia.org/zh-hans/%E5%8D%A1%E5%B0%94%E6%9B%BC%E6%BB%A4%E6%B3%A2)


本将继续讨论自动驾驶中用到的一些基本统计学知识，包括概率分布模型，卡尔曼滤波器以及简单的物理运动模型。本文假设你已经掌握了了一些简单的概率学知识，并且知道如何使用Python和Numpy进行简单的编程。

### 概率分布模型

概率分布模型是概率的一种数学描述，和任何一种数学公式一样，概率模型可以被二维可视化，也可以用于线性代数和微积分的计算。概率统计模型有两种，一种是离散概率，一种是连续概率。其分布曲线如下图所示:

{% include _partials/components/lightbox.html param='/assets/images/2018/07/ad-6.jpg' param2='1' %}

- 对于离散概率分布函数`g(x)`，有如下性质
    1. 对于所有的`x`值，对应的y值均大于等于`0`
    2. 对于任意`x`值，其对应的y值表示该x事件发生的概率`p(x)`
    3. 所有`g(x)`值的和为`1`
    4. 对任意`g(x)`的值均小于等于`1`
- 对于连续概率分布函数`f(x)`，有如下性质
    1. 对于所有的`x`值，对应的y值均大于等于`0`
    2. 对于任意`x`值，它的概率为`0`
    3. 对于某个区间内的event发生的概率为该区间的面积
    4. `f(x)`的面积为`1`
    5. 对任意`f(x)`的值可以大于`1`

## Localization的原理

在上一篇文章的末尾，我们曾简单的提到了机器人定位自己位置的大概原理，通常的做法是先使用多个GPS进行定位，然后再结合自身各种传感器采集的数据进行位置矫正。一种矫正的方法是使用到卡尔曼滤波，这一节我们先通过一个简单的一维离散模型初步认识一下卡尔曼滤波的过程。

### Sense

假设我们有一个机器人，想要从A点走到B点，AB之间可以平均分成5个格子，则在没有任何信息输入的情况下，机器人认为自己位于AB之间某一位置的概率为`1/5=0.2`，这个概率称为先验概率。 

<img class="md-img-center" src="{{site.baseurl}}/assets/images/2018/07/ad-uniform-1.png">

我们可以用一个Python数组表示每个位置的概率为:

```python
p=[0.2,0.2,0.2,0.2,0.2]
```

而实际上我们的机器人是装有传感器的，他可以感知每个格子中的颜色。由前一篇文章可知，通过传感器，机器人可以引入额外的信息，根据贝叶斯公式，这些信息可以帮助我们矫正先验概率，从而得到更为准确的后验概率。

我们假设AB之间的5个格子为绿色或者红色中的某一种，机器人的传感器可以识别颜色，则引入传感器后，机器人可以更精确的描述自己在AB两点之间的位置。比如当传感器感知到红色的时候，机器人知道自己可能大概率处于红色的格子里，那么在机器人眼中红色格子的概率就应该大于0.2，相应的绿色格子的概率就应该小于0.2。

那么该如何量化这个概率呢？答案是使用贝叶斯公式，我们将红色，绿色格子中的先验概率`0.2`乘以各自的矫正因子，假设红色的矫正因子为`0.6`，绿色为`0.2`，则可以得出下面结果：

<img class="md-img-center" src="{{site.baseurl}}/assets/images/2018/07/ad-uniform-2.png">

将上面结果归一化后得到下面结果：

<img class="md-img-center" src="{{site.baseurl}}/assets/images/2018/07/ad-uniform-3.png">

这个结果表示的是<mark>当机器人的传感器感知到红色时</mark>，5个格子的概率分布。其概率含义如下：当机器人看到红颜色时，它位于绿色格子的概率为`1/9`，位于红色格子的概率为`1/3`,即

$$
P(IN\_GREEN | SEEN\_RED) = 1/9 \\
P(IN\_RED | SEEN\_RED) = 1/3 \\
$$

上面从先验概率通过贝叶斯公式得到后验概率的过程称为**Sense**，我们可以接种用Python描述上述过程：

```python
p=[0.2, 0.2, 0.2, 0.2, 0.2] #先验概率
world=['green', 'red', 'red', 'green', 'green'] #格子颜色

#调整因子
pHit = 0.6 
pMiss = 0.2

#Z为传感器残疾结果
def sense(p, Z):
    q=[]
    for i in range(len(p)):
        hit = (Z == world[i])
        q.append(p[i] * (hit * pHit + (1-hit) * pMiss))

    #归一化
    total = sum(q)
    q = map(lambda x:x/total,q)

    return q
```

`sense`函数非常重要，接下来会不断使用这个函数，因此一定要充分理解。回到上面的例子，当机器人sense到自己红色的时候，此时自己内部的概率分布发生了变化，从左图变成了右图

<div class="md-flex-h md-flex-no-wrap">
<div><img src="{{site.baseurl}}/assets/images/2018/07/ad-sense-1.png"></div>
<div><img src="{{site.baseurl}}/assets/images/2018/07/ad-sense-2.png"></div>
</div>

右图说明，机器人大概率知道自己位于第2号或者第3号格子内，但是具体在哪一个格子自己并不清楚，想要知道自己具体在哪个格子里需要再引入额外信息进行判断，于是机器人向前移动了一步，又进行了一次sense

```python
p = move(p,1)
p = sense(p,'green')
```

> 对于move函数下一节会具体介绍，这里可理解为机器人向前移动了一个格子

此时，机器人发现自己移动了一格后，sense到了绿颜色，于是概率分布又发生了变化：


<div class="md-flex-h md-flex-no-wrap">
<div><img src="{{site.baseurl}}/assets/images/2018/07/ad-sense-2.png"></div>
<div><img src="{{site.baseurl}}/assets/images/2018/07/ad-sense-3.png"></div>
</div>

新的概率分布如右图所示，我们发现第4个格子的概率最大，此时机器人便可以明确的知道自己位于第4个格子中了。

当我们有了`sense`函数之后，机器人就可以边移动，边采集数据，然后不断更新每个位置上的概率从而确定自己的位置：

```python
p = sense(p,'red')
p = move(p,1)
p = sense(p,'green')
p = move(p,1)
...
```
简单总结一下，机器人通过传感器不断引入观测数据，从而完成了将原先均匀分布的先验概率提升为包含一定位置信息的后验概率概率。sense的本质上是一种<mark>引入信息的过程</mark>，它会提高系统整体的熵。

### Move

在定位问题上，除了上面提到的sense问题以外，我们还需要考虑另一个问题，就是机器人的移动问题。所谓移动问题是指机器人在前进过程中并不能总是能准确的移动到某个位置。比如，我们想让一个机器人向右移动一格，而当机器人移动时，它有一定的概率出错，比如移动了两格或者没移动，因此移动的不准确同样也会造成概率分布的混乱。

我们举一个具体的例子，还是假设机器人从A移动到B，AB之间有5个格子，而此时，机器人已经明确知道自己在第2个格子中，即概率分布为：

```python
p = [0,1,0,0,0]
```

<img class="md-img-center" src="{{site.baseurl}}/assets/images/2018/07/ad-move-1.png">

假设机器人现在想要向右移动2个格子，但由于会有一定几率的出错，根据多次试验的结果，假设我们统计出了下面数据：

$$
P(X_{i+2} | X_i) = 0.8 \\
P(X_{i+i} | X_i) = 0.1 \\ 
P(X_{i+3} | X_i) = 0.1 \\
$$

上面数据的含义是，假设机器人位于`i`的位置，那么它移动到`i+2`的位置的概率为`0.8`，移动1格或3个的概率为`0.1`。此时当机器人前进2格时，对应的概率分布变成了：

<img class="md-img-center" src="{{site.baseurl}}/assets/images/2018/07/ad-move-2.png">

```python
p = [0,0,0.1,0.8,0.1]
```
可见机器人的每次移动实际上都是丢失信息的，体现在从一个确定的位置（概率为`1`）变为三个位置不确定的位置。可以想象一下，假如机器人不听的移动1000次（假设走到第5个格子之后又回到第1个格子），概率分布会变成怎样呢？我们可以来一起计算一下

```python
def move(p, U):
    q = []
    for i in range(len(p)):
        s = pExact * p[(i-U) % len(p)]
        s = s + pOvershoot * p[(i-U-1) % len(p)]
        s = s + pUndershoot * p[(i-U+1) % len(p)]
        q.append(s)
    return q

p=[1,0,0,0,0]

pExact = 0.8
pOvershoot = 0.1
pUndershoot = 0.1

for _ in range(1,10001):
    p = move(p,1)
    
print(p) #[0.2,0.2,0.2,0.2,0.2]
```
> 如果不理解move函数可以不必关注其细节，只需观察move前后的概率分布变化即可

上面结果可以看出，如果移动的次数足够多，概率分布将变为均匀分布，变化如下图所示

<div class="md-flex-h md-flex-no-wrap">
<div><img src="{{site.baseurl}}/assets/images/2018/07/ad-move-3.png"></div>
<div><img src="{{site.baseurl}}/assets/images/2018/07/ad-move-4.png"></div>
</div>

总结一下，对于move来说，<mark>本质上是一个损失信息的过程</mark>，每次机器人移动都会引入一定的不确定性，移动的次数越多，每个位置的不确定性会越大，系统整体的熵会降低最终达到均匀分布的状态。

### Sense and Move

了解了sense和move，我们不难发现，机器人移动的过程实际上就是不断获得新信息和不断损失信息的交替循环，机器人没移动一步，会通过传感器得到一个观测值来矫正自己的位置，同时又因为移动带来的不确定性而损失掉一部分信息。这个过程可以如下图所示：

<img class="md-img-center" src="{{site.baseurl}}/assets/images/2018/07/ad-sm-1.png">

我们可以接着用代码来模拟下这个过程

```python
p=[0.2, 0.2, 0.2, 0.2, 0.2]
world=['green', 'red', 'red', 'green', 'green']

measurements = ['red', 'green']
motions = [1,1]

for k in range(len(measurements)):
    #sense
    p = sense(p,measurements[k])
    #move
    p = move(p,motions[k])
```
上述代码中，机器人先sense到了`red`，然后向右移动1个格子，接着又sense到了`green`，然后向右又移动了1个格子，此时得到的概率分布如左图所示

<div class="md-flex-h md-flex-no-wrap">
<div><img src="{{site.baseurl}}/assets/images/2018/07/ad-sm-2.png"></div>
<div><img src="{{site.baseurl}}/assets/images/2018/07/ad-sm-3.png"></div>
</div>

观察左图可发现，第5个格子的概率最大，说明机器人知道自己目前在第5个格子中。假如我们将`measurements`改为`['red','red']`，概率分布变为右图，此时机器人知道自己位于第4个格子里，这也符合我们的推测。

### 小结

这一节我们给出了一种确定机器人位置的理论模型和实现方法。我们先来一起回顾下这个过程，首先我们需要有一个位置的先验概率，然后通过`sense`函数来提升先验概率，从概率角度看，sense的过程是使用贝叶斯定理<mark>求乘积</mark>的过程。接下来由于机器人在移动过程中会带来不确定性，因此会损失一部分信息，从概率分布上看，损失信息的过程是求全概率的过程（求和）。最后机器人通过不断的`sense`和`move`来更新概率密度分布，并找到概率最大的位置作为自己的位置。整个过程如下图所示

<img class="md-img-center" src="{{site.baseurl}}/assets/images/2018/07/ad-kalman-1.png" >

## 一维卡尔曼滤波

如果理解了上面Localization的原理，也就大概理解了卡尔曼滤波的原理，我们后面会给出卡尔曼滤波的数学定义以及推导过程。这一节我们可以对上面的例子再进行一些完善，将离散概率分布函数变成连续的概率分布，因为在现实世界中，位置往往是时间的函数，而时间是连续的，因此我们需要用一个连续概率函数对上述场景进行模拟。但不论离散还是连续，其定位的原理是不变的，都是sense和move两个过程的交替更迭。

### 高斯函数

回顾前面介绍Localization的步骤，第一步是需要有一个先验概率，一个连续型的概率密度函数，这个函数可以使用高斯函数来表示，

$$
f(x) = \frac{1}{\sqrt{2\pi\sigma^2}} exp ^{-\frac{(x-\mu)^2}{\sigma^2}}
$$

对于高斯函数来说，方差越大，概率分布函数越“胖”，样本偏离均值的幅度越大；相反的，方差越小，样本偏离均值的幅度越小，概率分布函数越“瘦”。如果用高斯函数来描述机器人的位置，我们希望方差越小越好，这样均值位置就越接近机器人的真实位置。

> 更多关于高斯函数的特性请自行查阅文档。

### Measurement Update

回到定位问题上，假设我们有一个先验的高斯概率密度函数为$f(x)$，其均值和方差分别为`(mu=120, sigma=40)`，校验因子的概率密度函数为$g(x)$，其均值和方差分别为`(mu=200,sigma=30)`，如下图所示

<img class="md-img-center" src="{{site.baseurl}}/assets/images/2018/07/ad-gs-1.png" width="70%">

接下来机器人进行了一个次measure，根据贝叶斯公式，后验概率应该等于$f(x)*g(x)$，得到的结果如下图

<img class="md-img-center" src="{{site.baseurl}}/assets/images/2018/07/ad-gs-2.png" width = "70%">

从这个结果中，我们能观察出几条重要的结论：

1. 计算后的高斯分布的均值位于`120`和`200`之间
2. 方差变的更小，峰值为三者最高，说明measure获取了新的信息，提高了位置预测准确率

产生这个结果的原因从直观上不太好理解，实际上可以从数学的角度进行证明，当两个高斯函数相乘后，产生的新的均值为

$$
\mu = \frac{\sigma^{-2}_1\mu_1 + \sigma^{-2}_2\mu_2}{\sigma^{-2}_1 + \sigma^{-2}_2}
$$

由于$\mu_2 > \mu_1$，很直观的有 $\mu_1<\mu<\mu_2$。相应的，产生的新的方差为

$$
\sigma^2 = \frac{\sigma^2_1 + \sigma^2_2}{\sigma^2_1+\sigma^2_2}
$$

这个看起来没有那么直观，我们不妨代入几个数试试，令$\sigma^2_1 = \sigma^2_2 = 4$，$\sigma=2$的值为2；令$\sigma^2_1 = 8$,$\sigma^2_2 = 2$，$\sigma=1.6$，以此类推。

measure过程的Python代码如下：

```python
def measure(mean1, var1, mean2, var2):
    new_mean = (var2 * mean1 + var1 * mean2) / (var1 + var2)
    new_var = 1/(1/var1 + 1/var2)
    return (new_mean, new_var)
```
为了简化计算，上述代码只返回了均值和方差，如果要计算对于高斯函数，将其带入即可

### Motion Update

sense完之后就可以开始move了，move是一个损失信息的过程，为了量化损失的信息，我们引入一个损失概率密度函数，该函数也服从高斯分布，我们用`(u,r)`表示，其中`u`是均值，`r`是方差。为了计算损耗后的概率分布，我们只需要让当前的概率密度函数和损失概率密度函数“求和”即可：

$$
\mu' = \mu + u \\
\sigma'^2 = \sigma^2 + r^2\\
$$

因此我们可以写出move的代码:

```python
def predict(mean1, var1, mean2, var2):
    new_mean = mean1 + mean2
    new_var = var1 + var2
    return (new_mean, new_var) 
```

### sense & move

将生面的sense和move结合起来，我们就得到了<mark>一维卡尔曼滤波</mark>的完整过程:

```python
import math
from Kalman.predict import predict
from Kalman.measure import measure

##先验概率
mu = 0
sig = 1000

##矫正因子
measurements = [5.0,6.0,7.0,9.0,10.0]
measurement_sig = 4.0

##损耗因子
motion = [1.0,1.0,2.0,1.0,1.0]
motion_sig = 2.0

## sense and move
for index in range(0,len(measurements)):
    #sense
    (mu,sig) = measure(mu,sig, measurements[index],measurement_sig)
    #move
    (mu,sig) = predict(mu,sig, motion[index],motion_sig)
```
上述代码逻辑和前面离散demo基本一致，不同的是概率密度函数变成了连续的高斯函数，这里就不再做过多的分析。下图为移动5次后的概率分布结果

<img class="md-img-center" src="{{site.baseurl}}/assets/images/2018/07/ad-gs-sm-1.png" width = "70%">


## 多维卡尔曼滤波

为了真正模拟机器人或者汽车在现实中的运动，我们需要使用多维的卡尔曼滤波器。卡尔曼滤波器主要有两方面的作用：

1. 状态估计，对于某些无法直接测量的状态，可以使用卡尔曼滤波器进行估计
2. 对输入的信息做最优的状态估计


{% include _partials/post-footer-2.html %}

## Resources

- [Udacity Intro to Self-Driving Car](https://www.udacity.com/course/intro-to-self-driving-cars--nd113)
- [The Kalman Filter](https://www.youtube.com/playlist?list=PLX2gX-ftPVXU3oUFNATxGXY90AULiqnWT)
- [Understand Kalman Filter](https://www.youtube.com/watch?v=mwn8xhgNpFY&list=PLn8PRpmsu08pzi6EMiYnR-076Mh-q3tWr)