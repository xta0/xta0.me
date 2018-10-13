---
updated: "2018-07-10"
layout: post
list_title: 自动驾驶入门 | Autonomous Driving | 计算位置 | Localizations
title: 卡尔曼滤波器
categories: [AI,Autonomous-Driving]
mathjax: true
---

<img class="md-img-center" src="{{site.baseurl}}/assets/images/2018/07/ad-location.png">

### Prerequisite

- [Beyesian Model](https://xta0.me/2018/07/06/Self-Driving-Car-Udacity-1.html) 
- [Probability Distribution](https://zh.wikipedia.org/wiki/%E6%A6%82%E7%8E%87%E5%88%86%E5%B8%83)
- [Python Numpy](https://xta0.me/2017/05/10/Data-Science-Tools.html)


本将继续讨论自动驾驶中用到的一些基本统计学知识，包括概率分布模型，卡尔曼滤波器以及简单的物理运动模型。本文假设你已经掌握了了一些简单的概率学知识，并且知道如何使用Python和Numpy进行简单的编程。

### 概率分布模型

概率分布模型是概率的一种数学描述，和任何一种数学公式一样，概率模型可以被二维可视化，也可以用于线性代数和微积分的计算。概率统计模型有两种，一种是离散概率，一种是连续概率。其分布曲线如下图所示:

{% include _partials/components/lightbox.html param='/assets/images/2018/07/ad-6.jpg' param2='1' %}

- 对于离散概率分布函数`g(x)`，有如下性质
    1. 对于所有的`x`值，对应的y值均大于等于`0`
    2. 对于任意`x`值，其对应的y值表示该x事件发生的概率`p(x)`
    3. 所有`y`值的和为`1`
    4. 对任意`g(x)`的值小于等于`1`
- 对于连续概率分布函数`f(x)`，有如下四条性质
    1. 对于所有的`x`值，对应的y值均大于等于`0`
    2. 对于任意`x`值，它的概率为`0`
    3. 对于某个区间内的event发生的概率为该区间的面积
    4. `f(x)`的面积为`1`
    5. 对任意`f(x)`的值可以大于`1`

## Localization的原理

接下来我们讨论一个有趣的问题，即机器人如何知道自己在哪？这个问题在上一节最后曾简单的提到的过，通常的做法是先使用多个GPS进行定位，然后再结合自身各种传感器采集的数据进行位置矫正。矫正的过程会使用到贝叶斯定理以及这一节要介绍的卡尔曼滤波。这一节我们就从概率分布的角度来展开说说这个问题。

### Sense

假设我们有一个机器人，想要从A点走到B点，AB之间可以平均分成5个格子，则在没有任何信息输入的情况下，机器人认为自己位于AB之间某一位置的概率为`1/5=0.2`，这个概率称为先验概率。 

<img class="md-img-center" src="{{site.baseurl}}/assets/images/2018/07/ad-uniform-1.png}}">

我们可以用一个Python数组表示每个位置的概率为:

```python
p=[]
n=5 #格子数
for index in range(0,n):
    p.append(1.0/n)

```

而实际上我们的机器人是装有传感器的，他可以感知每个格子中的颜色。由前一篇文章可知，通过传感器，机器人可以引入额外的信息，根据贝叶斯公式，这些信息可以帮助我们矫正先验概率，从而得到更为准确的后验概率。

我们假设AB之间的5个各自为绿色或者红色中的某一种，机器人的传感器可以识别颜色，则引入传感器后，机器人可以更精确的描述自己在AB两点之间的位置。比如当传感器感知到红色的时候，机器人知道自己可能大概率处于红色的格子里，那么在机器人眼中红色格子的概率就应该大于0.2，相应的绿色格子的概率就应该小于0.2。

那么该如何量化这个概率呢？答案是使用贝叶斯公式，我们将红色，绿色格子中的先验概率`0.2`乘以各自的矫正因子，假设红色的矫正因子为`0.6`，绿色为`0.2`，则可以得出下面结果：

<img class="md-img-center" src="{{site.baseurl}}/assets/images/2018/07/ad-uniform-2.png}}">

将上面结果归一化后得到下面结果：

<img class="md-img-center" src="{{site.baseurl}}/assets/images/2018/07/ad-uniform-3.png}}">

这个结果表示<mark>当机器人的传感器感知到红色时</mark>，5个格子的概率分布。其概率含义如下：当机器人看到红颜色时，它位于绿色格子的概率为`1/9`，位于红色格子的概率为`1/3`,即

$$
\begin{cases}\
P(IN_GREEN \| SEEN_RED) = 1/9 \\
P(IN_RED \| SEEN_RED) = 1/3 \\
\end{cases}
$$

这个从先验概率通过贝叶斯公式得到后验概率的过程称为**Sense**，我们可以接种用Python表示上述过程：

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

`sense`函数非常重要，接下来会不断使用这个函数，因此一定要充分理解。有了`sense`函数后，机器人就可以边移动，边采集数据，然后不断更新每个位置上的概率从而确定自己的位置：

```python
#假设机器人移动3格，采集到了下面一组观察序列
measurements = ['green','red','red']
for m in measurements:
    #更新概率分布
    p = sense(p,m)
```

其概率分布变化如下图所示:
<div class="md-flex-h md-flex-no-wrap">
<div><img src="{{site.baseurl}}/assets/images/2018/07/ad-sense-1.png}}"></div>
<div><img src="{{site.baseurl}}/assets/images/2018/07/ad-sense-2.png}}"></div>
</div>

总结一下，机器人通过传感器不断引入观测数据，从而完成了将原先均匀的概率分布提升为包含一定位置的概率分布。sense的本质上是一种<mark>引入信息的过程</mark>

### Move

在定位问题上，除了上面提到的Sense问题以外，我们还需要考虑另一个问题，就是机器人的移动问题。所谓移动问题是指机器人在前进过程中并不能总是能准确的移动到某个位置。比如，我们想让一个机器人向右移动一格，而当机器人移动时，它有一定的概率出错，比如移动了两格或者没移动，因此移动的不准确同样也会造成概率分布的混乱。

我们举一个具体的例子，还是假设机器人从A移动到B，AB之间有5个格子，而此时，机器人已经明确知道自己在第2个格子中，即概率分布为：

```python
p = [0,1,0,0,0]
```

<img class="md-img-center" src="{{site.baseurl}}/assets/images/2018/07/ad-move-0.png}}">

假设机器人现在想要向右移动2个格子，但由于会有一定几率的出错，根据多次试验的结果，假设我们统计出了下面数据：

$$
\begin{cases}\
P(X_{i+2} \| X_i) = 0.8 \\
P(X_{i+i} \| X_i) = 0.1 \\ 
P(X_{i+3} \| X_i) = 0.1 \\
\end{cases}
$$

上面数据的含义是，假设机器人位于`i`的位置，那么它移动到`i+2`的位置的概率为`0.8`，移动1格或3个的概率为`0.1`。此时当机器人前进2格时，对应的概率分布变成了：

<img class="md-img-center" src="{{site.baseurl}}/assets/images/2018/07/ad-move-1.png}}">

```python
p = [0,0,0.1,0.8,0.1]
```
可见机器人的每次移动实际上都是丢失信息的，体现在概率分布从一个位置确定的`1`变为三个位置不确定的值。可以想象一下，假如机器人不听的移动1000次（假设走到第5个格子之后又回到第1个格子），概率分布会变成怎样呢？我们可以来一起计算一下

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
<div><img src="{{site.baseurl}}/assets/images/2018/07/ad-sense-3.png}}"></div>
<div><img src="{{site.baseurl}}/assets/images/2018/07/ad-sense-4.png}}"></div>
</div>

总结一下，对于move来说，<mark>本质上是一个损失信息的过程</mark>，每次机器人移动都会引入一定的不确定性，移动的次数越多，每个位置的不确定性会越大，最终达到均匀分布的状态。

### Sense and Move

了解了sense和move，我们不难发现，机器人移动的过程实际上就是不断获得新信息和不断损失信息的交替循环，机器人没移动一步，会通过传感器得到一个观测值来矫正自己的位置，同时又因为移动带来的不确定性而损失掉一部分信息。这个过程可以如下图所示：

<img class="md-img-center" src="{{site.baseurl}}/assets/images/2018/07/ad-sm-1.png}}">

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
上述代码中，机器人先sense到了`red`，然后向右移动1个格子，接着又sense到了`green`，然后向右又移动了1个格子，此时得到的概率分布如下

<img class="md-img-center" src="{{site.baseurl}}/assets/images/2018/07/ad-sm-2.png}}">

观察上图也可发现，第5个格子的概率最大，说明机器人知道自己目前在第5个格子中。假如我们将`measurements`改为`['red','red']`，概率分布变为下图

<img class="md-img-center" src="{{site.baseurl}}/assets/images/2018/07/ad-sm-3.png}}">

此时机器人知道自己位于第4个格子里，这也符合我们的推测。

### 小结

这一节我们给出了一种确定机器人位置的理论模型和实现方法。我们先来一起回顾下这个过程，首先我们需要有一个位置的先验概率，然后通过`sense`函数来提升先验概率，但由于机器人在移动过程中会带来不确定性，因此会损失一部分信息（确定性），最后机器人通过不断的`sense`和`move`来更新概率密度分布，并找到概率最大的位置作为自己的位置。整个过程如下图所示

<img class="md-img-center" src="{{site.baseurl}}/assets/images/2018/07/ad-kalman-1.png}}">

在实际应用中，`sense`采集到的数据不可能是颜色，而是路面上的一些路标。后面我们还会对localization问题做进一步分析。

## 卡尔曼滤波

在汽车的实际行驶中，它出现的位置是和是和时间相关的函数，因此位置的概率分布是连续的，而不是上面demo中的离散概率分布。但不论离散还是连续，定位的原理是不变的（不断的sense和move）。对于连续型的概率分布，我们可以使用更好的工具来完成密度函数的更迭计算，这个工具就是卡尔曼滤波器。

> 对于卡尔曼滤波器的数学定义和运算推导Youtube上有一套不错的视频，可参考文末的学习资料

### 高斯函数

回顾前面介绍Localization的步骤，第一步是需要有一个先验概率，一个连续型的概率密度函数，对于卡尔曼滤波来说，这个函数可以使用高斯函数来表示，

$$
f(x) = \frac{1}{\sqrt{2\pi\sigma^2}} exp ^{-\frac{(x-\mu)^2}{\sigma^2}}
$$

对于高斯函数来说，方差越大，概率分布函数越“胖”，样本偏离均值的幅度越大；相反的，方差越小，样本偏离均值的幅度越小，概率分布函数越“瘦”。如果用高斯函数来描述机器人的位置，我们希望方差越小越好，这样均值位置就越接近机器人的真实位置。

回到定位问题上，假设我们有一个先验的高斯概率密度函数为$f(x)$，其均值和方差分别为`[120,40]`，校验因子的概率密度函数为$g(x)$，其均值和方差分别为`[200,30]`，如下图所示

<img class="md-img-center" src="{{site.baseurl}}/assets/images/2018/07/ad-gs-1.png}}">


接下来机器人进行了一个次measure，根据贝叶斯公式，后验概率应该等于$f(x)*g(x)$，得到的结果如下图

<img class="md-img-center" src="{{site.baseurl}}/assets/images/2016/07/ad-gs-2.png}}">

从这个结果中，我们能观察出几条重要的结论：

1. 计算后的高斯分布的均值位于`30`和`40`之间
2. 方差变的更小，峰值为三者最高，说明measure获取了新的信息，提高了位置预测准确率

产生这个结果的原因从直观上不太好理解，实际上可以从数学的角度进行证明，当两个高斯函数相乘后，产生的新的均值为

$$
\mu = \frac{\sigma^{-2}_1\mu_1 + \sigma^{-2}_2\mu_2}{\sigma^{-2}_1 + \sigma^{-2}_2}
$$

由于$\mu_2 > \mu_1$，因此有 $\mu_1<\mu<\mu_2$。相应的，产生的新的方差为

$$
\sigma^2 = \frac{sigma^2_1 + sigma^2_2}{sigma^2_1+sigma^2_2}
$$

这个看起来没有那么直观，我们不妨代入几个数试试，令$\sigma^2_1 = \sigma^2_2 = 4$，新的$\sigma$的值为2


{% include _partials/post-footer-2.html %}

## Resources

- [Udacity Intro to Self-Driving Car](https://www.udacity.com/course/intro-to-self-driving-cars--nd113)
- [The Kalman Filter](https://www.youtube.com/playlist?list=PLX2gX-ftPVXU3oUFNATxGXY90AULiqnWT)