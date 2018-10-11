---
updated: "2018-07-10"
layout: post
list_title: 自动驾驶入门 | Autonomous Driving | 计算位置 | Localizations
title: 卡尔曼滤波器
categories: [AI,Autonomous-Driving]
mathjax: true
---


### Prerequisite

- [Beyesian Model](https://xta0.me/2018/07/06/Self-Driving-Car-Udacity-1.html) 
- [高斯分布]()
- [Python Numpy](https://xta0.me/2017/05/10/Data-Science-Tools.html)


本将继续讨论自动驾驶中用到的一些基本统计学知识，包括概率分布模型，卡尔曼滤波器以及简单的运动模型。本文假设你已经掌握了了一些简单的概率统计知识，并且知道如何使用Python和Numpy进行简单的编程。

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

## Localization

接下来我们讨论一个有趣的问题，即机器人如何知道自己在哪？这个问题在上一节最后曾简单的提到的过，通常的做法是先使用多个GPS进行三角定位，然后再结合自身各种传感器采集的数据进行位置矫正。矫正的过程会使用到贝叶斯定理以及这一节要介绍的卡尔曼滤波。





## 运动模型

- 匀速直线运动
- 匀变速直线运动

$$
v(t) = v_0 + at
$$

距离

$$
s(t) = v_0t + 0.5*at^2
$$