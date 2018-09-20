---
updated: "2018-07-10"
layout: post
list_title: 自动驾驶入门 | Autonomous Driving | 概率分布模型 | Probability Distributions
title: 概率分布模型
categories: [AI,Autonomous-Driving]
mathjax: true
---

### Prerequisite

- [Beyesian Model](https://xta0.me/2018/07/06/Self-Driving-Car-Udacity-1.html) 
- [Python Numpy](https://xta0.me/2017/05/10/Data-Science-Tools.html)

### 概率分布模型

概率分布模型是概率的一种数学描述，和任何一种数学公式一样，概率模型可以被二维可视化，也可以用于线性代数和微积分的计算。概率统计模型有两种，一种是离散概率，一种是连续概率。其分布曲线如下图所示:

{% include _partials/components/lightbox.html param='/assets/images/2018/07/ad-6.jpg' param2='1' %}

从上面的图中可以看出，X轴表示我们感兴趣的事情；Y轴对于离散概率表示某事情发生的概率，对于连续概率，表示概率密度函数。