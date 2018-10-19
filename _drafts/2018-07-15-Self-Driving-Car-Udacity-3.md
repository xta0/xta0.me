---
updated: "2018-08-12"
layout: post
list_title: 自动驾驶入门 | Autonomous Driving | 卡尔曼滤波器(二) | Kalman Filter Part 2
title: 卡尔曼滤波器(二)
categories: [AI,Autonomous-Driving]
mathjax: true
---

<img class="md-img-center" src="{{site.baseurl}}/assets/images/2018/07/ad-location.png">

> Course Notes from Udacity Self-Driving Car Nanodegree Program

### Prerequisite

- [Kalman Filter Part 1]()


## 多维卡尔曼滤波

为了真正模拟机器人或者汽车在现实中的运动，我们需要使用多维的卡尔曼滤波器。现在我们可以来正式的介绍一下什么是卡尔曼滤波了，首先卡尔曼滤波主要有两方面的作用：

1. 状态估计，对于某些无法直接测量的状态，可以使用卡尔曼滤波器进行估计
2. 对各种输入的信息做最优的状态估计