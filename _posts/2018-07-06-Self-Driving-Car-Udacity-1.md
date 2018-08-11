---
updated: "2018-07-10"
layout: post
list_title: 自动驾驶入门（一）| Udacity Autonomous Driving Part 1 | Bayesian Thinking
title: 贝叶斯概率模型
categories: [AI,Autonomous-Driving]
mathjax: true
---

> Course Notes from Udacity Self-Driving Car Nanodegree Program

### Prerequisite

- [Independent Probability](https://en.wikipedia.org/wiki/Independence_(probability_theory)) 
- [Conditional Probability](https://zh.wikipedia.org/wiki/%E6%9D%A1%E4%BB%B6%E6%A6%82%E7%8E%87)
- [Total Probability](https://zh.wikipedia.org/wiki/%E5%85%A8%E6%A6%82%E7%8E%87%E5%85%AC%E5%BC%8F)



### 条件概率

要理解贝叶斯定理，先要复习一下什么是条件概率。所谓条件概率是指在某件事发生的前提下，另一件事发生的概率。和独立概率比起来条件概率不是很好理解，因为平时生活中能直接使用上这种概率的机会比较少，对于条件概率，一个经典的实验是红蓝球实验

但是用文氏图却可以很直观的表达，例如我们想要表达在B发生的条件下A发生的概率，用文氏图可以表示如下

<img class="md-img-center" src="{{site.baseurl}}/assets/images/2018/07/ad-1.jpg">

根据文氏图，可以很清楚地看到在事件B发生的情况下，事件A发生的概率就是P(A∩B)除以P(B)。即

$$
P(A|B) \thinspace = \thinspace \frac{P(A∩B)}{P(B)} 
$$

这个就是条件概率公式，其中AB交集部分的概率 $P(A∩B)$ 不好解释，可通过代数运算将其化简掉

$$
P(A∩B) \thinspace = \thinspace P(A|B)P(B)
\\P(A∩B) \thinspace = \thinspace P(B|A)P(A)
$$

得到：

$$
\\P(A|B) \thinspace = \thinspace \frac{P(B|A)P(A)}{P(B)} 
$$





## 附录

- Simulating Probability 

```python


```

- Solutions Joy Ride Project

```python
car_parameters = {"throttle": 0, "steer": 0, "brake": 0}

def control(pos_x, pos_y, time, velocity):
    """ Controls the simulated car"""
    global car_parameters
    if(time < 3):
        car_parameters["throttle"] = -1.0
        car_parameters["steer"] = 1
        car_parameters["brake"] = 0
    elif(pos_y > 32):
        car_parameters["throttle"] = -1.0
        car_parameters["steer"] = -1
        car_parameters["brake"] = 0
    else:
        car_parameters["throttle"] = 0
        car_parameters["steer"] = 0
        car_parameters["brake"] = 1
    
    return car_parameters
    
import src.simulate as sim
sim.run(control)
```
- [Code simulation on Youtube](https://www.youtube.com/watch?v=pYvCvNFZFMw)