---
layout: post
list_title: 数据科学 | Data Science | Python For Data Analysis
title: Python For Data Analysis
categories: [Data Science, Python, Machine Learning]
---

## 概述

这篇文章会重点介绍Python数据分析工具的关系以及一些核心的知识点，这篇文章不会cover一些具体的API用法，这些内容可以参考各自网站的API手册

## Python Stack Overview

- Python3 
- Numpy
- Pandas
- Matplotlib
- Scipy

Python系统自带一些数值计算的API，[Numpy](https://www.python-course.eu/numpy.php)是Python数值计算API的封装，Pandas是Numpy的封装

### Python3

二维数组



### Numpy & Pandas for 1D Data


### Numpy & Pandas for 2D Data


- ndarray

```
```

- **Matrix**

对于数值计算，操作单元是矩阵，一维数组也可以看成是矩阵，

```python
#一维数组
x = np.array([1,2,3])
#二位数组
x = np.array([[1,2,3],[4,5,6],[7,8,9]])
type(x) #<class 'numpy.ndarray'>
#range
x = np.arrange(0,5)
x = np.arrange(1,11,2) #[1,3,5,7,9],step size: 2
#zeors
np.zeros(4) #一维zero g数组
np.zeros((5,5)) #5x5的zero数组
#ones
np.ones(4)
np.ones((3,5))
#linspace
np.linspace(0,10,30) #start, end , 点的个数 => 产生等距的点的数组
#单位阵
np.eye(4) #4x4的单位阵
#Random number
np.random.rand(1) #按照等概率分布产生一个[0-1)之间的随机数
np.random.rand(3,3) #产生一个[0-1)之间的3x3矩阵
np.random.randn(5) #按照正态分布产生5个(-1,1)之间的随机数，均值为0，方差为1
np.random.randint(1,100,10) #产生10个【1,100)之间的随机数组
```


### Resources

- [Numpy](https://www.python-course.eu/numpy.php)