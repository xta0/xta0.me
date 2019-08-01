---
layout: post
list_title: 数据科学 | Data Science | Python Stack For Data Analysis
title: Python Stack For Data Analysis
categories: [Data Science, Python, Machine Learning]
mathjax: true
---

## Python Stack Overview

- Numpy
- Pandas
- Matplotlib
- Scipy

Python系统自带一些数值计算的API，但它们使用起来效率不高；[Numpy](https://www.python-course.eu/numpy.php)被设计用来专门做数值计算，底层做了大量的优化；Pandas是Numpy的封装；Matplotlib用来做数据可视化

## Numpy

Numpy主要用于做矩阵的数值运算，对常用的数学操作做了封装，使用起来极为方便，而且在性能方面也比Python的list要高不少，因此在数值计算方面Numpy非常流行。

我们可以来对比一下numpy和python的矩阵点积运算，假设有两个矩阵$a=[x_a,y_a]$, $b=[x_b,y_b]$，则`a`和`b`点积的代数运算为 $a·b=x_ax_b+y_ay_b$，其结果为一个标量。代码如下

```python
import numpy as np 
from datetime import datetime

a = np.random.randn(100)
b = np.random.randn(100)
T = 100000

## Python code
## use a for loop
def slow_dot_product(a,b ):
    result = 0
    for e,f in zip(a,b):
        result += e*f
    return result

dt0 = datetime.now()
for t in range(T): 
    slow_dot_product(a,b)
dt1 = datetime.now() - dt0

## numpy version 
dt0 = datetime.now()
for t in range(T):
    #numpy dot product
    a.dot(b)
dt2 = datetime.now() - dt0

print("dt1/dt2: ", dt1.total_seconds() / dt2.total_seconds()) #~40
```

Numpy实际上是使用vectorization取代for循环来做运算，这也是为什么numpy会特别快的原因。另一个能提高效率的场景是对每个矩阵的元素做运算：

```python
m = np.ones([1,2]) #[1,1]
m = m*2 #[2,2]
```
可以看出numpy并没有使用for循环，当数组的size很大时，numpy可以极大的节约内存开销和提升开发效率。

- **Matrix**

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