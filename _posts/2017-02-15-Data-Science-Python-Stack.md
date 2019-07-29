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

我们可以来对比一下numpy数组和python做矩阵的点积

```python
import numpy as np 
from datetime import datetime

a = np.random.randn(100)
b = np.random.randn(100)
T = 100000

def slow_dot_product(a,b ):
    result = 0
    for e,f in zip(a,b):
        result += e*f
    return result

dt0 = datetime.now()
for t in range(T): 
    slow_dot_product(a,b)
dt1 = datetime.now() - dt0

dt0 = datetime.now()
for t in range(T):
    #numpy dot product
    a.dot(b)
dt2 = datetime.now() - dt0

print("dt1/dt2: ", dt1.total_seconds() / dt2.total_seconds()) #~40
```

可以看出Numpy会对数组中每个元素进行数学运算，当数组的size很大时，Numpy可以极大的节约内存开销和提升开发效率，除此之外，Numpy还提供了一系列的方便的数学运算

### Dot Product

数学中，我们经常需要计算两个向量的点积(dot product)。假设有两个二维想来那个$a=[x_a,y_a]$和$b=[x_b,y_b]$，则`a`和`b`点积的代数运算为 $a·b=x_ax_b+y_ay_b$，其结果为一个标量。使用numpy可以表示为

```python
import numpy as np

a = np.array([1,2])
b = np.array([2,1])
dot = np.dot(a,b) #4
dot = a.dot(b) #4
dot = b.dot(a) #4
```
如果我们将点积转化为其几何表达

### Vector and Matrices

```python
import numpy as np

# A simple robot world can be defined by a 2D array
# Here is a 6x5 (num_rows x num_cols) world
world = np.array([ [0, 0, 0, 1, 0],
                   [0, 0, 0, 1, 0],
                   [0, 1, 1, 0, 0],
                   [0, 0, 0, 0, 1],
                   [1, 0, 0, 1, 0],
                   [1, 0, 0, 0, 0] ])

# Visualize the world
print(type(world)) #ndarray
print(world.shape) #<row,column> (6,5)
print('height' + str(world.shape[0]));  #6
print('width' + str(world.shape[1]));  #5
```
上面我们使用`numpy`创建了一个2维数组，在`numpy`中，N维矩阵可以通过`np.array`构建，类型为`ndarray`。`ndarray`有一个`shape`属性，类型为`tuple`，用来保存矩阵的维度。

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

总结一下，和Python内置的数组实现相比，Numpy有下面几点优势

1. 语法简单，易上手
2. 数值运算的效率比Python list要高(没有for循环)
3. 消耗的内存更少



### Resources

- [Numpy](https://www.python-course.eu/numpy.php)