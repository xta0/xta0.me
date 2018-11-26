---
layout: post
list_title: 数据科学 | Data Science | Python For Data Analysis | Numpy
title: Python For Data Analysis
categories: [Data Science, Python, Machine Learning]
---

## Python Stack Overview

- Python3 
- Numpy
- Pandas
- Matplotlib
- Scipy

Python系统自带一些数值计算的API，[Numpy](https://www.python-course.eu/numpy.php)是Python数值计算API的封装，Pandas是Numpy的封装。Matplotlib用来做数据可视化

## Numpy

和Python内置的数组实现相比，Numpy数组有下面几点优势

1. compact, they don't take up as much space in memory as a Python list
2. efficient, computations usually run quicker on numpy arrays then Python lists
3. convenient 

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


## Numpy & Pandas for 1D Data

### N-Dimentional Array

Numpy中的array称为`ndarray`,所谓nd是多维的意思，实际上是pyhton数组的封装




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