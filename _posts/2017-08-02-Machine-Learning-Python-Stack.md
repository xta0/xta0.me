---
layout: post
list_title: 机器学习 | Machine Learning | Python工具 | Python Stack For ML
title: Python Stack For ML
categories: [Python, Machine Learning]
mathjax: true
---

## Python ML Stack Overview

- Numpy
- Pandas
- Matplotlib
- Scipy

Python系统自带一些数值计算的API，但它们使用起来效率不高；[Numpy](https://www.python-course.eu/numpy.php)被设计用来专门做数值计算，底层做了大量的优化；Pandas是Numpy的封装；Matplotlib用来做数据可视化

## Numpy

Numpy主要用于做矩阵的数值运算，对常用的数学操作做了封装，使用起来极为方便。在实现方面，Numpy采用了vectorization的实现方式，在性能方面也比Python的list要高不少，因此在数值计算方面Numpy非常流行。

### Problems with Python list

Python原生的多维数组存在很多缺陷，并不适用于数值计算的场景，比如

1. Python数组中的数字是Object类型，有很大的overhead
2. Python的数组对象不支持数值计算的API，比如矩阵相关运算
3. Python的解释器效率很低，和直接运行编译好的函数相比，解释器速度会比较慢

### Vectorization and Broadcasting

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
可以看出numpy并没有使用for循环，这个特性在Numpy中也叫做broadcast，当数组的size很大时，numpy可以极大的节约内存开销和提升开发效率。

### Vectors and Matrices

向量和矩阵是最基本的数值计算单元，我们还是来对比下Python和Numpy的实现

```python
#二维数组
# Python
L = [ [1,2], [3,4] ]
# Numpy
M = np.array([ [1,2], [3,4] ])
type(M) #<class 'numpy.ndarray'>
```
在定义上看，没有太大区别，但是在访问矩阵元素上，Numpy支持以tuple作为脚标进行访问

```python
#Python
a = L[0][0] #矩阵左上角元素
#Numpy
a = M[0,0] 
```

### Generating Matrices to Work With

除了前面提到的使用`np.array`创建矩阵的方法外，Numpy还提供了很多其它的API来创建特殊的矩阵

```python
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
np.random.randn(5) #产生5个随机数，服从一维正态分布
np.random.randn(10,10) #产生10x10个随机数，服从二维正态分布
np.random.randint(1,100,10) #产生10个[1,100)之间的随机数组
```

### Matrix Operations

除了定义矩阵之外，Numpy还内置了很多矩阵运算的API

```python
#转置矩阵
M = np.array([ [1,2], [3,4] ])
transposM = M.T #array([[1, 3], [2, 4]])
#逆矩阵
inverseM = np.linalg.inv(M) #array([[-2. ,  1. ],[ 1.5, -0.5]])
M.dot(inverseM) #should be identity matrix
#矩阵的秩
detM = np.linalg.det(A)
#对角矩阵
diagM = np.diag(A)
#矩阵的外积 
a = np.array([1,2])
b = np.array([3,4])
outerProduct = np.outer(a,b) #array([[3, 4], [6, 8]])
#矩阵的内积
innerProduct = np.inner(a,b) #same as a.dot(b)
#矩阵的对角线求和
np.diag(A).sum() #5
np.trace(M) #5
#矩阵的协方差
X = np.random.randn(100,3)
conv = np.cov(X.T)
#特征值和特征向量
np.linalg.eig(conv)
np.linalg.eigh(conv)
```
## Pandas

Panda主要解决数据处理的问题，假如我们有一份csv文件，我们如何有效的操作和管理这份数据呢？对于Data Scientist来说，最好的选择是将这份数据变成矩阵，因为一旦数据变成了矩阵，数学就可以立刻排上用场了。除了变成矩阵以外，Pandas还提供了一个`DataFrame`的数据结构，可以让我们像操纵数据库一样来操作我们手中的数据。

我们先来对比看一下如何使用Python和Pandas做数据加载，用到的数据为[data_2d.csv](https://github.com/lazyprogrammer/machine_learning_examples/blob/master/linear_regression_class/data_2d.csv)。

```python
import numpy as np 
## Use Python
data = []
for line in open("data_2d.csv"):
    row = line.rstrip().split(',')
    sample = list(map(float, row))
    data.append(sample)
# turns into numpy array
data = np.array(data)
#-------------------------------------
## Use Pandas
import pandas as pd

data = pd.read_csv("data_2d.csv", header=None)
```
### DataFrame

前面提到了Pandas中二维数据都有数据库的表结构来表示，而用来表示表结构的数据结构为`DataFrame`

```python
print(data.info())
#--------------------------
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 100 entries, 0 to 99
Data columns (total 3 columns):
0    100 non-null float64
1    100 non-null float64
2    100 non-null float64
dtypes: float64(3)
memory usage: 2.4 KB
```

上面代码给出了DateFrame的结构，我们可以使用`head()`做一下数据预览

```python
> data.head()
#----------------------------------
           0          1           2
0  17.930201  94.520592  320.259530
1  97.144697  69.593282  404.634472
2  81.775901   5.737648  181.485108
3  55.854342  70.325902  321.773638
4  49.366550  75.114040  322.465486
```
可以看到DataFrame输出的数据格式和上面打印出来的格式一致，且呈现方式为数据库的表结构。
总的来说，DataFrame的设计是为了方便对数据进行数据库操作，比如根据某种条件查询某条记录，等等。我们后面会看到

### Retrieve Column and Row

在Numpy数组中我们可以使用 `data[0]`返回第0行数据，但是在DataFrame中，`data[0]`表示的是返回名字为`0`的column中的所有数据

```python
type(data[0]) #<class 'pandas.core.series.Series'>
```
Pandas中的Series类型是一维数组。我们也可以同时获取多个column的值，此时返回结果将为二维数据，因此类型为DataFrame。

```python
data[[0,2]]
```
有了这个功能，我们便可以做条件查询，例如返回第一列中所有小于5的数据条目，我们可以这么写

```python
data[ data[0]<5 ]
#---------------------------------
           0          1           2
5   3.192702  29.256299   94.618811
44  3.593966  96.252217  293.237183
54  4.593463  46.335932  145.818745
90  1.382983  84.944087  252.905653
99  4.142669  52.254726  168.034401
```

那么如何返回DataFrame中的行呢？可以使用`iloc`或`ix`，返回数据的类型同样是一维的Series

```python
row0 = data.iloc[0]
row0 = data.ix[0]
```

### Add a New Column

如果我们想要增加一个新的Column，简单的方法可以用`data[3]=1.0`。这会另所有的row都增加一个名字为3的1.0的column，显然实际应用中，这种情况很少见，大多数情况是column的值需要从其它column中得来。这时候我们需要使用`apply()`函数

```python
data[3] = data.apply(lambda col: col[0]*col[1], axis=1)
print(data.head())
#-----------------------------------------------
           0          1           2            3
0  17.930201  94.520592  320.259530  1694.773232
1  97.144697  69.593282  404.634472  6760.618304
2  81.775901   5.737648  181.485108   469.201342
3  55.854342  70.325902  321.773638  3928.006993
4  49.366550  75.114040  322.465486  3708.121018
```
此时我们发现DataFrame中增加了一个名为`3`的column，其值为col[0]和col[1]的乘积。`axis=1`表示将计算结果应用在所有行上

### Joins

前面已经提到过，DataFrame的作用是提供一系列类似数据操作的API，便于我们方便处理数据。假设我们有两个csv文件，我们可以将它们当做两个表，各自结构如下

```python
t1 = pd.read_csv("table1.csv")
t2 = pd.read_csv("table2.csv")
print(t1.head())
#-------------------------------------------
   user_id   feature1   feature2    feature3
0        1  17.930201  94.520592  320.259530
1        2  97.144697  69.593282  404.634472
2        3  81.775901   5.737648  181.485108
3        4  55.854342  70.325902  321.773638
4        5  49.366550  75.114040  322.465486
print(t2.head())
#-------------------------------------------
  user_id Country  Age   Salary Purchased
1          France   44  72000.0        No
2           Spain   27  48000.0       Yes
3         Germany   30  54000.0        No
4           Spain   38  61000.0        No
5         Germany   40      NaN       Yes
```
接下来我们可以将两个表按照`user_id`进行JOIN，对应Python代码为

```python
m = pd.merge(t1,t2,on="user_id")
#or: t1.merge(t2,on="user_id")
#----------------------------------------------------------------------------
   user_id   feature1   feature2    feature3  Country  Age   Salary Purchased
0        1  17.930201  94.520592  320.259530   France   44  72000.0        No
1        2  97.144697  69.593282  404.634472    Spain   27  48000.0       Yes
2        3  81.775901   5.737648  181.485108  Germany   30  54000.0        No
3        4  55.854342  70.325902  321.773638    Spain   38  61000.0        No
4        5  49.366550  75.114040  322.465486  Germany   40      NaN       Yes
```

### Convert to Matrix

我们也可以使用`as_matrix()`方法将DataFrame转化成Numpy数组

```python
M = data.as_matrix()
print(type(M)) #<class 'numpy.ndarray'>
```

## Matplotlib

Matplotlib用来做数据可视化，API方便易用。例如我们向画一个Sin函数

```python
import numpy as np 
import matplotlib.pyplot as plt

x = np.linspace(0,10,100) #from 0 - 10, 100 points
y = np.sin(x)
plt.plot(x,y)
plt.xlabel("Time")
plt.ylabel("Some function of Time")
plt.title("My Chart")
plt.show()
```
我们可以用上面的代码绘制任何函数的曲线，观察其形态。实际应用中，更常用的应用场景是我们有一个数据集(`data_1d.csv`)，我们想可视化这份数据，此时我们需要用matplotlib来绘制离散点

```python
A = pd.read_csv('data_1d.csv', header = None).as_matrix()
x = A[:,0] #1st column
y = A[:,1] #2nd column
plt.scatter(x,y)
plt.show() 
```
另一个常用的功能是绘制直方图，使用方式比较简单。下面我们用Numpy随机生成10000个数，观察起分布状态

```python
R = np.random.random(10000)
plt.hist(R)
plt.show()
```



### Resources

- [Numpy](https://www.python-course.eu/numpy.php)