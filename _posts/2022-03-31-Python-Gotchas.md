---
list_title: Python | Gotchas | 常见的mistakes
title: Python Gotchas
layout: post
categories: ["Python"]
---

### Lists vs Tuples

我们可以先对比一下List和Tuple的bytecode

```shell
# compile tuple
dis(compile('(1, 2, 3, "a")', 'string', 'eval'))

0 LOAD_CONST      0 ((1, 2, 3, 'a'))
2 RETURN_VALUE

# compile list
dis(compile('[1, 2, 3, "a"]', 'string', 'eval'))

0 LOAD_CONST               0 (1)
2 LOAD_CONST               1 (2)
4 LOAD_CONST               2 (3)
6 LOAD_CONST               3 ('a')
8 BUILD_LIST               4
10 RETURN_VALUE
```

上面例子中，无论是tuple还是list，他们包含的是immutbale的data，我们可以发现，bytecode中tuple是一个constant，可以直接load，
而list需要通过更多的指令去动态创建。这是因为tuple本身是Immutable的数据结构。但是，如果tuple中包含mutable的数据结构，则tuple的创建将变的和list一样

```python
dis(compile('([1, 2], 3, "a")', 'string', 'eval'))

0 LOAD_CONST               0 (1)
2 LOAD_CONST               1 (2)
4 BUILD_LIST               2
6 LOAD_CONST               2 (3)
8 LOAD_CONST               3 ('a')
10 BUILD_TUPLE             3
12 RETURN_VALUE
```
此外，由于list的动态性，它需要preallocate space，因此它比tuple在内存上有更大的overhead

```python
import sys
s1 = sys.getsizeof((1,)) #48
s2 = sys.getsizeof([1,]) #64
```

### Shallow Copy vs Deep Copy

Python中有很多方法来copy一个list

```python
s = [1,2,3]

#1. for loop
cp = [e for e in s]

#2. copy()
cp = s.copy()

#3. slicing
cp = s[:]
cp = s[0:len(s)]

#4. list()
cp = list(s)
```
需要注意的是这里copy全是shallow copy，即`cp`是一个新的list，但是里面的元素还是指向原来list中的元素，即

```python
id(s) != id(cp)
```
对于immutable的sequence，如tuple，copy则不会创建新的对象

```python
t1 = (1, 2, 3)
t2 = tuple(t1)
id(t1) == id(t2)
```
Shallow copy的一个问题是，如果mutate原来的list中存在mutable的object，则同样会一向copy后list中的元素

```python
s1 = [[10, 20], 3, 4]
s2 = s1.copy()
s1[0][0] = 100 # s1 = [[100, 20], 3, 4]
# s2[0][0] will also be mutated
```
此时我们需要使用deep copy，但是实现deep copy是一件不容易的事情，比如一个list中有sublist，sublist中又有sublist，因此deep copy需要考虑递归的情况。对于标准库存中的sequence，Python提供`.deepcopy()`的API来实现deep copy

```python
import copy
x = [1, 2, 3]
y = copy.deepcopy(x)
```

### Mutable objects

如果concat两个包含mutable object的list，则结果

```python
x = [[1, 2]]
y = x + x
# y -> [[1, 2],[1, 2]] 
x[0] = [-1, -1]
# y -> [[-1, -1], [-1, -1]] 
```
实际上 `id(y[0]) == id (y0[1]) == id(x)`
