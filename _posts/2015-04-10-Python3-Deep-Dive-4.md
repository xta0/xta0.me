---
title: Iteration and Generators
list_title: Python Deep Dive | Iteration and Generators
layout: post
categories: [Python]
---

### Copying Sequences

由于list是mutable的，Python中有很多方法来copy一个list

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

### Slicing

Slicing只对sequence type有效，他的语法为

```python
my_list[i:j]
```
表示的范围为`[i,j)`，而实际上，slice是一个object

### List Comprehensions

## Generator

## Context Managers