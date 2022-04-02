---
title: Iteration and Generators
list_title: Python Deep Dive | Iteration and Generators
layout: post
categories: [Python]
---

### Slicing

Slicing只对sequence type有效，他的语法为

```python
my_list[i:j]
```
表示的范围为`[i,j)`，而实际上，slice是一个Python的object

```python
l = [1, 2, 3, 4, 5]
s = slice(0, 2) #s.start=0, s.end=2 
type(s) #<class 'slice'>
x = l[s] #[1, 2]
```
slice是独立于sequence的，它只需要定义start和end，而不需要知道sequence的信息，比如我们可以合法使用下面的slice

```python
l = ['a','b','c','d','e','f','g']
x = l[3:100] # ['d', 'e', 'f']
```
slice没有所谓的数组越界的概念，他只是指定了一个range，当被作用于sequence时，Python会根据一些规则计算`start`和`end`，并得出一个range。 

假设有数组

```python
l = ['a','b','c','d','e','f','g']
```
`seq[i: j]`的计算规则如下

- `if i>len(seq), i = len(seq)`
    - `[100, 1000] -> range(6, 6)`
- `if j>len(seq), j = len(seq)`
    - `[100, 1000] -> range(6, 6)`
- `if i<0, i = max(0, len(seq)+i)`
    - `[-5, 3] -> range(1, 3)`
- `if j<0, i = max(0, len(seq)+j)`
    - `[-10, 3] -> range(0, 3)`
- `if i is None, i=0`
    - `[:100] range(0, 6)`
- `if j is None, j=len(seq)`
    - `[3:] -> range(3, 6)`



### List Comprehensions

## Generator

## Context Managers