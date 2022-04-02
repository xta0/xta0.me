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
slice没有所谓的数组越界，他只是指定了一个range

### List Comprehensions

## Generator

## Context Managers