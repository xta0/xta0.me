---
title: 理解变量
list_title: 深入理解Python（二）| 变量 | Variables 
layout: post
categories: [Python]
---

### 类型系统

Python是一门动态类型语言，变量的类型依赖解释器的类型推导，可以通过关键字`type`来查看变量类型

```python
a=10
type(a) #<class 'int'>
```

需要注意的是，`int`并不是`a`的类型，而是其所指向内存数据的类型，`a`本身是没有类型的，或者说它的类型是动态的，它同样可以指向其它类型的内存数据

```python
a = "hello"
type(a) #<class 'str'>
```
此时`a`的"类型"变成了string，上述代码实际上是对`a`的reassignment，Python是怎么实现对变量的重新赋值呢？一个直观的想法是`a`所指向的内存数据被修改了，从`10`被修改为`hello`，也就是所谓的mutation。实际上则不是，`a`指向的是另一块新的内存，其值为`hello`，原来`10`的那块内存仍然存在，只是没有人引用它的，很快便会被GC回收。

```python
print(hex(id(a))) #x10c28eb90
a = 15
print(hex(id(a))) #0x10851fc30
```

上述代码对`a`进行了重新赋值，可见其指向的内存地址发生了变化，而并非是修改原来内存地址中的值。Python中另一个比较有趣的现象是如果两个变量的值相同，那么它们都是指向同一块内存的区域的引用

```python
a = 10 
b = 10
print(hex(id(a))) #x10c28eb90
print(hex(id(b))) #x10c28eb90
```
这点在其它的变成语言中似乎并不常见

<p class="md-h-center">(未完待续)</p>

