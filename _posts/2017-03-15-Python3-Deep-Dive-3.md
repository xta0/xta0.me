---
title: Python中的函数（一）
list_title: 深入理解Python | Python Deep Dive | 函数（一) | Function Part 1
layout: post
categories: [Python]
---

### 函数参数

- Tuple

在介绍函参数之前，先来回顾一下Python中的Tuple，在Python中，Tuple的定义是被`,`分割的value而不是`()`，`()`的作用是为了让Tuple看起来更容易理解

```shell
Python 3.7.0 (default, Mar 15 2017, 12:20:11)
Type "help", "copyright", "credits" or "license" for more information.
>>> 1,2,3  #declaure a tuple
(1, 2, 3)
```
比如我们想定义一个Tuple，该Tuple只有一个元素`1`，我们可能最先想到的写法是`(1)`，但实际上这并不是Tuple（这是一个`int`值），根据上面的介绍可知，定义Tuple需要使用`,`，因此正确的定义方式是`1,`：

```shell
>>> a = (1)
>>> type(a)
<class 'int'>
>>> b = 1,
>>> type(b)
<class 'tuple'>
>>> b
(1,)
```
如果想要定义一个空的Tuple，可以使用`x=()`，此时编译器理解`()`，或者使用类定义，`x=tuple()`

- Unpacking Values

所谓Packed Values是指一些值以某种方式被pack到一起，最常见的有tuple, list, string, set, 和map这些集合类。对于这些集合类，Python提供了一种展开的方式，即将集合类中的元素以tuple的形式展开

```python
a,b,c = [1,2,3] #a->1, b->2, c->3
a,b,c = 10,20,'hello' #a->10, b->20, c->hello
a,b,c = 10, {1,2}, ['a','b'] #a->10, b->{1,2}, c->['a','b']
a,b,c = 'xyz' #a->x, b->y, c->z
```
上述代码中，等号左边定义了一个tuple，右边是一个集合对象，unpacking的方式是按照位置一一对应。看起来所谓unpacking，实际上就是对集合类对象进行`for`循环为变量依次赋值。但是对于哈希表，`for`循环只得到`key`，因此unpacking的结果也是key，且由于哈希表是无序的，unpacking出来的结果也是不确定的，对于set同理。

```python
d = {'key1':1, 'key2':2, 'key3':3}
a,b,c = d #a->'key2' b->'key3', c='key1'
s = {'x','y','z'}
a,b,c = s #a->'z' b->'x', c='y'
```




在Python中，unpacking对于实现swap功能很方便，只需要unpack一次即可

```python
#swap a,b
a,b = b,a
```
在Python中，上面代码的执行顺序是先进行RHS求值，然后将得到的值再进行LHS赋值给`a,b`


{% include _partials/post-footer-2.html %}