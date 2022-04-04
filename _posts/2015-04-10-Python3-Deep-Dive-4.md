---
title: Generators and Coroutine
list_title: Python Deep Dive | Generators and Coroutine
layout: post
categories: [Python]
---

在介绍Generator之前，我们先来回顾下Python中的iterator和iterable。所谓iterable是指可以被遍历的对象，显然Python中所有可遍历的集合类型都是iterable。
而iterator则是用来获取iterable中元素的接口。比如我们可以用下面方式来获取list中的元素

```python
l = [1, 2, 3]
it = iter(l)
ele = next(it)
```
Python中Iterable和Iterator的实现是分离的，比如list是一个iterable，它的iterator是另一个实对象。这个对象需要实现iterator protcol，它需要实现下面几个方法

- `__iter__()` 这个函数只返回对象本身即`self`
- `__next__` 返回container中的下一个元素，如果元素全被consume完成，则需要raise `StopIteration`的exception

比如我们可以用下面代码来实现一个Python中集合类的iterator

```python
class SeqItor
    def __init__(self, seq):
        self.seq = seq
        self.index = 0
    def __next__(self):
        if self.index >= len(self.seq):
            raise StopIteration()
        else:
            item = self.seq[index]
            index += 1
            return item
    def __iter__(self):
        return self
```
此时，当`iter(l)`执行时，Python首先会找`__iter__`是否存在，如果存在，则直接使用。否则会继续查找`__getitem__()`方法，如果存在，则会创建一个iterator，否则抛异常。

而Iterable同样需要实现一组protocol，它只包含一个函数 `__iter__()` 这个函数返回一个新的iterator对象，我们这里不继续展开了。

## Generator

Generator用来解决下面的场景

```python
def func(arg)
    # 1. do something with arg
    # 2. pause the execution
    # 3. resume the function
    return sth
```
上面三个步骤可以用通过`yield`的关键字来完成，它的作用如下

1. 让`func`产生一个值
2. 暂停`func`的执行，所有call stack上的状态会被保留
3. 使用`next(func)`可以让`func`继续执行
4. `func`执行完后返回`sth`，此时`next`会抛`StopIteration`的异常

我们来看一个具体的例子

```python
def func():
    print('1')
    yield "1st yield"
    print('2')
    yield "2nd yield"

x = func() # --- (1)
```
当执行到`(1)`时，Python会检查


## Context Managers