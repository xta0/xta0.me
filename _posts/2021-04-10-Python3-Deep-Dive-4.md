---
title: Generators and Coroutine
list_title: Python Deep Dive | Generators and Coroutine
layout: post
categories: [Python]
---

在介绍Generator之前，我们先来回顾下Python中的iterator和iterable。所谓iterable是指可以被遍历的对象，显然Python中的集合类型都是iterable。
而iterator则是用来获取iterable中元素的接口。比如我们可以用下面方式来获取list中的元素

```python
l = [1, 2, 3]
it = iter(l)
ele = next(it)
```
Python中Iterable和Iterator的实现是分离的，比如list是一个iterable，它的iterator是另一个对象。这个对象需要实现iterator protcol，它需要实现下面几个方法

- `__iter__()` 这个函数只返回对象本身即`self`
- `__next__` 返回container中的下一个元素，如果元素全被consume完成，则需要raise `StopIteration`的exception

比如我们可以用下面代码来实现一个Python中集合类的iterator

```python
class SeqItor:
    def __init__(self, seq):
        self.seq = seq
        self.index = 0
    def __next__(self):
        if self.index >= len(self.seq):
            raise StopIteration()
        else:
            item = self.seq[self.index]
            self.index += 1
            return item
    def __iter__(self):
        return self
```
此时，当`iter(obj)`执行时，Python首先会找`__iter__`是否存在，如果存在，则直接使用。否则会继续查找`__getitem__()`方法，如果存在，则会创建一个iterator，否则抛异常。

而Iterable同样需要实现一组protocol，它只包含一个函数 `__iter__()` 这个函数返回一个新的iterator对象

```python
class MyList:
    def __init__(self, seq):
        self.seq = seq
    def __iter__(self):
        return SeqItor(self.seq)
```
上面例子中我们的`MyList`由于定义了`__iter__`，因此它是一个iterable类，他可以被遍历

```python
x = MyList([1, 2, 3])
for n in x:
    print(n)

# use iter and next
it = iter(x)
n = next(it)
print(n)
```

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

x = func() # --- (1) x is a generator 
y1 = next(x) # --- (2) 
y2 = next(x) # --- (3)
y3 = next(x) # --- (4)
```
1. 当执行到`(1)`时，Python发现`func()`中存在`yied`而不会求值，此时函数处于pause状态。实际上，返回值`x`是一个generator对象
2. 当执行到`(2)`时，根据上面的步骤`yield`会先产生一个值，其内容为`yield`后面的部分。然后让`func`继续执行，因此，`y1`的值为`1st yied`。同时执行`print`操作输出`1`
3. 第`(3)`步和前一步同理
4. 当执行到`(4)`时，`func`返回，此时会抛一个`StopIteration`异常

在Python中，如果一个函数包含了`yied`关键字，则这个函数为Generator Factory，它用来生成Generator对象。而Generator实现了itorator的protocol，因此我们可以用`next()`来通过Generator取值。

### Generator Expression

## Coroutine

关于Coroutine的概念，我们曾在之前[Lua的文章中](https://xta0.me/2014/02/04/Lua-2.html)介绍过，可以先回顾下那篇文章。在Python中有两种方法可以创建coroutine

1. 通过 `yield`
2. 通过 `async / await`

这里不会介绍`asyncio`和`async/wait`，而是会探讨`coroutine`实现的基本原理





