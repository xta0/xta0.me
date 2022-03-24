---
title: Scopes, Closures and Decorators
list_title: Python Deep Dive | Scopes, Closures and Decorators
layout: post
categories: [Python]
---

### Scope

首先，Python中有一个built-in Scope，它包含了最基本的的数据结构，定义在这个namespace中的类型对所有的module可见。而所谓的**Global Scope**都是相对于module的，它的作用域局限于单个文件。假设我们在`module1.py`中调用了

```python
print(True)
```
Python首先会在当前module的namespace中寻找`print`和`True`，如果没找到，则会向上寻找built-in namespace中是否有该符号。

> built-in namespace中的符号可以被module中的符号override

函数中创建的variable被保存到所谓的**local scope**中。只有当函数被调用时，local scope才会被创建出来。举个例子

```python
def my_func(a, b):
    c = a * b
    return c
```
在编译时，编译器会将`a,b`和`c`定义在local scope中，但是只有在函数被调用时(runtime)，这个local scope才会被创建出来，当函数调用结束后又会将scope释放。具体来说，编译器规则如下


```python
a = 10

def func1():
    print(a) # at compile time, a is non-local

def func2():
    a = 100 # at compile time, a is local

def func3():
    global a
    a = 100 # at compile time, a is global

def func4():
    print(a) # runtime error
    a = 100 # at compile time, a is local
```

Python同样可以定义nested function，这就会带来nested local scope的问题，比如下面代码

```python
def outer():
    x = 'hello'
    def inner():
        nonlocal x
        x = 'monty'
```
此时如果`inner()`想要override `outer`中的`x`，则需要使用`nonlocal` keyword

## Closure

Closure允许inner function来capture函数外的变量(nonlocal)。注意，closure不等于lambda，它更像一个scope，如下面例子中被虚线圈出的部分称为一个closure

```python
def outer():
   |-----------------------------------|
   |x = 'hello'                        |
   |def inner():                       |
   |    print( "{0} rocks!".format(x)) |
   |-----------------------------------|
   return inner

fn = outer()
fn()
```
上面例子中，当`fn`被执行时，closure会被创建出来，我们虽然返回了一个`inner`函数，但实际上我们返回的是一个closure。

这里`inner`和`outer`都引用了同一个`x`，此时`x`具有两个scope属性，一个是`outer`的local scope，一个是`inner`的nonlocal scope。而`x`在内存中只有一份，理论上当`fn()`执行时，`outer()`已经执行完成了，`x`应该被释放调了，那么closure中的`x`指向哪里呢？

实际上，Python会为处于不同scope的变量创建一个intermediate对象叫做**cell**，上面例子中，不论是`outer`还是`inner`所引用的`x`实际上是一个cell对象

```shell
outer.x  -----> |-------------|    |------------|
                | cell 0xA500 |    | str 0xFF100|
                | 0xFF100   --|--->|     python |
inner.x  -----> |-------------|    |------------|
```

即使`outer()`执行完，`outer.x`消失，对于cell来说相当于引用计数减1，而由于`inner.x`还在，因此cell不会被释放，它仍然可以引用到其指向的string对象。

因此，我们可以将Closure理解成function + extended scope。extended scope中会包含一些所谓的free variables，即intermediate cell objects。

我们可以用下面代码查看closure中包含哪些free variable

```python
fn.__code__.co_freevars # （‘x’）
fn.__closure__ # (<cell at 0xA500: str object at 0xFF100>)
```

closure是在运行时创建的，包括free variable 也是运行时创建的，因此两个closure对象中的cell object是不同的。

```python
def outer():
    cnt = 0
    def inc():
        nonlocal cnt
        cnt += 1
        return cnt
    return inc
f1 = outer()
f2 = outer()
f1() # 0
f1() # 1
f2() # 0
```
上面`f1`和`f2`是两个独立的closure，他们指向的`cnt` cell object也是不同的。因此`f1()`并不会改变`f2`中`cnt`的值。如果我们像让两个closure share `cnt`，则可以在同一个local scope中定义两个closure

```python

def outer():
    cnt = 0
    def inc1():
        nonlocal cnt
        cnt += 1
        return cnt
    return inc

     def inc2():
        nonlocal cnt
        cnt += 1
        return cnt
    return inc

    return inc1, inc2
```
此时，`inc1`和`inc2`指向同一个cell对象，因此share同一个`cnt`。
