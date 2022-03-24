---
title: Scopes, Closures and Decorators
list_title: Python Deep Dive | Scopes, Closures and Decorators
layout: post
categories: [Python]
---

### Scope

首先，Python中有一个built-in Scope，它包含了最基本的的数据结构，定义在这个namespace中的类型对所有的module可见。而所谓的**Global Scope**都是相对于module的，它的作用域局限于单个文件。如下图所示

<img class="md-img-center" src="{{site.baseurl}}/assets/images/2015/04/pl-05-01.png">

假设我们在`module1.py`中调用了

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

Closure允许inner function来capture函数外的变量(nonlocal)，

