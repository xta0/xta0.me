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

closure是在运行时创建的，但并不会evaluate，只有当closure被调用时，才会创建所用到的free variable，因此两个closure对象中的cell object是在各自extended scope中的。

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
上面`f1`和`f2`是两个独立的closure，他们指向的`cnt` cell object也是不同的。因此`f1()`并不会改变`f2`中`cnt`的值。如果我们像让两个closure share `cnt`，则可以让两个closure share同一个extended scope

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

f1, f2 = outer()
f1() # 1
f2() # 2
```

此时，`inc1`和`inc2`指向同一个cell对象，因此share同一个`cnt`。

Closure在Python中有很多应用，在介绍decorator前，我们先来看一个例子，假设我们要记录一个函数被调用了多少次，我们可以写这样一个closure

```python
def counter(fn):
    cnt = 0
    def inner(*args, **kwargs):
        nonlocal cnt
        cnt += 1
        print("{0} has been called {1} times.".format(fn.__name__, cnt))
        return fn(args, kwargs)
    return inner

def add(a, b):
    return a + b

add = counter(add)
add.__closure__
# <cell at 0x001234: int object at 0x5678>
# <cell at 0x00abcd: function object at 0xff33>
add(10, 20)
```
这里`add`是一个closure，它包含两个free var，一个是`fn`，一个是`cnt`。

不难发现，我们用`count`封装了一个closure，这个closure除了会调用我们想要调用的函数之外，还可以做一些其它事情，此时`counter`就是所谓的decorator。当我们再次调用`add`时，我们调用的实际上是一个closure，是一个“加强版”的`add`。

由于此时的`add`已经不再是我们定义的`add`，而我们还是希望在debug时，`add`能返回正确的信息，因此，我们加上下面代码

```python
def counter(fn):
    cnt = 0
    def inner(*args, **kwargs):
        nonlocal cnt
        cnt += 1
        print("{0} has been called {1} times.".format(fn.__name__, cnt))
        return fn(args, kwargs)
    inner.__name__ = fn.__name__
    inner.__doc__ = fn.__doc__
    return inner
```

Python中提供了更简洁的方法来保持原函数的metadata

```python
from functools import wraps

def counter(fn):
    cnt = 0
    def inner(*args, **kwargs):
        nonlocal cnt
        cnt += 1
        print(count)
        return fn(args, kwargs)
    inner = wraps(fn)(inner)
    return inner

```

## Decorator

一个Decorator主要有以下几个特点

1. Decorator接受一个函数作为参数
2. 返回一个closure
3. closure通常接受任意可变参数
4. inner函数通常会执行一些其它代码
5. 调用传入的函数
6. 返回传入函数的调用结果

### The `@` symbol

上面提到如果我们想用`counter`来decorate `add`，即`add = couter(add)`，Python提供了一个方便的syntax sugar - `@`

```python
@counter
def add(a, b):
    return a + b

# same as writing
def add(a, b):
    return a + b
add = counter(add)
```
decorator可以被stack，一个函数可以被多个decorator修饰

```python
def dec1(fn):
    def inner():
        print("dec1 is called")
        return fn()
    return inner
def dec2(fn):
    def inner():
        print("dec2 is called")
        return fn()
    return inner()

@dec1
@dec2
def my_func():
    print("my func is called")
```

此时需要注意decorator的执行顺序，此时会先执行`dec2`再执行`dec1`，最后执行`my_func`。

项目中我们可以用decorator做很多操作，比如logging，cache以及timing，

### Decorator Prameters

Python的标准库中提供了一些很好用的decorator，比如 `@wrap(fn)`,`@lru_cache(maxsize=256)`，这些decorator支持参数传递，这是如何做到的呢？

我们还是先来看一个例子，假设我们有下面函数

```python
def timed(fn, reps):
    from time import perf_counter
    from functools import wraps

    @wraps(fn)
    def inner(*args, **kwargs):
        total_elapsed = 0
        for i in range(reps):
            start = perf_counter()
            result = fn(*args, *kwargs)
            end = perf_counter()
            total_elapsed += (end - start)
        avg_eplapsed = total_elapsed / reps
        print(avg_eplapsed)
        return result
    return inner
```
我们为closure引入了一个新的free var - `reps`，用来控制循环次数，`timed`包含一个参数。一个比较直观的想法是像下面这样，给decorator加一个参数

```python
@timed(10)
def my_func():
    pass
```
但实际上，上面代码将无法编译。如果想让上面代码工作，则`@timed(10)`需要返回一个`@time`的decorator，因为我们知道`@timed`是可以正常工作的

```python
dec = timed(10) #returns a decorator
@dec
def my_func()
    pass
```
显然，`timed(10)`是一个函数，他需要返回一个decorator，也叫做decorator factory，于是我们可以试着定义下面的函数

```python
def timed(reps):
    def dec(fn):
        from time import perf_counter
        from functools import wraps
        @wraps(fn)
        def inner(*args, **kwargs):
            total_elapsed = 0
            for i in range(reps):
                start = perf_counter()
                result = fn(*args, *kwargs)
                end = perf_counter()
                total_elapsed += (end - start)
            avg_eplapsed = total_elapsed / reps
            print(avg_eplapsed)
            return result
        return inner
    return dec
```
现在如果调用`timed(10)`将返回一个decorator/closure，然后再通过它来调用`my_func`就能实现上面的效果

```python
my_func = timed(10)(my_func)

# equals to
@timed(10)
def my_func():
    pass
```

### Decorator Class

Python中的class也可以实现类似于C++的`operator()`函数，在Python中叫做`__call__`，比如下面代码

```python
class MyClass:
    def __init__(self, a, b):
        self.a = a
        self.b = b
    def __call__(self):
        print("called a ={0}, b={1}".format(self.a, self.b))

obj = MyClass(10,20)
obj.__call__()
obj()
```
实际上我们可以将`__call__`变成一个decorator

```python
def __call__(self, fn):
    def inner(*args, **kwargs):
        print("called a ={0}, b={1}".format(self.a, self.b))
        return fn(*args, **kwargs)
    return inner
```
此时我们可以用`@MyClass(a,b)`来decorate函数，此时`MyClass(a, b)`返回一个Callable，例如

```python
@MyClass(10, 20)
def my_func(s):
    print("Hello {0}".format(s))

my_func('world')
```
上面代码等价于

```python
obj = MyClass(10, 20)
my_func = obj(my_func)
my_func('world')
```










