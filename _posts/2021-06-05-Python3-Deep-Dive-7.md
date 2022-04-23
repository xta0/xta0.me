---
title: Classes
list_title: Python Deep Dive | Classes
layout: post
categories: [Python]
---

Python中的objects都有自己的类型，`class`也不例外，在Python中，class是first-class citizen。即`class`也是一个object，它有自己的类型，也就是说`class`有所谓的metaclass，具有meta programing的能力。

```python
class MyClass:
    pass
```
上面代码中，python会创建一个`MyClass`的object，它的类型是`type`，因此我们可以用它来定义其他对象。既然`MyClass`是个object，则它有自己的method，比如

```python
MyClass.__name__ # 'MyClass'
```

### Attributes

每个`class` object可以关联attribute，我们用`getattr`和`setattr`来操作attribute，也可以用`.`

```python
class MyClass:
    version = 3.6
    def say_hello(self):
        print("hello")

setattr(MyClass, 'version', 3.8)
MyClass.version = '3.8'

getattr(MyClass, 'version', 'N/A')

# this add a new attribute to MyClass
setattr(MyClass, 'language', 'python')
# or
MyClass.language = 'python'
```
注意，此时我们还是在讨论`MyClass`作为object本身而不是作为type，我们上面的操作是对`MyClass`这个object本身状态的修改，因此也可以看做meta programming。

> 实际上在Python中，class object和class instance是完全分开的

Python中，`class`对象的的状态都保存在一个dictionary里，比如

```python
MyClass.__dict__


mappingproxy({
    '__module__': '__main__',  # namespace
    'version': 3.6,
    'language': 'python',
    'say_hello': <function __main__.Program.say_hello()>,
    '__dict__': <attribute '__dict__' of 'MyClass' objects>,
    '__weakref__': <attribute '__weakref__' of 'MyClass' objects>,
    '__doc__': None})
```
可以将`mappingproxy`看成一个read-only的map，我们是无法修改它的，因此想要修改class的state，只能通过`setattr`或`.`来完成。

我们还可以delete attribute

```python
delattr(MyClass, 'version')
del MyClass.version
```

接下来我们来看看所谓的instance attribute，还是看下面的代码

```python
my_obj = MyClass()
my_obj.__dict__ # {}
my_obj.language = 'java'
my_obj.__dict__ #{ 'language' : 'java' }
MyClass.language # 'Python'
```
当我们创建了一个`my_obj`时，它的namespace和`MyClass`的namespace是不同的，因此，他们的`__dict__`也不同。对于`my_obj`，当它调用`.language`的时候，python会首先从它自己的`__dict__`中查找，发现没有，接着才会继续向上查找它`class`的attribute，即`MyClass.language`。找到后，当`my_obj`需要需改其值时，它并不会修改`MyClass.language`而是会创建一个instance attribute，即`language`会出现在自己的`__dict__`中。这是一种很巧妙的设计实现了attribute的默认值。

而对于attribute method来说，情况会有些不同，上面例子中，一个method包含下面两部分

1. `__self__` 表示这method被bound到哪个object
2. `__func__` 表示实际的定义在class上的function

当`obj.method(args)`被调用时，Python实际调用的是

```python
mothod.__func__(method.__self__, args)
```
因此，当调用`my_obj.say_hello()`时，实际上调用的时`MyClass.say_hello(my_obj)`。

我们可以用`MethodType`来给`MyClass`的object动态增加一个method

```python
from types import MethodType

obj = MyClass()
obj.foo = MethodType(lambda self: f'foo is called', obj)
obj.foo()
```
此时，`foo`会出现在`obj`的namespace中(`obj.__dict__`)，但是却不会出现在`MyClass`的namespace中。即`foo`只存在于`obj`上，而不存在于`MyClass`这是上

### Properties

`property`会自动给attribute添加getter和setter，使用方式如下

```python
class MyClass:
    def __init__(self, language):
        self._language = language
    def set_language(self, value):
        self._language = value
    def get_language(self):
        return self._language
    language = property(fget=get_language, fset=set_language)

m = MyClass('Python') # m.__dict__ -> {'_language' : 'Python'}
m.language = 'Java' # (1)
```
上面`property`实际上是一个class，当`(1)`执行时，python首先会从`__dict__`中寻找`language`，发现没有，接着会从`MyClass`的`__dict__`中寻找，发现这样一条记录

```python
'language': <property object at 0x7f96c0093770>,
```
此时，`lauguage`是一个property，于是Python会接着找他的getter和setter方法，从而取得想要的结果。

实际上，我们可以将`property`做为一个decorator。`property`的第一个参数是getter，第二个参数是setter，第三个是deleter。

```python
def language(self):
    return self._language

def set_language(self, value):
    self._language = value

language = propery(lauguage)

@property
def language(self):
    return self._language

language = language.setter(set_language)

@language.setter
 def set_language(self, value):
    self._language = value
```
有了property，虽然我们不能彻底的hide attribute，但是可以像objective-c的property一样，可以有readonly的属性以及lazy evaludation

```python
class Circle:
    def __init__(self, r):
        self._r = r
        self._area = None

    @property
    def radius(self):
        return self._r

    @radius.setter
    def radius(self, r):
        if r < 0:
            return
        self._r = r
        self._area = None

    @property
    def area(self):
        if self._area is None:
            self._area = math.pi * (self._r ** 2)
        return self._area
```
上面代码中，我们没有为`area`指定setter，这从一定程度上可以保护`area`不被外部修改。此外，getter中我们实现了caching，不需要每次都进行计算`area`