---
title: 对变量的编译优化
list_title: 深入理解Python | 对变量的编译优化 | Variables compiler optimizations
layout: post
categories: [Python]
---

## 常识

在介绍Python对变量的编译优化前，我们先来复习下变量在Python中的一些常识

### Mutable & Immutable

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
此时`a`的"类型"变成了string，表面上看，上述代码实际上是对`a`的重新赋值，因此一个直观的想法是`a`所指向的内存数据被修改了，从`10`被修改为`hello`，也就是所谓的mutation。实际上则不是，重新复制后的`a`指向的是另一块新的内存，其值为`hello`，原来`10`的那块内存仍然存在，只是没有人引用它的，很快便会被GC回收。

我们可以用代码试验一下上述结论

```python
print(hex(id(a))) #x10c28eb90
a = 15
print(hex(id(a))) #0x10851fc30
```

可见`a`指向的内存地址发生了变化，而并非是修改原来内存地址中的值。这种情况下，我们称`a`是Immutable的，Python中有一些数据结构是Immutable的，包括

|Immutable| Mutable|
|-------| --------|
| Numbers(int, float, Booleans, etc)| Lists| 
| Strings | Sets |
| Tuples | Dictionaries |
| Frozen Sets| User-Defined Classes |
| User-Defined Classes| |

对于Mutbale的数据结构，我们可以修改其内容，然后观察其地址

```python
#1
my_list = [1,2,3]
print(hex(id(my_list))) #0x10ae13f48
my_list.append(4)
print(hex(id(my_list))) #0x10ae13f48
#2
my_str = "123"
print(hex(id(my_str))) #0x10da15030
my_str += "4"
print(hex(id(my_str))) #0x10da15068
```

第一个例子，`append`会对`my_list`指向内存中的数据进行修改，但是`my_list`自己的内存地址并不发生变化，复合预期。第二个例子，`my_str`是Immutable的，因此第二个`my_str`是一个新的`my_str`，`+=`操作并未对旧的的`my_str`进行修改

综上所述，在使用变量时头脑中要清楚该变量是Immutable的还是mutable的。这个问题在函数传参时尤为明显，如果pass了一个mutbale变量到函数内部，可能存在side effect。

### Shared References and Memory

回顾前面的例子，Python中的对变量的重新赋值实际上是将变量的指针指向了一块新的内存，但是如果将一个变量的值赋值给另一个变量，会发生什么呢？

```python
a = 10  #0x10ac99130
b = a #0x10ac99130
```

此时对b的赋值并不会产生对a的拷贝，b也并不会指向新开辟的内存空间，而是和a指向同一片内存空间，相当于b是a的一个别名。这时如果我们改变了`a`的值，令`a = 11`，此时会影响`b`吗？由前面的内容可知，由于`a`是Immutable的，`a=11`是令`a`指向了一片新的内存，对`b`没有影响。

因此对于Immutable类型的变量，Python内存管理器会让变量共享内存以节约开销。下面的例子则更能说明这一点：

<div class="md-flex-h md-margin-bottom-24">
<div>
<pre class="highlight language-python md-no-padding-v md-height-full">
<code class="language-python">
#a,b同样指向同一片内存区域
a = 10 #0x10375e130
b = 10 #0x10375e130
</code>
</pre>
</div>
<div class="md-margin-left-12">
<pre class="highlight md-no-padding-v md-height-full">
<code class="language-python">
a=None #0x101031148
b=None #0x101031148
</code>
</pre>
</div>
</div>

Python中的`None`对象也是共享内存的，所有被赋值为`None`的变量，它们所指向的内存地址相同

> `None`表示某个变量的值为"empty"，类似JavaScript中的`Undefined`，也可以理解为该变量并没有被赋值。Python中，`None`是一个具体的Object，被Memory Manager管理

如果变量是mutbable的，这么做就有风险，例如下面代码对`a`指向内存的修改直接会影响b,因此对于mutbale类型的变量，Python内存管理器则不会让变量共享内存。

<div class="md-flex-h md-margin-bottom-24">
<div>
<pre class="highlight language-python md-no-padding-v md-height-full">
<code class="language-python">
a = [1,2,3]
b = a
a.append(4)
b #[1,2,3,4]
</code>
</pre>
</div>
<div class="md-margin-left-12">
<pre class="highlight md-no-padding-v md-height-full">
<code class="language-python">
a = [1,2,3] #0x103a74348
b = [1,2,3] #0x103a74488
</code>
</pre>
</div>
</div>

### Variable Equality

Python中有两种Equality，一种是内存地址相同（identity），用`is`来表示，另一种是使用`==`表示两个对象的"值"（内部state）是否相同（equality），与之相对应的不等关系则用`is not`和`!=`表示。我们看几个例子


<div class="md-flex-h md-margin-bottom-24">
<div>
<pre class="highlight language-python md-no-padding-v md-height-full">
<code class="language-python">
a=10
b=a
a is b #true
a == b #true
</code>
</pre>
</div>
<div class="md-margin-left-12">
<pre class="highlight md-no-padding-v md-height-full">
<code class="language-python">
a = [1,2,3]
b = [1,2,3]
a is b #false
a == b #true
</code>
</pre>
</div>
<div class="md-margin-left-12">
<pre class="highlight md-no-padding-v md-height-full">
<code class="language-python">
a=10
b=10.0
a is b #false
a == b #true
</code>
</pre>
</div>
<div class="md-margin-left-12">
<pre class="highlight md-no-padding-v md-height-full">
<code class="language-python">
a = None
b = None
a is None #true
a == b #true
</code>
</pre>
</div>
</div>

### Everything is Object

Python中所有的变量有自己的类型，它们的类型都是某种class，就连`class`本身也是某个class的实例，因此可以说Python中的所有变量都是Object。

以函数为例，和绝大多数现代编程语言一样，函数在Python中也是first-class的，可以被像变量有自己的类型，可以被做为参数传递，赋值，以及做函数的返回值。当我们定义一个函数时，函数名是Object的名称，它有自己的内存地址

```python
def square(a):
    return a**2

type(square) #<class 'function'>
print(hex(id(square))) #0x1011b4268
f = square
f is square #ture
```

此外，一些基本的数据类型比如`int`,`float`等在python中也是以`class`形式存在的，在第一小节的第一个例子中，我们看到了`a`的类型为`<class 'int'>`，`int`类的定义如下

```
>>> help(int)
class int(object)
 |  int([x]) -> integer
 |  int(x, base=10) -> integer
 |
 |  Convert a number or string to an integer, or return 0 if no arguments
 |  are given.  If x is a number, return x.__int__().  For floating point
 |  numbers, this truncates towards zero.
 |
 |  If x is not a number or if base is given, then x must be a string,
 |  bytes, or bytearray instance representing an integer literal in the
 |  given base.  The literal can be preceded by '+' or '-' and be surrounded
 |  by whitespace.  The base defaults to 10.  Valid bases are 0 and 2-36.
 |  Base 0 means to interpret the base from the string as an integer literal.
 |  >>> int('0b100', base=0)
 |  4
 ...
```
我们可以像创建对象一样来创建一个`int`型变量

```python
c = int()
c #0
c = int(10)
c #10
c = int('101',base=2) #二进制数
c #5
```

## 变量的编译器优化

有了上面的知识做铺垫，我们可以开始研究Python的编译器是如何优化变量的

### Interning

回顾前面的例子，对于两个具有相同值的Immutable对象，它们的内存是共享的，但某些情况下，却有例外，如下面代码所示

```python
a = 10 #0x10375e130
b = 10 #0x10375e130

a=500 #0x1015ed110
b=500 #0x1015ed030
```
为了解释这个问题，需要理解Python的Interning机制，所谓Interning是Python编译器中的一种内存优化技术，它会"按需"的重用对象。不同版本的Python引擎对这个机制的实现不同，以标准的CPython为例，当执行Python代码前，会提前创建一批整型单例(Singletons)对象(范围从`[-5, 256]`)，并将他们缓存起来，当程序中需要使用这些数值，直接从缓存中取出而不会重新创建对象，这就是为什么`a,b`指向同一块内存地址的原因，对于缓存中没有的数值，则会重新创建一份，这是为什么`a,b`的值为`500`之后，它们各自指向了不同的地址。

```python
a = -5 #0x10107af50
b = -5 #0x10107af50

a = -6 #0x1015ed110
b = -6 #0x1015ed030
```

对于字符串，由前面小节可知，也有类似的Interning，例如:

```python
a = 'some_long_string'
b = 'some_long_string'
a is b # true
```
如果整型数据的预加载是为了减少内存开销，提高程序运行的速度，那么对于字符串，我们可以思考一下这种Interning机制有什么好处呢？

能想到的一个好处是字符串比较，对于两个字符串`a`,`b`当我们使用`a==b`进行比较时，会逐个字符比较(或者用高级一点的KMP之类的算法，但思路还是遍历字符并比较)，但是我们如果知道`a is b`即`a,b`来指向同一片内存，那么则不用进行比较了，可以直接返回true，这显然是最快的方法。

但和前面介绍的整型一样，不是所有的字符串都可以被interning，一些被编译器判定为`identifier`特征的字符串会被interned，例如下面两个字符串则不会被interned

> identifier是指只包含数字，字母，下划线的字符串

```python
a = 'hello world'
b = 'hello world'
a is b #false
a == b #true
```
使用`identifier`作为判断标准的一个原因是这些字符可能会表示函数名，类名等，在runtime中可能被用到。Python也提供了API可以强行令string被interned:

```python
import sys
a = sys.intern('the quick brown fox')
b = sys.intern('the quick brown fox')
a is b #true
```

### Peephole

Peephole是Python编译器的另一项优化技术，他可以对一些常量表达式进行提前求值，并将结果缓存，例如

```python
def my_func():
    a = 24*60  #1440
    b = (1,3)*5 #(1,3,1,3,1,3,1,3,1,3)
    c = 'abc'*3 #abcabcabc
    d = 'ab'*11
    e = 'the quick brown fox' * 5
    f = ['a','b']*3
```

对上述函数，编译器会对函数中的常量表达式进行提前求值，求值结果可以通过`my_func.__code__.co_consts`输出

```
(None, 
1440, 
(1, 3, 1, 3, 1, 3, 1, 3, 1, 3), 
'abcabcabc', 
'ababababababababababab', 
'the quick brown foxthe quick brown foxthe quick brown foxthe quick brown foxthe quick brown fox', 
'a', 'b', 3
)
```
我们看到上面`f`并没有被求值，原因是`['a','b']`是mutable数组，并非是常量表达式，因此编译器不会对这条语句进行优化。

除了常量表达式的提前求值，Python编译器还会对某些集合操作进行优化，比如对检查某个成员是否属于某个集合的优化：

```python
if e in [1,2,3]:
```
上述代码中`[1,2,3]`是mutable的object，会被编译器将类型替换为Immutable的类型tuple`(1,2,3)`。

我们来看一个具体例子

<div class="md-flex-h md-margin-bottom-24">
<div>
<pre class="highlight language-python md-no-padding-v md-height-full">
<code class="language-python">
#using array
def my_func(e):
    for e in [1,2,3]:
        pass
</code>
</pre>
</div>
<div class="md-margin-left-12">
<pre class="highlight md-no-padding-v md-height-full">
<code class="language-python">
#using set
def my_func(e):
    for e in {1,2,3}:
        pass
</code>
</pre>
</div>
</div>

执行`my_func.__code__.co_consts`，结果为

```
#using tuple
(None, (1, 2, 3))

#using set
(None, frozenset({1, 2, 3}))
```
说明编译器将数组`[1,2,3]`转化成了tuple，将set转成了`frozenset`。

总的来说，对集合操作的优化是将mutable版本转化为对应的Immutable版本的。编译器做这种优化的目的还是为了提高程序运行的速度，显然array，tuple，set三者里面set的查询速度速度最快`O(1)`，我们可以用一个demo来验证

```python
import string
import time
letters = string.ascii_letters
char_list = list(letters)
char_tuple = tuple(letters)
char_set = set(letters)
print(char_list)

def membership_test(n,container):
    for i in range(n):
        if 'z' in container:
            pass

#test array
start = time.perf_counter()
membership_test(1000000,char_list) #一百万次
end = time.perf_counter()
print('list:',end-start) #list: 0.461002645

#test tuple
start = time.perf_counter()
membership_test(1000000,char_tuple) 
end = time.perf_counter()
print('tuple:',end-start) #tuple: 0.44785893200000004

#test set
start = time.perf_counter()
membership_test(1000000,char_set)
end = time.perf_counter()
print('set:',end-start) #set: 0.04525098400000005
```

{% include _partials/post-footer-1.html %}

### Resource

- [CPython](https://github.com/python/cpython)
- [The internals of Python string interning](http://guilload.com/python-string-interning/)
- [A Peephole Optimizer for Python](https://pdfs.semanticscholar.org/a456/b29641318e078c3780289a9ee688ff09e5d7.pdf)