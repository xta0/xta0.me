---
list_title: Python | Gotchas | 常见的mistakes
title: Python Gotchas
layout: post
categories: ["Python"]
---

### 解释器的差别

看下面代码，你觉得那个是正确的呢？还是都不正确或者都正确呢？

<div class="highlight md-flex-h md-margin-bottom-24">
<div>
<pre class="highlight language-javascript md-no-padding-v md-height-full">
<code class="language-python">
#Javascript
function func1(){
	return func2();
}

func1();

function func2(){
	return "running func2";
}
</code>
</pre>
</div>
<div class="md-margin-left-12">
<pre class="highlight language-python md-no-padding-v md-height-full">
<code class="language-python">
#python
def func1():
    return func2() 

func1()

def func2():
    return "running func2"
</code>
</pre>
</div>
</div>

上面的两段代码，Javascript代码可以正常执行，Python代码则报错。错误的原因是:

```
NameError: name 'func2' is not defined
```

从这个例子可以看出，Python的解释器的设计和Javascript似乎有些区别。在分析具体原因之前，先来回顾一下[编程语言的原理](2014/04/24/Programming-Language-1-1.html)，对于任何一条表达式，编译器都需要确定三个问题

1. Syntax
2. Type-Checking Rules
3. Evaluation Rules

对于Function来说，在编译器确定完其类型后便将这个符号（函数名或者是按照某种规则mangle后的名字）放入了static environment中，留着运行时调用。而函数的Evaluation的规则是在运行时求值，对函数内部的符号是从static environment中寻找，找不到则报错。上面例子中，在执行`func1()`时，Python和JS均会在static environment中寻找`func2`，显然一个找到了，另一个没找到，因此，分歧可能出在`func2`这个符号注册的时机上。

接下来，我们可以大致分析一下JS和Python的解释器是怎么工作的。对于JS来说，在执行前代码前，对所代码从头至尾进行扫描，如果出现static environment中没有的符号，则向其内部注册该符号，并赋初值undefined（这个特性据说叫做Hoisting）。注意在static environment中并不会对符号求值，求值的过程在dynamic environment中。而python的解释器似乎不会提前在static environment中注册所有符号，而是在运行时不断更新static enviroment中的符号, 并在dynamic environment中对其求值，当然如果发现没该有符号，则会在求值的过程中报错。

哪种设计合理呢？感觉Python解释器的设计更合理一些，JS在执行前要扫描并注册所有符号，其效率显然不如逐句解释来的快，并且一般有良好变成素养的程序员也不会写出上面的代码。

### Lists vs Tuples

我们可以先对比一下List和Tuple的bytecode

```shell
# compile tuple
dis(compile('(1, 2, 3, "a")', 'string', 'eval'))

0 LOAD_CONST      0 ((1, 2, 3, 'a'))
2 RETURN_VALUE

# compile list
dis(compile('[1, 2, 3, "a"]', 'string', 'eval'))

0 LOAD_CONST               0 (1)
2 LOAD_CONST               1 (2)
4 LOAD_CONST               2 (3)
6 LOAD_CONST               3 ('a')
8 BUILD_LIST               4
10 RETURN_VALUE
```

上面例子中，无论是tuple还是list，他们包含的是immutbale的data，我们可以发现，bytecode中tuple是一个constant，可以直接load，
而list需要通过更多的指令去动态创建。这是因为tuple本身是Immutable的数据结构。但是，如果tuple中包含mutable的数据结构，则tuple的创建将变的和list一样

```python
dis(compile('([1, 2], 3, "a")', 'string', 'eval'))

0 LOAD_CONST               0 (1)
2 LOAD_CONST               1 (2)
4 BUILD_LIST               2
6 LOAD_CONST               2 (3)
8 LOAD_CONST               3 ('a')
10 BUILD_TUPLE             3
12 RETURN_VALUE
```
此外，由于list的动态性，它需要preallocate space，因此它比tuple在内存上有更大的overhead

```python
import sys
s1 = sys.getsizeof((1,)) #48
s2 = sys.getsizeof([1,]) #64
```

### Shallow Copy vs Deep Copy

Python中有很多方法来copy一个list

```python
s = [1,2,3]

#1. for loop
cp = [e for e in s]

#2. copy()
cp = s.copy()

#3. slicing
cp = s[:]
cp = s[0:len(s)]

#4. list()
cp = list(s)
```
需要注意的是这里copy全是shallow copy，即`cp`是一个新的list，但是里面的元素还是指向原来list中的元素，即

```python
id(s) != id(cp)
```
对于immutable的sequence，如tuple，copy则不会创建新的对象

```python
t1 = (1, 2, 3)
t2 = tuple(t1)
id(t1) == id(t2)
```
Shallow copy的一个问题是，如果mutate原来的list中存在mutable的object，则同样会一向copy后list中的元素

```python
s1 = [[10, 20], 3, 4]
s2 = s1.copy()
s1[0][0] = 100 # s1 = [[100, 20], 3, 4]
# s2[0][0] will also be mutated
```
此时我们需要使用deep copy，但是实现deep copy是一件不容易的事情，比如一个list中有sublist，sublist中又有sublist，因此deep copy需要考虑递归的情况。对于标准库存中的sequence，Python提供`.deepcopy()`的API来实现deep copy

```python
import copy
x = [1, 2, 3]
y = copy.deepcopy(x)
```

### Mutable objects

如果concat两个包含mutable object的list，则结果

```python
x = [[1, 2]]
y = x + x
# y -> [[1, 2],[1, 2]] 
x[0] = [-1, -1]
# y -> [[-1, -1], [-1, -1]] 
```
实际上 `id(y[0]) == id (y0[1]) == id(x)`
