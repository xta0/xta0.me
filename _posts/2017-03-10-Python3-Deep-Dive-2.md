---
title: 理解变量
list_title: 深入理解Python | 变量 | Variables 
layout: post
categories: [Python]
---

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

综上，在变量时，头脑中要清楚该变量是Immutable的还是mutable的。这个问题在函数传参如果是Immutable的，那么

### Shared References and Memory

回顾前面的例子，Python中的对变量的重新赋值实际上是将变量的指针指向了一块新的内存，但是如果将一个变量的值赋值给另一个变量，会发生什么呢？

```python
a = 10
b = a
```

此时对b的赋值并不会产生对a的拷贝，b也并不会指向新开辟的内存空间，而是和a指向同一片内存空间，类似指针。这时如果我们改变了`a`的值，令`a = 11`，此时会影响`b`吗？由前面的内容可知，由于`a`是Immutable的，`a=11`是令`a`指向了一片新的内存，对`b`没有影响。

因此对于Immutable类型的变量，Python内存管理器会让变量共享内存以节约开销。下面的例子则更能说明这一点：

```python
#a,b同样指向同一片内存区域
a = 10 #0x10375e130
b = 10 #0x10375e130
```

如果变量是mutbable的，这么做就有风险，例如下面代码对`a`指向内存的修改直接会影响b

```python
a = [1,2,3]
b = a
a.append(4)
b #[1,2,3,4]
```

因此对于mutbale类型的变量，Python内存管理器则不会让变量共享内存

```python
a = [1,2,3] #0x103a74348
b = [1,2,3] #0x103a74488
```


<p class="md-h-center">(未完待续)</p>

