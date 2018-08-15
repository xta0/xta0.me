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


<p class="md-h-center">(未完待续)</p>

