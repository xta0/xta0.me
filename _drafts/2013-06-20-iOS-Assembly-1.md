---
list_title: iOS中的汇编(一) | Register Calling Convention
title: 寄存器的Calling Convention
layout: post
categories: [iOS, Assembly]
---

### Calling Convention

Calling Convention也叫做调用约定，维基百科的解释为

> 在计算机科学中, 调用约定是一种定义子过程从调用处接受参数以及返回结果的方法的约定。

不同调用约定的区别在于
- 参数和返回值放置的位置（在寄存器中；在调用栈中；两者混合）
- 参数传递的顺序（或者单个参数不同部分的顺序）
- 调用前设置和调用后清理的工作，在调用者和被调用者之间如何分配
- 被调用者可以直接使用哪一个寄存器有时也包括在内。（否则的话被当成ABI的细节）
- 哪一个寄存器被当作volatile的或者非volatile的，并且如果是volatile的，不需要被调用者恢复

iOS中的Calling Convention指的是遵从Apple的CPU体系结构的函数调用规则，了解这个的目的在于它对于理解iOS中的函数调用非常有帮助，即是在没有源码的情况下，如果有一定的汇编语言基础，仍然可以分析出代码的执行逻辑。

### Assembly 101

在上一篇文章中我们介绍了ARM汇编的一些基础知识，并分析了C函数的汇编代码了这里我们简单回顾下，

