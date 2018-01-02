---

layout: default
categories: UNIX
tags: [Tools,UNIX]
title: UNIX Notes

---

## UNIX Tools

### MTR

- [mtr on linode](https://www.linode.com/docs/networking/diagnostics/diagnosing-network-issues-with-mtr)

### cURL

- [Send HTTP Request using cURL](http://docs.couchdb.org/en/latest/intro/curl.html)

## UNIX Signal

### SIGSEGV

`segment fault`是操作系统抛出的。原因有两种：

- 硬件引起的错误
- 访问了一个不可读取的内存地址或者向一个不可写的内存地址写数据。

通常情况下是第二种情况引起的。默认情况下代码段(code pages)是只读的，数据段(data pages)是不可执行的。当代码段的数据被试图修改或者访问脏数据时，都会抛出这个异常。`SIGSEGV`最常见的错误是指针类型转换错误。

### SIGBUS

这种异常通常也是`access bad memory access`引起的，通常是指针访问的物理内存地址不存在，这种错误和`SIGSEGV`都是`EXC_BAD_ACCESS`的一种。

### SIGTRAP

`SIGTRAP`的含义是`signal trap`，通常它不是crash的信号。当CPU执行的指令为`trap`时，会抛出这个信号。LLDB通常会捕获这个信号，并断到所在的指令位置。如果查不出明显的问题，重新clean再build可能会fix这个问题。

### EXC_ARITHMETIC

如果抛出这个异常，说明有某些运算`0`做了除数

### SIGILL

`SIGILL`表示`SIGNAL ILLEGAL INSTRUCTION`。当CPU执行非法的指令时会抛出这个异常。通常情况下是由函数地址无效引起的。`EXC_BAD_INSTRUCTION`也是相同的情况。

### SIGABRT

`SIGABRT`是`SIGNAL ABORT`的意思，这个信号通常是系统的framework主动抛出的，例如`UIKit`中出现某些不合理的判断会调用C函数：`abort`。这个函数会抛出`SIGABRT`信号。对于这种类型的crash，通常系统会保留大量有用的信息，在LLDB中使用`bt`能看到丰富的堆栈信息

