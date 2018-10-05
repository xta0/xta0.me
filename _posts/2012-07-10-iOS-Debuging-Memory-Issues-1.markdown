---
updated: "2018-08-12"
layout: post
list_title: 理解iOS中的内存结构（一） | Debuging Memory Issues in iOS Part 1
title: 理解iOS中的内存结构
categories: [iOS]
---

<em> updated @2014/01/10 : 补充了WWDC2013：session410 </em>
<em> updated @2014/08/12 : 补充了WWDC2018：session416 </em>

排查内存问题，是每个项目都会遇到的。几个项目下来，总结下经验：

- 要了解iOS内存模型的基本概念，懂了这些才能更有效的debug。
- 需要学习一些使用Instrument调试的技巧。
- 深入到内核看一看内存是如何分配的，能否从内核的角度来优化内存的分配和使用

本文主要讨论是第一点



### Virtual vs. Resident

首先我们从寻址开始。
iOS设备CPU是32位，因此寻址空间为：0x00000000 ~ 0xffffffff
例如一个指针地址为0x8031 fea0：
每一位展开成 1000 0000 0011 0001 1111 1110 1010 0000 为32bit。

2^32 = 4G

理论上可以寻址4GB的空间，这个范围远大于设备的物理内存：iPhone4/4S的内存为512MB，iPhone5的内存为1G。

多出的部分怎么处理？

- 虚拟内存

操作系统的RAM有限，所有进程分配的内存总量会超过系统的RAM数量，于是才会有虚拟内存，虚拟内存是中逻辑上的内存，它保证了进程的地址空间不受RAM数量的限制，每个进程都假设自己拥有全部的RAM:寻址范围从0x00000000~0xffffffffff。有了虚拟内存，操作系统可以使用硬盘来缓存RAM中无法保存的数据。但使用虚拟内存带来的一个问题是：进程使用的内存地址与物理RAM并不对应，也就是说我们程序中看到的内存地址并不是物理地址，而是虚拟内存地址。这就需要内核做一个从虚拟内存到物理RAM的映射。


回忆我们以前学的微机原理，内存是按照4KB为一帧被划分开。虚拟内存同样按照4KB为一页来划分，每页的字节数与每帧的字节数始终相同。这样便可以将进程中的每页无缝映射到物理RAM中的每帧。WWDC的这张图说明了上面的过程：

<a href="/assets/images/2014/01/virtual_mem.png"><img src="{{site.baseurl}}/assets/images/2014/01/virtual_mem.png" alt="virtual_mem" width="625" height="342"/></a>

虚拟内存这么划分的好处是，可以将进程中连续的地址空间映射到物理RAM中不连续的地址空间中，这可以最大限度节省内存碎片的产生。因此，当一个进程启动时，OS会创建一张表，来保存虚拟内存到物理RAM的映射关系，这个表被成为“分页表”，类似windows中的HANDLE。然后我们讨论两种情况：

- 当RAM被写满后怎么办？

在OS X中，会将某些不活跃的帧存到磁盘。但是在iOS中，会直接将某些不活跃的帧清空！也就是进程在后台被系统杀死了，这主要是由于RAM和磁盘进行数据交换会极大的影响性能。

- 如果进程访问的地址找不到怎么办？

如果OS确定进程访问的地址是错误的，则报错，终止进程；如果进程访问的地址被保存到了硬盘上，OS首先分配一帧，用来保存请求页。如果当前没有可用帧，将现有帧缓存到磁盘，腾出空间。然后将请求页读到内存中，更新进程的页表，最后将控制权返回给进程。

Resident Memory是进程的virtual memory中常驻在物理RAM中的部分。

<h3>Heap vs. Stack</h3>

mike ash的<a href="https://www.mikeash.com/pyblog/friday-qa-2010-01-15-stack-and-heap-objects-in-objective-c.html">这篇文章</a>总结的很详细。除了在Heap上创建的Object，还有很多看不到的内存被占用，比如：layer的backing store；代码段和常量段(__TEXT,__DATA)，Thread stack，图片数据，cache等等。

### Memory Footprint

操作系统中的内存按页分布，一般来说每个Page的大小为16KB，我们平时分配的heap object都存放在Page中，系统占用内存的大小可以用下面式子计算：

```
Memory in use = Number of pages x page size
```

每个Page中的内存又为两种类型：

1. Clean
2. Dirty

例如下面代码：

```c
int *array = malloc(20000 * sizeof(int));
array[0] = 32
array[19999] = 64
```



通常一个进程中有三种类型的内存：clean，dirty和free：

clean的内存是：硬盘上应用程序的二进制代码（代码段），常量段（__TEXT段），系统的各种framework，上面提到的分页表（memory-mapped files）等，clean内存在memory warning的时候可以被discard掉，然后recreate出来。

dirty的内存是：应用程序产生的数据，包括Heap上分配对象的控件，UIImage，Caches等。dirty内存在memory warning时候会被清空

例如：
```objc
NSString* str = [NSString stringWithUTF8String:"welcome"];
```

这属于动态分配在heap上变量，为dirty memory，会被回收,但是

```objc
NSString* str = @"welcome";
```

是clean的，因为这个字符串在编译的时候会被存放在程序代码段中的read-only的常量区

例如：

```objc
UIImage* wwdcLogo = [UIImage imageNamed:@"WWDC12logo"];
```

由于UIImage是decompress出来的data，是dirty memory

当我们的app在前台运行时，不断消耗内存，导致内存不足时，系统首先会将不用的clean memroy干掉一部分，腾出空间来继续创建drity memory，当dirty memory越来越多，又导致内存不足时，系统会将运行在后台app的dirty memory干掉，然后将之前干掉的clean memory重新load回来。

<h3>Private vs. Shared</h3>

RAM中可以被多个进程共享的部分称为Shared Memory，比如系统的framework，它只映射一份代码到内存，这部分内存会被不同的进程共用。而每个进程单独alloc的内存，则是Private Memory。



