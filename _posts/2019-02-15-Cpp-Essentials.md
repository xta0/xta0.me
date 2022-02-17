---
layout: post
list_title: C++ Essentials | Memory Management
title: Memory Management
categories: [C++]
---

### Virtual Memory 101

我们知道程序访问的地址是虚拟内存地址，CPU中的MMU(Memory Management Unit)会将虚拟内存地址映射到实际的物理内存地址。常见的虚拟内存实现方式是将地址空间按照固定的大小划分成若干个block，每个block也叫做**memory page**。当程序访问某个memory page时，系统会检查该memory page是否有对应的物理内存(memory frame)，如果没有，则会触发**page fault**，将数据从disk加载到内存中。

当物理内存不够时，系统需要清理一些memory page，如果当前page是dirty的(有过写操作)，则系统会将其swap到disk上，如果是clean的page，则系统会直接将其清理，这个过程也叫**paging**。实际上并不是每个操作系统都会swap内存，iOS上如果发生内存紧张，系统会首先kill掉一些进程来释放内存，dirty page是不会被swap到disk上的。

下图展示了连个进程的虚拟内存情况，假设进程1要执行`0x1000`中的代码，则会触发page fault，系统会从disk加载数据到一片空的物理内存中

<img class="md-img-center" src="{{site.baseurl}}/assets/images/2019/02/vm.png">

如果系统处于一个low memory的状态，内存置换会非常频繁，这种情况叫做**Thrashing**。

### Stack Memory

Stack Memory有下面一些特点

- Stack Memory是一块连续的内存空间
- Stack有固定大小，可以通过`ulimit -s`命令查看stack大小的最大值(bytes)
- Stack Memory永远不会产生内存碎片
- 从Stack上分类内存速度快，基本不会发生Page fault
- 每个线程都有自己的stack，stack的内存分配不用考虑并发的问题