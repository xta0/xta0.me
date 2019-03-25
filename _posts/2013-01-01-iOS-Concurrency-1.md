---
updated: "2018-10-22"
layout: post
title: 并发编程为什么难
list_title: iOS中的多线程问题（一） | Why concurrency is HARD
categories: [iOS]
---

新年伊始，我们来聊一聊iOS中的多线程问题，准确来说是并发编程的问题。并发编程在各个语言中都是一个很高阶的话题，这方面相关的书籍和教程非常少，其原因在于它是一个非常复杂的问题，想真正的讲清楚它并不容易，涉及到的知识点非常多，而且非常零散，彼此之间的相关度也比较低。从底层的CPU体系结构到操作系统原理再到上层的编程语言，每一个环节都有可能出错，而且错误往往是随机的，没有规律可循。因此想分析清楚一个并发问题，需要将底层知识和上层知识融汇贯通，有时还可能涉及到一点硬件和汇编语言的知识，甚至还要用到一些高级的调试技术。

### 并发问题的根源

如果从硬件的角度来看，<mark>并发问题的根源在实际上是CPU，内存，I/O这三者读读写的的差异</mark>。这个矛盾一直存在以至于我们往往忽视它的存在。为了平衡这三者之间的速度差异，计算机体系结构，操作系统以及编译原理均作出各自的贡献，主要体现为：

1. CPU增加了缓存，以均衡与内存读写速度的差异
2. 操作系统增加了进程，线程，分时复用等技术用来均衡CPU和I/O设备的速度差异
3. 编译技术优化指令的执行次序，使得缓存可以得到更加合理的利用

这些技术虽然极大的提升了程序的运行效率，却也带来了很多诡异的问题，这些问题也是并发问题的根源，具体有三类，分别是可见性问题，原子性问题和有序性问题

### 可见性问题

在单核时代，所有线程都在同一颗CPU上执行，CPU的缓存与内存的数据一致性容易解决。因为所有线程都是对同一个CPU缓存进行操作，一个线程对缓存的写操作对另一个线程来说是可见的。但是到了多核时代，每颗CPU都有自己的缓存，这时CPU与内存的数据一致性就会有问题。当多个线程在不同CPU上执行时，它们操作的是不同CPU的缓存，对于同一份内存数据，线程1的操作的是CPU-1的缓存，线程2操作的是CPU-2的缓存，它们对各自数据的修改对彼此不可见的，如下图所示


接下来我们可以用一段代码来验证一下多核CPU的可见性问题。下面代码中，我们并发了两个线程，每个线程执行`add10k()`方法，该方法会循环10k次`self.data+1`操作。当两个线程均执行完毕后，我们来观察`self.data`的值是多少呢？

```objc
- (void)add10K {
    int idx = 0;
    while( idx++ < 10000 ){
        self.data += 1;
    }
}

- (void)calculate {
    self.data = 0;
    dispatch_group_t group = dispatch_group_create();
    //run task #1
    dispatch_group_enter(group);
    dispatch_async( dispatch_get_global_queue( 0, 0 ), ^{
        [self add10K];
        dispatch_group_leave(group);
    } );
    //run task #2
    dispatch_group_enter(group);
    dispatch_async( dispatch_get_global_queue( 0, 0 ), ^{
        [self add10K];
        dispatch_group_leave(group);
    } );

    //sychronization point
    dispatch_group_notify( group, dispatch_get_main_queue(), ^{
        NSLog( @"data: %d", self.data );
    } );
}
```
