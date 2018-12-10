---
layout: post
title: iOS中的线程同步问题 
list_title: iOS中的线程同步问题 | Thread Synchronization in iOS
categories: [iOS]
---

## 线程同步

考虑这样一个场景，假如我们有一个计数器从0开始递增，现在有100个线程并发同时修改计数器的值，怎么保证计数器的值有序递增的输出？

```objc
dispatch_queue_t queue =
dispatch_queue_create(NULL, DISPATCH_QUEUE_CONCURRENT);
for(int i=0;i<100;i++){
   dispatch_async(queue, ^{
            self.j +=1;
            printf("%d\n",self.j);
        });
}
```
上述代码中，由于线程是并发的，导致`self.j`在`print`时I/O缓冲区中的数据并不是当前最新的，因此输出的`self.j`的值是乱序的。这个问题的本质是线程之间的同步问题，block中两句代码的执行存在时间差。如果想要保证输出顺序，我们需要强制每个线程执行完这两行代码后，其它线程才能开始执行，即block中的代码具备原子性。

- 使用FIFO的无锁队列

```objc
//修改queue为串行队列
dispatch_queue_t queue =
dispatch_queue_create(NULL, DISPATCH_QUEUE_SERIAL);
```
- 使用`@synchronize`代码块

```objc
@synchronized (self) {
	self.j = self.j+1;
	printf("%d\n",self.j);
}
```
- 使用Barrier

```objc
dispatch_barrier_async(queue, ^{
	self.j = self.j+1;
	printf("%d\n",self.j);
});
```

- 使用锁

```objc
dispatch_async(queue, ^{
	std::lock_guard<std::mutex> guard(_m);
	self.j = self.j+1;
	printf("%d\n",self.j);
});
```

## Locks

常用的锁有下面几种：

- `Mutex locks`: 互斥锁是一种信号量，在某个时刻只允许一个线程对资源进行访问，如果互斥锁正在被使用，另一个线程尝试使用，那么这个线程会被block，直到互斥锁被释放。如果多个线程竞争同一个所锁，只有一个线程能获取到。互斥锁对应POSIX中的实现是`pthread_mutex_t`，Objective-C`中的@synchronized`关键字底层是对`pthread_mutex_t`的封装

- `Spin locks`: 自旋锁的原理是不断check lock条件，直到条件为true。自旋锁经常被用在多核处理器上并且lock时间很短的场合，如果lock时间很长，则会耗尽CPU资源

- `Reader/writer locks`: 读写锁，这种锁通常用在"读"多，"写"少的场合。当写操作发生时，该线程会先被block，直到所有"读"操作完成。对应POSIX的实现是`pthread_rwlock_t`

- `Recursive locks`: 递归锁是互斥锁的一个变种，它允许某一个线程在释放锁之前可以多次获取，其它线程只能等待获取它的线程释放，它最初设计被用来做函数的递归调用，但是也可以用在多个方法同时需要获取一个lock的场合




## Memory Barrier

为了达到最佳性能，编译器通常会对汇编基本的指令进行重新排序来尽可能保持处理器的指令流水线。作为优化的一部分，编译器有可能对访问主内存的指令，如果它认为这有可能产生不正确的数据时，将会对指令进行重新排序。不幸的是，靠编译器检测到所有可能内存依赖的操作几乎总是不太可能的。如果看似独立的变量实际上是相互影响，那么编译器优化有可能把这些变量更新位错误的顺序，导致潜在的不正确结果。

为了解决这个问题，我们可以使用内存屏障。所谓内存屏障（memory barrier）是一个使用来确保内存操作按照正确的顺序工作的非阻塞的同步工具。<mark>内存屏障的作用就像一个栅栏，迫使处理器来完成位于障碍前面的任何加载和存储操作，才允许它执行位于屏障之后的加载和存储操作</mark>。内存屏障同样可以用来确保一个线程（但对另外一个线程可见）的内存操作总是按照预定的顺序完成。如果在这些地方缺少内存屏障有可能让其他线程看到看似不可能的结果。为了使用一个内存屏障，你只要在你代码里面需要的地方简单的调用`OSMemoryBarrier()`函数。`OSMemoryBarrier()`定义在`OSAtomic.h`中，ReactiveCocoa中,通过`OSMemoryBarrier()`保证`_disposeBlock`的赋值

```objc
- (id)init {
	self = [super init];
	if (self == nil) return nil;

	_disposeBlock = (__bridge void *)self;
	OSMemoryBarrier();

	return self;
}
```

## Volatile

`Volatile` 变量适用于独立变量的另一个内存限制类型。编译器优化代码通过加载这些变量的值进入寄存器。对于本地变量，这通常不会有什么问题。但是如果一个变量对另外一个线程可见，那么这种优化可能会阻止其他线程发现变量的任何变化。<mark>在变量之前加上关键字volatile可以强制编译器每次使用变量的时候都从内存里面加载</mark>。如果一个变量的值随时可能给编译器无法检测的外部源更改，那么你可以把该变量声明为`volatile`变量。

```c
int *ip = 1;
*ip = 2;
*ip = 3;
```
会被编译器优化为:

```c
int *ip = 1;
*ip = 3;
```

如果使用`volatile`修饰，则编译器就不会对`*ip`进行优化。多线程中使用`volatile`要考虑下面两种情况：

1. 在本线程内, 当读取一个变量时，为提高存取速度，编译器有时会先把变量读取到一个寄存器中；以后，再取变量值时，就直接从寄存器中取值；当变量值在本线程里改变时，会同时把变量的新值copy到该寄存器中，以便保持一致。<mark>但是当变量在因别的线程等而改变了值，该寄存器的值不会相应改变，从而造成应用程序读取的值和实际的变量值不一致。</mark>

2. 当该寄存器在因别的线程等而改变了值，原变量的值不会改变，从而造成应用程序读取的值和实际的变量值不一致。

```c
int square(volatile int *ptr){
	return *ptr * *ptr;
}
```

该程序的目的是用来返指针`*ptr`指向值的平方，但是，由于`*ptr`指向一个`volatile`型参数，编译器将产生类似下面的代码：

```c
int square(volatile int *ptr){
	int a,b;
	a = *ptr;
	b = *ptr;
	return a * b;
}
```
上述代码中，`a`,`b`将位于CPU的寄存器内。但是由于`*ptr`的值可能被其它线程意想不到地改变，因此`a`和`b`可能会发生变化。结果，这段代码可能返不是所期望的平方值。正确的代码如下：

```c
long square(volatile int *ptr){
	int a;
	a = *ptr;
	return a * a;
}
```
> 频繁地使用volatile很可能会增加代码尺寸和降低性能,因此要合理的使用volatile。

## Resources

- [Synchronization](http://www.dreamingwish.com/article/the-ios-multithreaded-programming-guide-4-thread-synchronization.html)
- [几种锁的性能比较](http://perpendiculo.us/2009/09/synchronized-nslock-pthread-osspinlock-showdown-done-right/)
- [Friday Q&A 2017-10-27: Locks, Thread Safety, and Swift: 2017 Edition](https://www.mikeash.com/pyblog/friday-qa-2017-10-27-locks-thread-safety-and-swift-2017-edition.html)
- [Friday Q&A 2015-05-29: Concurrent Memory Deallocation in the Objective-C Runtime](https://www.mikeash.com/pyblog/friday-qa-2015-05-29-concurrent-memory-deallocation-in-the-objective-c-runtime.html)
- [Friday Q&A 2015-02-20: Let's Build @synchronized](https://www.mikeash.com/pyblog/friday-qa-2015-02-20-lets-build-synchronized.html)
- [Friday Q&A 2013-08-16: Let's Build Dispatch Groups](https://www.mikeash.com/pyblog/friday-qa-2013-08-16-lets-build-dispatch-groups.html)
- [Friday Q&A 2013-03-08: Let's Build NSInvocation, Part I](https://www.mikeash.com/pyblog/friday-qa-2013-08-16-lets-build-dispatch-groups.html)
- [Friday Q&A 2013-03-08: Let's Build NSInvocation, Part 2](https://www.mikeash.com/pyblog/friday-qa-2013-03-22-lets-build-nsinvocation-part-ii.html)
- [Friday Q&A 2012-11-30: Let's Build A Mach-O Executable](https://www.mikeash.com/pyblog/friday-qa-2012-11-30-lets-build-a-mach-o-executable.html)
- [Friday Q&A 2012-11-16: Let's Build objc_msgSend](https://www.mikeash.com/pyblog/friday-qa-2012-11-16-lets-build-objc_msgsend.html)