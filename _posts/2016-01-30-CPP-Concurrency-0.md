---
layout: post
list_title: 谈谈 C++ 中的多线程 | Concurrency in Modern C++ | 概述 | Overview
title: C++ 中的多线程概述
categories: [C++, Concurrency, Thread]
---

<img src="{{site.baseurl}}/assets/images/2016/01/cpp-con.png" class="md-img-center">


## Overview

C++ 11为C++引入了多线程的概念并以标准库的形式提供了许多线程操作的API，比如atomic variables，线程管理，线程同步，锁，异步任务等。以此为基础，在接下来的C++ 14到C++ 20中，对各种并发API进行了不断地完善和增强。比如C++ 14中增加了`reader-writer`锁的实现等，C++ 17中引入了支持并行计算的STL。

这篇文章我们先对C++多线程涉及的方方面面进行一个初步的介绍，在后面几篇文章中， 会对每个部分进行详细的分析，并结合例子给出实际的用法。

### Memory Model

在C++中，理解Memory Model是理解多线程操作的基础，这也是C++ 11中新引入的一个概念，它包含下面几个方面

1. 支持原子数据类型和对象的原子操作（Atomic Operations）
2. 多线程中的代码执行符合顺序一致性模型（sequential consistency）
3. 保证对象对不同线程的可见性

C++的Memory Model的概念借鉴了Java，但是和Java不同的是，C++的Memory Model支持严格的顺序一致性，并以此作为原子操作的基本原则。顺序一致性可以确保：

1. 在线程中代码执行顺序和源代码书写顺序一致
2. 对并发的所有线程维护一个全局的执行顺序表，保证每个线程执行的时机是可预测的

### Atomics

C++提供了一些内置的简单的原子数据类型，包括boolean，char，numbers和pointers等，这些数据类型可以在多线程环境下直接使用。同时C++也可以使用`std::atomic`模板自定义原子数据类型。

### Threads

C++提供了`std::thread`类，一个`std::thread`对象表示一个executable unit，需要传入一个回调函数/functor，或者一个lambda表达式。对象被创建后会立刻执行，其生命周期需要被手动的管理。具体来说要保证线程可以被正常释放和回收，通常有两种做法：

1. 提供线程同步点，等待线程执行完成。如果没有线程同步，在`std::thread`对象的析构函数中会调用`std::terminate`抛出异常。

2. 使用`detach()`，让线程执行完后自行销毁。

### Locks

一般来说使用锁是为了控制对shared data的并发读写。如果不对线程进行同步控制，则会出现race condition的情况，即一个线程在读数据的时候可能读到正在被另一个线程修改一半的数据。

使用锁来同步线程的方式有很多种，常用的是互斥信号量（Mutexes），mutex可以保证在任意时刻只有一个线程可以访问共享资源，C++中提供了5种不同的mutexes。在锁的实现上C
++使用RAII机制，提供了`std::lock_guard`，`std::unique_lock / std::shared_lock`分别应对不同的场景

如果线程共享数据是只读的，则在初始化时可使用常量表达式或者`std::call_once`

### Condition Variables

条件变量是线程同步的一种方式，它通过消息机制来同步多个线程。当一个线程需要等待另一个线程时，等待线程可以通过监听消息的方式阻塞住直到消息到来。条件变量适用于生产者消费者模型，但是想要用好条件变量并不容易，C++提供了更优雅的方案：Task

### Tasks

Task可以理解为对异步任务的封装，和使用线程不同的是，C++的Runtime可以自动管控task的生命周期，C++中提供不同的task模板，比如最简单的`std::async`，以及稍微复杂一些的`std::future`和`std::promise`。

### Parallel STL

在C++ 17中，并发操作有了比较大的改变，其中最终要的变化是引入了STL的并行算法。在C++ 11/14虽然提供最基本的线程模型，但这些API还是太偏底层了适合于编写framework。而对于应用开发人员，在使用STL容器或者算法时并不希望关心底层的细节。针对这个问题C++ 17为STL算法提供了一个叫做`execution policy`的东西，这个policy可指定算法是同步顺序执行`std::seq`，还是并发执行`std::par`。

C++ 17除了支持69个可并行计算的算法外，还新添加了8个新的算法API，比如`std::reduce`，这些算法补充了C++并行计算的能力，使函数式编程变得更加容易

## C++ 20

<img src="{{site.baseurl}}/assets/images/2016/01/cpp-con.png" class="md-img-center">

C++ 20还未正式标准化，但已经有很多不错的多线程Feature出来了

### Atomic Smart Pointers

C++ 11中引入的两个智能指针`std::shared_ptr`和`std::weak_ptr`并不能保证对其指向资源的访问是原子性的，也就是说如果一个对象被多个指针在多线程的环境下同时访问可能会发生race condition的情况。为了解决这个问题，C++ 20中引入了另外两个线程安全的智能指针`std::atomic_shared_ptr`和`std::atomic_weak_ptr`


### Latches and Barriers

C++ 20中引入了semaphores的概念。所谓semaphore也是一种线程同步的技术，它通过控制信号量的计数器来协调线程间的调度。例如某线程监听某个信号量的值，当该信号量的计数器为0时该线程才能继续执行，而另一个线程在执行完任务后可以修改信号量的计数器的值，从而实现两个线程串行执行的效果。C++ 20中提供了三个类来实现semaphore，分别为`std::latch`,`std::barrier`和`std::flex_barrier`。

### Coroutines

C++ 20还引入了协程，协程这个概念其实并不新鲜，很多现代的编程语言都有协程的相应实现，比如JavaScript和Python的Generator，Go的GoRoutine等。如果想了解协程的原理，可参考[之前的文章](https://xta0.me/2014/02/04/Lua-2.html)。

### Transactional Memory

C++ 20还引入了一个叫做Transactional Memory的概念，如果熟悉DBMS相信对ACID不会陌生，这四个字母分别代表（Atomicity, Consistency, Isolation 和 Durability)。ACID用来表示DBMS的事务特性，简单地说就是事务操作要具备原子性。C++ 20的新标准中也引入了类似的概念（只有ACI，没有D）并给出了相应的实现方式: 使用synchronized block或者atomic block。例如，如果你想让某段代码具备事务的特性，可以将这部分代码放入synchronized block中，这样就可以保证块中代码某一个时刻只有一个线程可以执行，并且块中代码的执行顺序和书写顺序也是一致的。atomic block的效果类似。


## 小结

总结到这里不禁略有感慨，C++进化的一直很慢，一个智能指针就折腾了近4年才想清楚怎么设计，而引入的这些多线程模型早在8，9年前就均已被Apple实现了，比如atomic对象，这个概念早在iOS 5中Objective-C就已经支持了，而semaphores，synchronize block这些C++ 20才提出的东西，也很早就已经出现在Objective-C中了，以至于iOS开发人员对这些名词早就习以为常了。而Apple的Foundation Framework还提供了更加强大的GCD多线程模型，极大地改善了多线程开发的体验，对应用开发非常友好，并且性能也能得到保证。

这说明一门语言的发展还是依赖它所支持的平台的发展，而平台的发展又离不开业务的发展。Apple在这方面做的是真心的不错，API每年都有优化，很为开发者着想，整个开发者生态也运营的不错，实际上OSX/iOS的多线程模型也并非是一蹴而就的，老一点的开发者还应该记得使用`NSThread`或者`pthread`的年代，但是Apple每年都会改进自己的API，不断降低开发门槛，使得越来越多的人可以加入到其产品的开发队伍中。反观C++在这一点上就明显缺乏前进的推动力，在编程语言层出不穷的情况下，C++的生存空间还有多大呢？可能就像左耳朵耗子说的，C++在应用层开发上的优势已经越来越小了，会逐渐被Go取代，C++将会被压的更底层，能想到的还需要用到C++的业务场景有游戏引擎，手机等嵌入式系统，自动驾驶引擎等等这些对代码执行效率有严格要求的底层系统，也正因为这个原因，学习C++也有一定的好处，可以让我们对操作系统有更深入的了解，辩证的看待吧。


## Resources

- [《C++ Concurrency in Action》](https://www.manning.com/books/c-plus-plus-concurrency-in-action?)
- [《C++ Primer 3rd edition》]()
- [C++ 11 Concurrency](https://www.classes.cs.uchicago.edu/archive/2013/spring/12300-1/labs/lab6/)
- [Modern C++ Concurrency in Practice: Get the most out of any machine](https://www.educative.io)