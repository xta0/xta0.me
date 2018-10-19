---
layout: post
list_title: 谈谈 C++ 中的多线程 | Concurrency in Modern C++ | 概述 | Overview
title: 概述 
categories: [C++, Concurrency, Thread]
---

<img src="{{site.baseurl/assets/images/2016/01/cpp-con.png}}" class="md-img-center">


## C++ 11: The foundation

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

C++ 20还未标准化，但已经很多不错的多线程Feature出来了：

### Atomic Smart Pointers



### Coroutines

