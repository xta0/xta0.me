---
layout: post
list_title: 谈谈 C++ 中的多线程 | Concurrency in Modern C++  | 数据共享 | Sharing Data
title: 线程间共享数据 
categories: [C++, Concurrency, Thread]
---

如果有某种数据需要被两个线程共享，如果该数据是`read-only`则没有问题，如果是可读写的，则会有几率出现`race condition`的情况，即一个线程在进行读操作时，另一个线程在进行写操作，这时对于读操作的线程将有机率读到不完整的数据。共享数据对于多线程来说是一个很经典的问题，解决这个问题的办法有很多种，比如设计无锁的数据结构，使用原子操作等，这些将在后面的文章中做具体分析，本节将先介绍最基本的互斥锁

### 使用mutex

**互斥(mutex exclusive)**是指当某个线程已经获得某个数据的控制权时，其它线程如果想访问该数据只能等待前面线程释放控制权。具体来说，当某线程需要访问共享数据时，首先需要lock the mutex associated with the data，当访问完成时，unlock the mutex。但实际情况往往并不是这么简单，后面我们会分析种种情况。

C++中提供了`std::mutex`类来操作mutex，但实际应用中并不建议直接操纵`mutex`对象，因为`unlock()`操作很难覆盖所有代码执行的路径，尤其是当出现异常的情况，可能会导致mutex无法被`unlock()`。常见的做法是使用上面曾介绍过的RAII技术，将`mutex`与某个对象的声明周期进行绑定，而C++ 11恰好提供了这样一个类`std::lock_guard`：

```cpp
std::list<int> some_list;
std::mutex some_mutex;

void write(int new_values){
    std::lock_guard<std::mutex> guard(some_mutex);
    some_list.push_back(new_values);
}
bool find(int value_to_find){
    std::lock_guard<std::mutex> guard(some_mutex);
    return (std::find(some_list.begin(), some_list.end(), value_to_find) != some_list.end());
}
```
上面是一个简单的使用mutex的例子，当`guard`对象析构时，会自动滴啊用`some_mutex.unlock()`来说释放mutex。然后在实际应用中，我们不太会定义全局的mutex对象，而是将其定义在某个类中和某个data关联：

```cpp
class some_data{
	int value;
};
class data_wrapper(){
private:
	some_data data;
	std::mutex m;
public:
	void process_data(){
		std::lock_guard<std::mutex> guard(m);
		do_something_with_data(){
			//
		}
	}
}
```
上面我们将`mutex`和要保护的`data`关联起来，`lock_guard`之锁住与该`data`相关的`mutex`。但是这里要注意`do_something_with_data()`这个方法，在这个方法中要保证对`data`的直接操作，如果将`data`通过某种方式传到该函数以外，而在该函数外部来操作`data`，则`data`的线程安全将无法保证，显然这不是一个好的编程习惯，因此要避免。

由于绝大多数的STL数据结构都不是线程安全的，因此使用`mutex`并不能保证对某些数据结构接口操作的原子性，例如对stack的操作：

```cpp
stack<int> s;
if(!s.empty()){ //--1
	int value = s.top(); //--2
	s.pop(); //--3
	do_something(value);
}
```

上述代码中，如果`s`被线程共享，则在`1,2,3`这个三个位置均有可能发生`race condition`，进而带来数据不一致。例如，假设当前`stack`只有一个数据，当线程1判断了`empty() == false`后，线程2执行了`s.pop()`，此时线程1继续执行`s.top()`则会发生crash。另一个比较有趣的`race condition`的可能是这样的：

```
        thread #1               thread #2
-------------------------|-----------------------
if(!empty()){            |
                         | if(!empty()){
    int value = s.top(); |
                         |    int value = s.top();
    s.pop();             |  
                         |  s.pop();
    do_something(value); | 
                         |    do_something(value);
}                        |
                         | }
--------------------------------------------------
```

如果`stack`中有两个以上的元素，上面两个线程均不会crash，只不过在逻辑上会有些问题，它们将读到相同的元素，并且`stack`被`pop`了两次。

`mutex`对接口操作失效的原因在于其粒度太大了，`mutex`无法真正lock到对内部data的操作上。解决这个问题需要重新设计一个线程安全的数据结构，篇幅原因，这里不做过多介绍，在后面几篇文章中将对这个问题做更详细的论述

### Deadlock

 上面使用mutex的场景是多个线程竞争**同一个**公共资源，而死锁则是多个线程同时在等待对方释放资源从而进入无休止的等待状态。死锁发生的条件通常是一个线程需要同时操作两份或者多份公共资源，每份公共资源都有一个mutex，当该线程（thread #1）已经获取了某个资源(B)的mutex后，试图获取资源(A)的mutex时，发现该mutex已经被另一个线程(thread #2)占据，而另一个线程(thread #2)则是相同的逻辑，它获取了该资源(A)的mutex，同时等待另一个线程(thread #1)释放资源(B)的mutex，从而两个线程进入死锁状态。
 
 产生上面问题的原因是每个线程试图同时获取两个mutex，C++ 11中提供了一种可以同时获取两个或多个mutex的方法，使用`std::lock`:

```cpp
class some_class;
void swap(some_class& v1, some_class& v2);
class X{
private: 
    some_class data;
    std::mutex m;
public:
    friend void swap(X& lhs, X& rhs){
        if(&lhs == &rhs){
            return;
        }
        std::lock(lhs.m, rhs.m);
        std::lock_guard<std::mutex> lock_a(lhs.m, std::adopt_lock);
        std::lock_guard<std::mutex> lock_b(rhs.m, std::adopt_lock);
        swap(lhs.data, rhs.data);
    }
};
```

`std::lock`可以保证同时获取多个共享数据的mutex，从而一次性锁住两块内存区域。如果`std::lock()`在获取其中一个mutex时发生了异常，则会自动释放已经获取的mutex。而`std::adopt_lock`表示`lhs.m`已经被`std::lock`所占用，则`std::lock_guard`在构造`lock_a`时无需对`lhs.m`再次lock，只需获得其所有权(引用)即可。

### 避免死锁的一些办法

死锁的产生不一定局限于对某个mutex的获取与等待，即使在无锁的情况下，两个线程互相`join()`也会进入死锁状态。要避免死锁的根本方法是避免两个线程之间相互等待的情况发生。下面给出几种常用的避免死锁的方式

1. 避免锁的嵌套

	当某个线程已经获得某种锁时，不要尝试继续获取其它的锁。如果需要同时获取多个锁的控制，使用`std::lock()`做批量处理。

2. 尽量减少锁的粒度

	当某个线程获得lock之后，其后面的代码应尽量避免调用其它非操作共享数据的API，尽量减少非共享数据外的控制逻辑。

3. 按照某种固定顺序获得lock

	将需要同时lock的场景重构为可以分几步执行的情况，每一步lock一次，做完这一步则unlock，保证操作的有序进行

### 使用`std::unique_lock`

细心观察，可发现前面提到的`guard_lock`存在一定的局限性，即根据RAII原则，只有当`guard_lock`对象释放时，mutex才会被unlock。这个特性依赖函数执行完成来析构`guard_lock`对象。但是如果希望在函数执行完之前再做一些其它操作，那么这部分操作将不可避免的被包含在锁中，破坏锁的粒度。因此我们需要一个更灵活的lock。

```cpp
class some_class;
void swap(some_class& v1, some_class& v2);
class X{
private: 
    some_class data;
    std::mutex m;
public:
    friend void swap(X& lhs, X& rhs){
        if(&lhs == &rhs){
            return;
        }
        std::unique_lock<std::mutex> lock_a(lhs.m, std::defer_lock);
        std::unique_lock<std::mutex> lock_b(rhs.m, std::defer_lock);
		std::lock(lock_a, lock_b);
        swap(lhs.data, rhs.data);
    }
};
```
上述代码的功能和使用`std::lock_guard`相同。分析如下：

1. 当构造`std::unique_lock<std::mutex>`对象时，第二个参数为`std::defer_lock`表示在构造`lock_a`或`lock_b`时，不持有mutex。
2. 当`std::lock(lock_a, lock_b);`执行时，`lock_a`和`lock_b`才会分别执行`lhs.m.lock()`和`rhs.m.lock()`。可以看出`std::unique_lock`的`lock，unlock`方法不过是透传给mutex来完成，而其内部维护了一个关于mutex的状态，该状态用于表明其是否真正持有mutex对象
3. 这个状态可以通过`owns_lock()`来查看，如果其持有mutex对象，则在其析构函数中要调用mutex的`unlock()`方法，如果其不持有，则不需要调。

从某种意义上讲，`std::unique_lock`是对mutex的一种封装，因此相对`std::lock_guard`它有一定的开销，但它却更灵活，可以defer lock和转移对mutex的控制权，这两个特性在某些场合非常有用。

> 对大部分场景，如果`std::lock_guard`够用，则不需使用`std::unique_lock`

由于`std::unique_lock`可以调用`unlock`，因此它也可以较好的控制锁的粒度

```cpp
void get_and_process_data(){
	std::unique_lock<std::mutex> my_lock(data_mutex);
	Data_Object data = get_some_data();
	my_lock.lock();
	Result_Object result = process_some_data(data);
	my_lock.unlock();
	write_result(data,result);
}
```

### 对象初始化的线程安全

如果某个指针需要进行惰性初始化：

```cpp
std::shared_ptr<some_resource> resource_ptr;
void foo(){
	if(!resource_ptr){
		resource_ptr.reset(new some_resource);
	}
	resource_ptr->do_something();
}
```
如果两个线程同时访问该段代码，很有可能出现的情况是一个线程在执行完`resource_ptr.reset(new some_resource);`时，另一个线程刚好走到`resource_ptr->do_something();`因此，`resource_ptr->do_something();`可能会被执行两遍。除了上述可能，还有一系列其它的可能情况，这里不一一列举。

解决这个问题，C++ 11提供了`std::once_flag`和`std::call_once`两个函数：

```cpp
std::shared_ptr<some_resource> resource_ptr;
std::once_flag resource_flag;
void foo(){
	std::call_once(resource_flag, []{
		resource_ptr.reset(new some_resource);
	});
}
```

`std::call_once`的调用方式和`std::thread`的构造函数以及`std::bind`函数相同，即第二个参数可以是一个函数指针，functor或者lambda表达式，后面参数为该函数所用到的参数。`std::once_flag`对象也可被定义为某个类的成员变量使用

### Reader-Writed Mutex

考虑下面一个场景,一个DNS表中存放了域名和IP地址的对应关系，虽然实际情况是DNS地址不会经常更新，但如果有新的条目加入进来，则需要对其进行修改。实际应用中，对DNS表的查找访问往往是多个线程同时进行的，因此当这个表需要被修改时，需要一条线程来完成这个工作，为了保证DNS表的线程安全，我们希望“写线程”在修改时，其它读线程不会读到错误数据。

这种情况下是不能使用mutex的，原因是如果读写都加锁，由于写操作频率很低，导致则大量的读操会被无意义的锁block，性能会大大降低；如果只给写操作加锁，读操作不加锁，则会发生`race condition`的情况，比如有可能读到写了一半的不完整数据。因此我们需要一种新的锁结构来解决上面问题，我们希望这总锁可以让读线程并发访问不受影响，只当写操作时block读线程，等写操作完成后，读线程继续执行。这种锁叫做“读写锁”，它用来处理写操作不频繁，读操作频繁的场景。C++ 17标准库新增了`shared_mutex`对应这类读写锁的实现。

1. 对于写操作，可以使用`std::lock_gurad<std::shared_mutex>`
2. 对于读操作，可以使用`std::shared_lock<std::shared_mutex>`

```cpp
class dns_entry{ 
	//some code 
};
class dns_cache{
    std::map<std::string, dns_entry> entries;
    mutable std::shared_mutex entry_mutex;
public:
    dns_entry find_entry(std::string const& domain ) const{
        std::shared_lock<std::shared_mutex> lk(entry_mutex);
        auto itor = entries.find(domain);
        if(itor != entries.end()){
            return itor->second;
        }else{
            return dns_entry();
        }
    }
    void update_or_add_entry(std::string const& domain, dns_entry const& dns_details){
        std::lock_guard<std::shared_mutex> lg(entry_mutex);
        entries[domain] = dns_details;
    }
};
```
在上面的场景中，如果某个线程想进行写操作，则需要等所有读操作的线程执行完，并释放`shared_lock`之后才能执行。同理，当写线程执行时，读操作线程被block，无法获取`shared_lock`，写线程也需要等待当前的写线程完成后，才能重新获取`shared_mutex`。

### Recursive Mutex

对于同一个mutex对象，如果在一个线程内被连续`lock()`多次，则会导致`undefined behavior`，但是如果某些情况下，一个线程需要对同一个mutex对象进行多次`lock()`操作，这时可以使用C++提供的另一种recursive mutex，`std::recursive_mutex`它的用法和普通的mutex相同，不同的是它的对象可以在同一个线程内可以被`lock()`多次。但有一点要注意的是，当另一个线程试图获取对`std::recursive_mutex`对象的控制权时，该对象必须要释放它分配出去的所有锁，即如果它之前进行了三次`unlock()`操作，那么需要配对的进行三次`unlock()`操作。同样的，并不建议直接使用`std::recursive_mutex`，还是需要使用它的封装类`std::lock_guard`和`std::unique_lock`。

对于绝大多数情况来说，如果代码中遇到需要使用recursive mutex的场景，那说明该段代码需要重构。一种使用recursive mutex的典型场景是某个类对象被多个线程共享，对此在类的每个成员函数中都加了锁保护内部数据的读写，如果在每个成员函数中都只操作该类的内部数据，则没有任何问题，但是某些情况下在某个成员函数中部会访问另一个成员函数，这时候就出现类中的同一个mutex对象被`lock`两次的情况，如果是普通的mutex则会出问题，因此一个简单粗暴的解决办法就是将普通的mutex替换为recursive mutex。但是这种做法并不推荐，并且这种调用也是一个很糟糕的设计，遇到这种情况，可以将调用另一个成员函数的函数拆成两个粒度更小的函数，保证每个函数都只做一件事情。




## Resources

- [《C++ Concurrency in Action》](https://www.manning.com/books/c-plus-plus-concurrency-in-action?)
- [《C++ Primer 3rd edition》]()
- [C++ 11 Concurrency](https://www.classes.cs.uchicago.edu/archive/2013/spring/12300-1/labs/lab6/)
- [Modern C++ Concurrency in Practice: Get the most out of any machine](https://www.educative.io)
