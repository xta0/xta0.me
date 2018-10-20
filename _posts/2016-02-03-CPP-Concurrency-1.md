---
layout: post
list_title: 谈谈 C++ 中的多线程 | Concurrency in Modern C++  | 线程管理 | Managing Threads
title: 线程管理 
categories: [C++, Concurrency, Thread]
---

### Lunching Thread

C++ 11中对线程的管理集中在`std::thread`这个类中，创建线程的方式包括：

1. 使用回调函数
2. 使用函数对象
3. 使用Lambda表达式

```cpp
void funcptr(std::string name){
    cout << name<<endl;
}
struct Functor{
    void operator()(string name){
        cout<<name<<endl;
    }
};
void runCode(){
	//使用function pointer
	std::thread t1(func1,"abc");
	t1.detach();

	//使用Functor
	Functor f;
	std::thread t2(f,"Functor thread is running");
	t2.detach();

	//使用lambda
	std::string p = "lambda thread is running";
	std::thread t3([p]{
		cout<<p<<endl;
	});
	t3.join();
}
```

`std::thread`是C++11引入的用来管理多线程的新类，是对UNIX C中`pthread_t`结构体的封装，构造时调用`pthread_create`传入`pthread_t`和回调函数指针

```cpp
typedef pthread_t __libcpp_thread_t;
class _LIBCPP_TYPE_VIS thread{
    __libcpp_thread_t __t_; //pthread_t
	...
}
thread::thread(_Fp&& __f, _Args&&... __args){
	...
	int __ec = __libcpp_thread_create(&__t_, &__thread_proxy<_Gp>, __p.get());
	 if (__ec == 0)
        __p.release();
   	 else
        __throw_system_error(__ec, "thread constructor failed");
}
int __libcpp_thread_create(__libcpp_thread_t *__t, void *(*__func)(void *),
                           void *__arg){
  return pthread_create(__t, 0, __func, __arg);
}
```
`std::thread`的构造函数中前两个参数均为右值引用，第二个参数将传入的lambda表达式或者functor通过`__thread_proxy<_Gp>`转化成C函数指针（`void *(*__func)(void *)`，这个问题可参考[之前对C++11中 `move`语义的介绍](https://xta0.me/2009/08/30/CPP-Basics-3.html)。`std::thread`对象在创建后，如果不做其它操作，线程立刻执行，这里称这个线程为`worker_thread`，称发起`worker_thread`的线程为`launch_thread`。

如果`std::thread`对像在被销毁前未执行`join()`或`detach()`操作，则在其析构函数中会调用`std::terminate`造成系统崩溃。因此需要确保所有创建的`std::thread`对象都能被正常释放，在《C++ Concurrency in Action》中，提到了一种方法：

```cpp
class thread_guard{
	std::thread &t;
public:
	explicit thread_guard(std::thread& t_):t(t_){}
	~thread_guard(){
		if(t.joinable()){
			t.join();
		}
	}
	thread_guard(thread_guard const& ) = delete;
	thread_guard& operator=(thread_guard const& ) = delete;
};
void runCode(){
	...
	std::thread t(f);
	thread_guard g(t); //g在t之前释放，保证join的调用
	do_something_in_current_thread();
}
```
由于`thread_guard`对象总是在`std::thread`对象之前析构，因此可以在`t`析构之前调用`join`函数，保证`t`可以安全释放。

 > 这种方式是所谓的**RAII(Resource Acquisition Is  Initialization)**，即通过构造某个对象来获得某个资源的控制，在该对象析构时，释放被控制的资源。也就是将某资源和某对象的生命周期做绑定，在C++中这是一种很常用的设计方式，背后的原因是C++允许栈对象的创建和析构，后面在讨论mutex时还会继续用到这种技术

### Join & Detach

`join()`是`launch_thread`和`worker_thread`的一个线程同步点，`launch_thread`会在调用`join()`后等待`worker_thread`执行完成后继续执行

```cpp
std::string p = "lambda";
//using lambda expression as a callback function
std::thread td([p]{cout<<p<<" thread is running"<<endl;});
cout<<"lanched thread is running"<<endl;
td.join();
cout<<"lanched thread is running"<<endl;
td.joinable(); //return false
```

1. 如果`td`在`main thread`执行`td.join()`之前完成，则`td.join()`直接返回，否则`launch_thread`会暂停，等待`td`执行完成
2. 如果不调用`td.join()`或`td.detach()`，在`td`对象销毁时，在`std::thread`的析构函数中，如果则系统会发出`std::terminate`的错误
3. `td`在调用`join`后，`joinable`转态变为`false`，此时`td`可被安全释放
4. 确保`join()`只被调用一次


如果使用`td.detach()`则`workder_thread`在创建后立刻和`launch_thread`分离，`launch_thread`不会等待`workder thread`执行完成。即两条线程没有同步点，各自独立执行

```cpp
void runCode()
{
    cout << "lanched thread is running" << endl;
    std::string p = "lambda";
    std::thread td([p] { 
        std::this_thread::sleep_for(std::chrono::milliseconds(5000));
        cout << p << " thread is running" << endl; 
    });
    td.detach();
    cout << "lanched thread is ending" << endl;
}
```
上述代码中令`workder thread`暂停5s，则`launch_thread`继续执行，不会等待`worker_thread`执行完，如果将`detach()`改为`join()`则`launch_thread`会阻塞等待

```
//detach
lanched thread is running
lanched thread is ending

//join
lanched thread is running
lambda thread is running
lanched thread is ending
```

### 向线程传递参数

向`std::thread`构造函数传递参数的规则为：

1. 第一个参数为函数指针，可以是functor或者lambda表达式，在第一节中已经介绍
2. 后面参数为该函数指针需要用到的参数

观察前面的`std::thread`的构造函数可知，传递的参数均是拷贝到线程自己stack中，但是有某些场景，需要修改`lauch_thread`所在线程的局部变量，这是需要将该变量的引用传递给`worker_thread`。例如下面的例子中需要在`worker_thread`中修改`data`变量

```cpp
void updateData(widget_data& data);
void oops(){
	widget_data data;
	std::thread t(updateData, data); //这里传过去的是data的copy
	t.join();
	process_widget_data(data);
}
```
[参考之前文章中对`bind`函数的介绍]()，可知这里只需要一个很小的改动，使用`std::ref(x)`，即可把`data`从传拷贝变成传引用：

```cpp
std::thread t(updateData, std::ref(data))
```
> 如果传引用或者指针要特别注意变量的生命周期，如果该变量的内存在线程还未结束时被释放则会引`undefined behavior`

为了进一步加深对`std::thread`构造函数的理解，继续参考`bind`函数不难发现，`std::thread`的构造函数和`bind`的传参机制是相同的，这意味者只要第一个函数时一个函数指针，后面是该函数的参数即可，因此可以不局限于使用第一小节介绍的三种构建线程的方式，比如：

```cpp
class X{
public:
    void do_some_work(){
        cout<<"do_some_work"<<endl;
    };
};
X x;
std::thread t(&X::do_some_work, &x);
t.detach();
```
上述代码中，`X::do_some_work`方法的第一个参数为`this`指针，因此可将`x`取地址后传入，可达到相同效果。


## Resources

- [《C++ Concurrency in Action》](https://www.manning.com/books/c-plus-plus-concurrency-in-action?)
- [《C++ Primer 3rd edition》]()
- [C++ 11 Concurrency](https://www.classes.cs.uchicago.edu/archive/2013/spring/12300-1/labs/lab6/)
- [Modern C++ Concurrency in Practice: Get the most out of any machine](https://www.educative.io)