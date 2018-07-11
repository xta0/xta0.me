---
layout: post
list_title: Concurrency in C++ Part 1
title: C++ 11中的多线程（一） | Concurrency in C++ Part 1
---

## 线程管理

### Lunching Thread

C++ 11中对线程的管理集中在`std::thread`这个类中，创建线程的方式包括：

1. 使用回调函数
2. 使用functor
3. 使用Lambda表达式

```cpp
struct Functor{
    void operator()(string name){
        cout<<name<<endl;
    }
};
void runCode(){
	//使用Functor
	Functor f;
	std::thread th1(f,"Functor thread is running");

	//使用lambda
	std::string p = "lambda thread is running";
	std::thread th2([p]{
		cout<<p<<endl;
	});
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
`std::thread`的构造函数中前两个参数均为右值引用，第二个参数将传入的lambda表达式或者functor通过`__thread_proxy<_Gp>`转化成C函数指针（`void *(*__func)(void *)`，这个问题可参考[之前对C++11中 `move`语义的介绍]()。`std::thread`对象在创建后，如果不做其它操作，线程立刻执行，这里称这个线程为`worker_thread`，称发起`worker_thread`的线程为`launch_thread`。

如果`std::thread`对像在被销毁前未执行`join()`或`detach()`操作，则在其析构函数中会调用`std::terminate`造成系统崩溃。因此，一个重要的问题是要确保所有创建的`std::thread`对象都能被正常释放，在《C++ Concurrency in Action》中，提到了一种方法

```cpp
class thread_guard{
	std::thread &t;
	explicit thread_guard(std::thread& t_):t(t_){}
	~thread_guard{
		if(t.joinable()){
			t.join();
		}
	}
	thread_guard(thread_guard const& ) = delete;
	thread_guard& operator=(thread_guard const& ) = delet;
};
void runCode(){
	...
	std::thread t(f);
	thread_guard g(t); //g在t之前释放，保证join的调用
	do_something_in_current_thread();
}
```

### Join & Detach

`join()`是`launch_thread`和`worker_thread`一个线程同步点，`launch_thread`会在调用`join()`后等待`worker_thread`执行完成后继续执行

```cpp
std::string p = "lambda";
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


如果使用`td.detach()`则`launch_thread`不会等待`workder thread`，即两条线程没有同步点，各自独立执行

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

### 参数传递

向`std::thread`构造函数传递参数的规则为：

1. 第一个参数为函数指针，可以是functor或者lambda表达式
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
参考之前文章中对`bind`函数的介绍，可知这里只需要一个很小的改动，即可把`data`从传拷贝变成传引用，答案是使用`std::ref(x)`。

```cpp
std::thread t(updateData, std::ref(data))
```
如果传引用或者指针要特别注意变量的生命周期，如果该变量的内存在线程还未结束时被释放则会引起严重的问题


## 线程间共享数据

如果有某种数据需要被两个线程共享，如果该数据是`read-only`则没有问题，如果是可读写的，则会有几率出现`race condition`的情况，即一个线程在进行读操作时，另一个线程在进行写操作，这时对于读操作的线程将有几率读到不完整的数据。这时一个很经典的问题，解决这个问题的办法有很多种，比如设计无锁的数据结构，

## Resources

- [C++ 11 Concurrency](https://www.classes.cs.uchicago.edu/archive/2013/spring/12300-1/labs/lab6/)



## C++ 11 Features 



### auto keyword

C++11引入`auto`关键字用于省略变量类型，编译器可以根据变量的值自动推断其类型，比如

```cpp
auto i = 100; //i is int
auto p = new A();
auto add(T1 x, T2 y) -> decltype(x+y){ //add is a function
	return x+y;
}
```

但是对于函数的返回值，如果声明为`auto`，则需要指定其类型

```cpp
auto test()->int{
	return 100;
}
cout<<test()<<endl;
```
我们可以使用上面的写法来定义函数，上面这个函数定义和传统的函数定义相比貌似



### 统一初始化方法

```cpp

int arr[3]{1,2,3};
vector<int> iv{1,2,3};
map<int,string> mp{ {1,"a"}, {2,"b"} };
string str{"hello!"};
int *p = new int[20]{1,2,3};

struct A{
	int i,j;
	A(int m, int n):i(m),j(n){}
};

A func(int m, int n){
	return A{m,n};
}

int main(){
	A* pa = new A{3,6};
}
```

### 成员变量默认初始值

```cpp
class B{
	public:
		int m=1234;
		int n;
};
```

### shared_ptr

- 头文件:<memory>，类模板
- 通过shared_ptr的构造函数，可以让shared_ptr对象托管一个new运算符返回的指针,写法如下：

```cpp
shared_ptr<T> ptr(new T);
```

此后ptr就可以像T* 类型的指针一样使用， *ptr就是用new分配的那个对象，而且不必操心内存释放的事情。

- 多个`shared_ptr`对象可以同时托管一个指针，系统会维护一个托管计数。当无`shared_ptr`托管该指针时，`delete`该指针

- `shared_ptr`对象不能托管指向动态分配的数组的指针，否则程序会出错


```cpp
#include <memory>
struct A{
	int n;
	A(int v = 0) : n(v){}
	~A(){}
};

shared_ptr<A> sp1(new A(10));
A* p = sp1.get(); //获取原始指针

shared_ptr<A> sp2(sp1);
shared_ptr<A> sp3 = sp1;

//此时A(10)对象被三个指针托管
sp1.reset(); //sp1放弃对A的托管

if(!sp1){
	cout << sp1 is null; //sp1放弃对A的托管后，自己也为null
}

A *q = new A(11);
sp1.reset(q); //sp1托管q

shared_ptr<A> sp4(sp1) //sp4托管q
shared_ptr<A> sp5; 
sp5.reset(q); //报错
sp1.reset(); //sp1放弃托管q
sp4.reset(); //sp4放弃托管q
```
### 空指针nullptr

```cpp
int *p1 = NULL;
int *p2 = nullptr;
shared_ptr<int> p3 = nullptr;

//p1 == p2 //yes
//p3 == nullptr //yes
//p3 == p2 //error!
//p3 == NULL //yes

bool b = nullptr; //b = false
int i = nullptr; //error
```

### Range For-loop

```cpp
struct A { int n; A(int i):n(i){}};
int main(){
	int arr[] = {1,2,3,4};
	for(int/auto &e : arr){
		e *= 10; 
	}
	for(int e : arr){
		cout << e;
	}
}
```

### 右值引用和move语义

右值：一般来说，不能取地址的表达式就是右值，能取地址的，就是左值

```cpp
class A{};
A &r = A(); //error, A()是无名变量，是右值
A &&r = A(); //ok, r是右值引用
```

C++ 11之前引用都是左值引用，右值引用的主要目的是提高程序的运行效率，减少需要进行深拷贝对象的拷贝次数

- C++11之前写法:

```cpp
struct String{
	char* p;
	
	//默认构造函数
	String():p(new char[1]){ p[0] = '\0'};
	
	//赋值构造函数
	String(const char* s){
		p = new char[strlen(s)+1];
		strcpy(p,s);
	}
	
	//拷贝（复制）构造函数
	String(const String &s){
		p = new char[strlen(s.p)+1]	;
		strcpy(p,s.p);
	}
	
	//重载赋值
	String& operator=(const String &str)
	{
		if(p != str.p)
		{
			delete []p;
			p = new char[strlen(str.p)+1];
			strcpy(p,str.p);
		}
		return *this;
	}
};

```

- 新写法：move constructor

```cpp

//move constructor
String(String && s):p(s.p){
	s.p = new char[1];
	s.p[0] = 0;
	//将参数s.p指向了另一片存储空间

}//此时当前对象的p直接指向了s.p，并不会触发拷贝的操作

//move assigment
String & operator=(String && s){
	if(p != s.p){
		delete[] p;
		p = s.p; //直接赋值
		s.p = new char[1];
		s.p[0] = 0; //将s.p指向另一片内存
	}
	return * this;
}

template<class T>
void MoveSwap(T& a, T& b){
	T tmp(move(a)); //std::move(a)为右值，会调用move constructor，std::move(x)将一个左值变成右值
	a = move(b); //std::move为右值,因此会调用move assignment
	b = move(tmp); //std::move为右值,因此会调用move assignment
}

int main()
{
	String &r = String("this"); //error
	
	String s;
	s = String("ok"); //String("ok")是右值
	String && r = String("this");
	String s1 = "hello", s2 = "world";
	MoveSwap(s1,s2);
	return 0;
}

```

### unorderd_map

```cpp
unorded_map<string, int> turingWinner;
turningWinner.insert(make_pair("Scott",1976));
//查询
unorded_map<string,int>::iterator p = turningWinner.find(name);
if(p != turingWinnder.end()){

}else{
	//
}
```

### Lambda表达式

- 只用一次的函数对象，能否不要专门为其编写一个类
- 只调用一次的简单函数，能否在调用时才定义
- 形式：

```
[外部变量访问方式说明符](参数表) -> 返回值类型
{
	//函数体
}

[=] 以传值的形式使用外部所有变量，值不可以被修改
[] 不适用任何外部变量
[&] 以引用的形式使用所有外部变量，引用可以修改
[x,&y] x以传值形式引入， y以引用形式引入
[=,&x,&y] x,y以引用形式使用，其余变量以传值形式引入
[&,x,y] x,y以传值形式引入，其余变量以引用形式使用
"-> 返回值类型"，可以忽略，编译器可以自动推断
```

```cpp
int x=100, y = 200, z = 300;
cout << [](double a, double b){return a+b}(1.2,2.5)<<endl;

auto ff = [=,&y, &z](int n){
	cout <<x<<endl;
	y ++;
	z++ ;
	return n*n;
};

ff(15);

//操作集合
vector<int> a{1,2,3,4};
int total = 0;
for_each(a.begin(), a.end(), [&](int &x){total+=x; x*=2;});
for_each(a.begin(), a.end(), [](int x){cout << x <<endl} ); 

//实现递归
function<int(int)> fib = [&fib](int n){
	return n<2?1:fib(n-1) + fib(n-2);
}
//function<int(int)>表示输入为int，返回值为int的函数
```
