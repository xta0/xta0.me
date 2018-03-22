---
layout: post
title: C++ 11 New Features Overview
---

## C++ 11新特性

### Features

- Encompasses features of C++11
- Move semantics
- Smart pointers
- Automatic type inference
- Threading
- Lambda functions

### ISO Standard

- Responsible for adding new featuers to C++
- Has members from all over the world
- Some are representatives of their companies (Microsoft, Google, IBM, etc)
- Published first standartd in 1998,followed by a minor revision in 2003
- Major change in 2011, lots of new features
- 2014 added a minor change, mostly enhancements

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
### auto关键字

用于定义变量，编译器可以自动判断变量类型,变量必须初始化

```cpp
auto i = 100;
auto p = new A();
auto k = 343;
auto add(T1 x, T2 y) -> decltype(x+y){
	return x+y;
}

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

if(p != turingWinnder.end())
{
	//
}
else{

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

- 例子:

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