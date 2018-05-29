---
layout: post
title: C++ Part 4| STL Containers | STL容器
categories: PL
tag: C++
mathml: true
---

## STL容器概述

- 容器：可容纳各种类型的通用数据结构，是<mark>类模板</mark>。STL容器按照存放的数据结构类型，可分为三种，分别为顺序容器，关联容器，和容器适配器。
- 迭代器：可以用于依次存取容器中的元素，类似指针
- 算法：用来操作容器中的元素的函数模板
	- `sort()`来对一个vector中的数据进行排序
	- `find()`来搜索一个list中的对象

算法本身与他们操作的数据类型无关，因此可以用在简单的数组到高级的数据结构中使用。

### 迭代器

STL算法以及STL容器中的操作大部分需要使用迭代器来完成，以访问容器中的元素为例，少数顺序容器重载了`[]`可支持元素的随机访问，其余容器需要先找到该元素的迭代器，再通过迭代器来访问元素内容。这个设计和其它编程语言有些不同，因此对于初学C++的人或者从别的动态语言转过来的开发者来说，需要先适应并熟悉迭代器。对于迭代器，下面几点需要重点理解。

- 迭代器范围
	- `[begin, end)`，`end`指向容器最后一个元素的下一个位置
- 容器头尾迭代器
	- `begin/end`: 返回指向容器中第一个/最后一个元素的迭代器
	- `rbegin/rend`:返回指向容器中最后一个/第一个元素的迭代器
	- `cbegin/rend` 返回一个const类型的迭代器

```cpp
list<string>::iterator it5 = a.begin();//显式指定类型
list<string>::const_iterator it6 = a.begin(); //是普通迭代器还是const迭代器依赖a的类型
auto it7 = a.begin(); //根据a的类型来返回是否是const迭代器
auto it8 = a.cbegin(); //不管a的类型，一定返回一个const迭代器
```

### 顺序容器
	
- 元素是非排序的，元素的插入位置与元素的值无关。
- 标准库提供下列容器			
	- `vector`
		- 头文件`<vector>`
		- 动态数组。元素在内存中连续存放。
		- 支持随机访问随机存取任何元素都能在常数时间完成。
		- 在尾端增删元素具有较佳的性能（大部分情况下是常数时间）
	- `array`
		- 头文件`<array>`，<mark>C++ 11</mark>新增
		- 支持随机访问
		- 固定大小数组，不能添加或者删除元素
	- `deque`
		-  头文件`<deque>`
		- 双向队列，元素在内存内连续存放
		- 支持随机访问你，随机存取任何元素都能在常数时间完成（但次于vector）。
		- 在两端增删元素具有较佳的性能
	- `list`
		- 头文件`<list>`
		- 双向链表，元素在内存中不连续存放。
		- 在任何位置增删元素都能在常数时间完成
		- 不支持下标随机存取，支持双向顺序访问
	- `forward_list`
		- 头文件`<list>`，<mark>C++ 11</mark>新增
		- 单向链表，元素在内存中不连续存放。
		- 在任何位置增删元素都能在常数时间完成
		- 不支持下标随机存取，支持单向顺序访问
	- `string`
		- 同`vector`，字符容器

- 顺序容器的初始化方式

| -- | -- |
| `C c` | 默认构造函数，如果`C`是一个`array`，则`c`中元素按默认方式初始化；否则`c`为空 |
| `C c1(c2)` or `C c1=c2`| 拷贝构造，`c1`初始化为`c2`的拷贝 |
| `C c{a,b,c..}` or `C c={a,b,c...}`| 使用初始化列表，`c`初始化为initialize list中元素的<mark>拷贝</mark>。初始化列表中元素的个数必须小于等于`array`的大小，遗漏的元素进行默认初始化|
| `C c(start, end)`| 使用迭代器初始化，`c`初始化为迭代器`start`和`end`指定范围中的元素的拷贝|
| `C c(n)`| `c`包含`n`个元素，这些元素进行了值初始化（默认值），string不适用|
| `C c(n,t)`| `c`包含`n`个值为`t`的元素|


- 顺序容器API
	- `front`:返回容器中第一个元素的引用
	- `end`:返回容器中最后一个元素的引用
	- `push_back`:在容器末尾增加新元素
	- `pop_back`:删除容器末尾的元素
	- `erase`:删除迭代器指向的元素
	- `clear`: 从容器中删除所有元素

### 关联容器

- 元素是排序的，插入任何元素，都按相应的排序规则来确定其位置
- 在查找时具有很好的性能
- 通常以平衡二叉树方式实现，插入和检索的时间都是`O(log(N))`
- 常用的关联容器：
	- `set/multiset`
		- 头文件`<set>`，
		- 集合,`set`不允许有相同的元素，`multiset`中允许存在相同的元素 
	- `map/multimap` 
		- 头文件`<map>`，
		- `map`与`set`不同在于map中的元素有且皆有两个成员变量，一个名为`first`，一个名为`second`
		- `map`根据`first`值进行大小排序，并可以快速的根据first来检索。
		- `map`同`multimap`的区别在于是否允许相同的`first`值。
- 关联容器API
	- `find`：查找等于某个值的元素(x小于y和y小于x同时不成立即为相等),返回一个迭代器类型对象
	- `lower_bound`：查找某个下界
	- `upper_bound`：查找某个上界
	- `equal_range`：同时查找上界和下界

### 容器适配器：

- `stack`
	- 头文件`<stack>`
	- 栈，是项的有限序列，并满足序列中被删除
	- 检索和修改的项只能是最近插入序列的项（栈顶的项）
- `queue`
	- 头文件`<queue>`
	- 队列，插入只可以在尾部进行，删除
	- 检索和修改只允许从头部进行，先进先出
- `priority_queue`
	- 头文件`<queue>`
	- 优先级队列

## 顺序容器

### Array

- 静态数组
	
<mark>C++ 11</mark>新增的静态数组表示。

```cpp
//类型为42个int型数组
array<int,42>;
```

### List

- 双向链表
- 迭代器不支持完全随机访问
	- 不能用标准库中的`sort`函数排序
- 排序使用自己的`sort`成员函数
	- `void sort()` 将`list`中的元素按照"<"规定的比较方法升序排列
	- 实现比较函数
	
	```cpp
	template<class Compare>
	void sort(Compare op); //list中的排序规则由op(x,y)的返回值指定

	template <class T>
	class Comp{
		public:
			bool operator()(const T& c1, const T& c2){
				return c1<c2;
			}
	};

	list<int> lst(10);
	lst.sort(Comp<int>());
	```
- 只能使用双向迭代器
	- 迭代器不支持大于/小于的比较运算符，不支持`[]`运算符和随机移动

```cpp
list<int> numbers;
numbers.push_back(1);
numbers.push_back(2);
numbers.push_back(3);
numbers.push_front(4);
auto itor = numbers.begin();
itor++;
itor = numbers.insert(itor,100); //在第二个位置插入100, 返回第三个位置的迭代器
itor = numbers.erase(itor); //删除第三个元素，返回第四个位置的迭代器
//遍历
for(;itor!=numbers.end();++itor){
	cout<<*itor<<endl;
	if(*itor == 2){
		numbers.insert(itor,1234);//在2之前插入1234
	}
}

```

### deque

- 双向队列：double ended queue
- 包含头文件`#include<deque>`
- `deque`可以`push_front`和`pop_front`

## 关联容器

### pair

- 定义:

```cpp
template<class T1, class T2>
struct pair{
	typedef T1 first_type;
	typedef T2 second_type;
	T1 first;
	T2 second;
	pair():first(),second(){}
	pair(const T1& _a, const T2& _b):first(_a),second(_b)	{}
	template<class U1, class U2>
	pair(const pair<_U1,_U2>& _p):first(_p.first),second:(_p.second){
	
	};
}
```
`map`,`multimap`容器里存放着的都是pair模板类对象，且first从小到大排序，第三个构造函数用法实例：

```cpp
pair<int, int> p(pair<double,double>(5.5,4.6))
//p.first = 5,
//p.second=  4
```

### set & multiset

`set`和`multiset`表示数学中集合，`set`和`multiset`的定义如下：

```cpp
//set
tempate<class key, class pred = less<key>>
class set{...}

//multiset
template<class key, class Pred=less<key>,class A = allocator<key>>
class multiset{...}
```
和数学中的集合一样，在`set`中是不能存在重复元素的，另外我们发现`set`模板中第二个类型是函数模板，其默认值为`less<key>`，这说明元素在`set`内部的存储是有序的，`less<key>`这个函数模板定义如下：

```cpp
template<class T>
struct less:publi binary_function<T,T,bool>{
	bool operator()(const T& x, const T& y) const{
		//默认使用<比较大小
		return x<y;
	}
}

```
- set中不允许有重复元素,插入set中已有元素时，忽略

```cpp
int main(){

	std::set<int> ::iterator IT;
	int a[5] = {3,4,5,1,2};
	set<int> st(a,a+5);
	pair<IT,bool> result = st.insert(6); //返回值类型是pair
	if(result.second){ 
		//插入成功，则输出被插入的元素
	}
}
```

#### multiset

- 定义

```cpp

```
- `pred`类型的变量决定了`multiset`中的元素顺序是怎么定义的。`multiset`运行过程中，比较两个元素`x`，`y`大小的做法是通过`Pred`类型的变量，例如`op`，若表达式`op(x,y)`返回true则 x比y 小，`Pred`的缺省类型为`less<key>`，其中less模板为一个functor:


- 成员函数:
	- `iterator find(const T&val);`：在容器中查找值为val的元素，返回其迭代器。如果找不到，返回end()。
	- `iterator insert(const T& vale);`：将val插入到容器中并返回其迭代器
	- `void insert(iterator first, iterator last);`将区间[first,last)插入容器
	- `int count(const T& val);`统计多少个元素的值和val相等
	- `iterator lower_bound(const T& val);`查找一个最大位置it，使得[begin(),it)中所有元素都比val小。
	- `iterator upper_bound(const T& val);`查找一个最大位置it，使得[it,end())中所有元素都比val小。

```cpp
class A{};
int main(){
	
	std::multiset<A> a; //等价于multiset<A,less<A>> a;
	a.insert(A()); //error,由于A没有重载<无法比大小，因此insert后编译器无法知道插入的位置
}

//修改class A
class A
{
	private: 
		int n;
	
	public:
		A(int n_):n(n_){}
		
	friend bool operator<(const A& a1, const A& a2){
		return a1.n < a2.n;
	}	
	friend class Myless;
}

struct Myless{
	bool operator()(const A& a1, const A& a2){
		return (a1.n % 10)< (a2.n % 10);
	}
}

int main(){
	const int SIZE = 6;
	A a[SIZE] = {4,22,3,9,12};
	std::multiset<A> set;
	set.insert(a,a+SIZE);
	set.insert(22);
	set.count(22); //2
	
	//查找:
	std::multiset<A>::iterator pp = set.find(9);
	if(pp != set.end()){
		//说明找到
	}
}
```

### pair

### map/multimap

#### map

- 定义

```cpp
template<class key,class T, class Pred = less<key>,class A = allocator<T>>
class map{	
	//typedef pair<const key, T> value_type;
};
```	
- map**有序的**k-v集合，元素按照`key`**从小到大**排列，缺省情况下用`less<key>`即`<`定义
- map中相同的`key`的元素只保留一份
- map中元素都是`pair模板类`对象。`first`返回key，`second`返回value
- map有`[]`成员函数，支持k-v赋值
- 返回对象为second成员变量的引用。若没有关键字key的元素，则会往pairs里插入一个关键字为key的元素，其值用无参构造函数初始化，并返回其值的引用

```cpp
map<string,int> ages;
ages["mike"]=40;
ages["kay"]=20;
cout<<map["kay"]<<endl;
//add
pair<string, int> peter("peter",44);
ages.insert(perter);
//find
if(ages.find("Vickey")!=ages.end()){ //find返回一个迭代器
	cout<<"Found"<<endl;
}
for(auto itor=ages.begin(); itor!=ages.end(); itor++){
	pair<string, int> p = *itor;
	cout<<p->first;
	cout<<p->second;
}
```

- 使用自定义对象作为`key`
	- 需要实现对象的排序方式

```cpp
class Person{
private:
	int age;
	string name;
public:
	Person():name(""),age(0){}
	Person(const Person& p){
		name = p.name;
		age = p.age;
	}
	Person(string name, int age):name(name),age(age){}
	bool operator<(const Person& p) const {  //重载<做比较运算,声明成const，不会改变内部状态
		return age<p.age;
	}
};
int main(){
	map<Person, int> people;
	people[Person("mike",44)] = 40;
	people[Person("kay",22)] = 40;

	return 0;
}
```

#### multimap

- 定义：

```cpp
template<class key, class T, class Pred = less<key>, class A = allocator<T>>
class multimap{
	//typedef pair<const key, T> value_type;

};
```

- `multimap`和`map`的区别
	- `multimap`没有重载`[]`，插入元素只能使用`insert`
	- `multimap`中允许相同的key存在，key按照first成员变量从小到大排列，缺省用`less<key>`定义关键字的`<`关系


```cpp
#include<map>
using namespace std;
int main(){
	
	typedef multimap<int,double,less<int>> mmid;
	mmid mmap;
	
	//typedef pair<const key, T> value_type;
	mmap.insert(mmid::value_type(15,2.7)); 
	mmap.insert(mmid::value_type(15,99.3)); 
	mmap.insert(make_pair(14,22.3));
	mmap.count(15); //3
	
	for(auto itor = pairs.begin();itor != paris.end(); itor++){
		i->first;
		i->second;
	}
}
```

## 容器适配器

- 可以用某种顺序容器实现
	- `stack` : LIFO
	- `queue`: FIFO
	- `priority_queue`
		- 优先级队列，最高优先级元素第一个出列
- 通用API
	- `push/pop`：添加，删除一个元素
	- `top`: 返回容器头部元素的引用

- 容器适配器上**没有迭代器**
	- STL中各种排序，查找，变序等算法不适合容器适配器

### stack 

- LIFO数据结构
- 只能插入，删除，访问栈顶元素
- 可用`vector`,`list`,`deque`来实现
	- STLss默认情况下用`deque`实现
	- `vector`和`deque`实现性能优于`list`实现

```cpp
template<class T, class Cont=deque<T> >
class stack{

};
```

### queue

- FIFO数据结构
- 和`stack` 基本类似, 可以用 `list`和`deque`实现
	- 缺省情况下用deque实现

```cpp
template<class T, class Cont = deque<T> >
class queue {
};
```
- 同样也有push, pop, top函数
	- `push`发生在队尾
	- `pop`, `top`发生在队头, 先进先出

### priority_queue

- 优先级队列，最大元素在队头
- 可以用`vector`和`deque`实现
	- 缺省情况下用`vector`实现
	- `priority_queue` 通常用**堆排序**实现, 保证最大的元
素总是在最前面,默认的元素比较器是 `less<T>`
- 执行pop操作时, 删除的是最大的元素
- 执行top操作时, 返回的是最大元素的引用

```cpp
#include <queue>
#include <iostream>
using namespace std;
int main() {
priority_queue<double> priorities;
priorities.push(3.2);
priorities.push(9.8);
priorities.push(5.4);
while( !priorities.empty() ) {
	cout << priorities.top() << " ";	
	priorities.pop();//输出结果: 9.8 5.4 3.2
}
return 0;
} 
```