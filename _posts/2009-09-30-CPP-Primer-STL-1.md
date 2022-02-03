---
layout: post
list_title: C++ Primer | STL Containers | STL容器
title: 顺序容器 & 关联容器
categories: [C++]
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
对于非集合类的静态数组，也可以使用迭代器

```cpp
int a[4] = {1,23,3,5};
int* begin = std::begin(a);
int* end = std::end(a);
int* result = find(begin,end,10);
if(result!=end){
	//found
} 
```
对于动态数组，则不能使用`std::begin`和`std::end`

```cpp
int *a = new int[4]{1,2,3,5};
int* result = find(a,a+4,10);
if(result!=end){
	//found
} 
```

## 顺序容器

### 几种顺序容器概述
	
- 元素是非排序的，元素的插入位置与元素的值无关。
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

### 顺序容器API

- 顺序容器的**初始化**方式

| -- | -- |
| `C c` | 默认构造函数，如果`C`是一个`array`，则`c`中元素按默认方式初始化；否则`c`为空 |
| `C c1(c2)` or `C c1=c2`| 拷贝构造，`c1`初始化为`c2`的拷贝 |
| `C c{a,b,c..}` or `C c={a,b,c...}`| 使用初始化列表，`c`初始化为initialize list中元素的<mark>拷贝</mark>。初始化列表中元素的个数必须小于等于`array`的大小，遗漏的元素进行默认初始化|
| `C c(start, end)`| 使用迭代器初始化，`c`初始化为迭代器`start`和`end`指定范围中的元素的拷贝|
| `C c(n)`| `c`包含`n`个元素，这些元素进行了值初始化（默认值），string不适用|
| `C c(n,t)`| `c`包含`n`个值为`t`的元素|

- 顺序容器的**赋值**操作

| -- | -- |
| `c1=c2` | c1的元素替换为c2元素的拷贝, c1和c2类型相同 |
| `c={a,b,c,...}` | c元素替换为初始化列表中元素的拷贝 |
| `swap(c1,c2)`| 交换c1和c2中的元素，swap操作不对任何元素进行拷贝，删除或者插入操作，因此可以在常数时间完成，<mark>swap通常比从c2拷贝到c1快的多</mark> | 
| `seq.assign(b,e)`| 将`seq`中的元素替换为迭代器`b`和`e`中间的元素，迭代器不能指向`seq`|
| `seq.assign(il)`| 将`seq`中的元素替换为初始化列表中的元素|
| `seq.assign(n,t)`| 将`n`个值为`t`的元素|

- 顺序**容器的大小**

|--|--|
|`size()`| 容器中元素数目，`forward_list`不支持该操作 |
|`empty()`| 如果容器中元素个数为0返回`true`|
|`max_size`| 返回一个大于或等于该类容器所能容纳的最大元素数值|

- 顺序容器**比较**

1. 如果两个容器大小相等，元素值相等，则这两个容器相等
2. 如果两个容器大小不同，但较小的容器中的每个元素都等于较大元素中的每个元素，则较小容器小于较大容器
3. 如果两个容器都不是另一个容器的前缀子序列，则它们的比较结果取决于第一个不相等的元素的比较结果

```cpp
vector<int> v1 = { 1, 3, 5, 7, 9, 12 };
vector<int> v2 = { 1, 3, 9 };
vector<int> v3 = { 1, 3, 5, 7 };
vector<int> v4 = { 1, 3, 5, 7, 9, 12 };

v1 < v2 // true; 
v1 < v3 // false; 
v1 == v4 // true; 
v1 == v2 // false; 
```

- **添加元素**

1. 添加元素会导致容器元素个数增加，因此`array`不支持该操作
2. `forward_list`不支持`push_back`
3. `vector`和`string`不支持`push_front`和`emplace_front`

> 这里忽略了对`emplace`操作的讨论，个人认为用处不大

|--|--|
| `c.push_back(t)` | 在`c`尾部追加元素 |
| `c.push_front(t)`| 在`c`头部追加元素 |
| `c.insert(p,t)`| 在迭代器`p`的<mark>位置之前</mark>插入元素`t`,返回`p+1`的位置|
| `c.insert(p,n,t)`| 在迭代器`p`的<mark>位置之前</mark>插入`n`个元素`t`,返回`p+1`的位置|
| `c.insert(p,b,e)`| 在迭代器`b`和`e`中的元素插入到`p`的<mark>位置之前</mark>,返回`p+1`的位置|
| `c.insert(p,il)`| `il`是一组花括号包围的元素值列表，将这些值插入到迭代器`p`的<mark>位置之前</mark>,返回`p+1`的位置|

- **访问**

|---|---|
| `c.back()` | 返回尾部元素的<mark>引用 |
| `c.front()` | 返回头部元素的引用 |
| `c[n]` | 返回`n`下标对应元素的引用 |
| `c.at(n)` | 返回`n`下标对应元素的引用 |

- **删除**

|--|--|
| `c.pop_back()`| 删除尾部元素 |
| `c.pop_front()`| 删除头部元素 |
| `c.erase(p)`| 删除迭代器`p`指向的元素，返回`p+1` |
| `c.erase(b,e)`| 删除`b`和`e`中的元素，返回`e+1` |
| `c.clear()`| 删除全部元素 |


### Array
	
<mark>C++ 11</mark>新增的<mark>静态数组</mark>类`array`,构造时必须要同时指定传入类型和size。 和C的静态数组不同的是，`array`可以拷贝和赋值

```cpp
//类型为42个int型数组
array<int,10> ia1; //初始化10个默认值的int数组
array<int,10> ia2 = {0,1,2,3,4,5,6,7,8,9};
array<int,10> ia3 = {42}; //ia3[0]为42，其余为0
array<int,10> ia4 = {0}; //所有元素均为0
array<int,10> copy = ia3; //正确，类型匹配即合法
```

### List

<mark>List是双向链表，不支持完全随机访问, 不能用标准库中的`sort`函数排序</mark>，排序需要使用自己的`sort`成员函数`void sort()`，默认将`list`中的元素按照`<`排列，

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

标准库提供8个关联容器，这个8个容器的不同点体现在下面3个维度上：

1. 或者是一个`set`或者是一个`map`
2. 是否允许集合内存在重复元素
3. 元素在集合内是否按顺序存储。无序的容器以`unordered`开头

- 有序容器
	- `set`
	- `multiset`
	- `map`
	- `multimap`
- 无序容器
	- `unordered_map`
	- `unorderd_multimap`
	- `unordered_set`
	- `unordered_multiset`

> 对于关联容器，通常不使用泛型算法，而是使用其自带的算法API，其原因有两点：1.是关联容器的key都是const的，对于需要修改容器内容的泛型算法不适合。 2.对于只读的泛型算法，比如find，只能对容器进行顺序检索，而如果使用关联容器自带的find算法则会进行hash查找，效率要高很多。

### 关联容器的迭代器

- 对于map中的每个pair，key是const的，不能修改

```cpp
map<string,size_t> m { {"abc",100} };
auto itor = map.begin();
itor->first = "def"; //wrong! key是const
```

- 对于set中的每个元素是const的，不能修改

```cpp
set<int> s{1,2,3,4};
auto itor = s.begin();
*itor = 100; //wrong, set中的key是const
```

- 集合的遍历

对于关联容器，可以使用迭代器进行遍历，对于map，集合中的每个元素是pair对象

```cpp
map<string,size_t> m { {"abc",100},{"def",101} };
for(auto itor=m.begin(); itor!=m.end(); itor++){
	//itor是指向pair的指针，用->访问
	string key = itor->first;
	size_t value = itor->second;
}
//range loop
for(auto &p : m){
	//此时p是pair对象，不是itor，访问元素用.
	string key = m.first;
	size_t value = m.second;
}
```

### 关联容器API

- **添加元素**

|--|--|
| `c.insert(v)` | v是value_type对象 |
| `c.insert(b,e)`| 向c中插入迭代器b，e中的元素，返回void|

对于map，如果集合中已经有相同key的元素，则插入无效，并返回一个pair，类型为`pair<map<string,size_t>::iterator, bool>`，其中pair的first指向插入元素的itor，pair的second表示插入是否成功。

```cpp
map<string,size_t> word_count;
string word;
while(cin>>word){
	auto ret = word_count.insert({word,1}); 
	if(!ret.second){ //word已经在集合中
		++ret.first->second; //更新已有元素的数量
	}
}
```
对于multimap，insert没有限制。实际中，用到multimap的场景不多，multimap适用于一对多的结构，例如我们可能想建立作者到他的著作之间的映射，一个作者可能有多份著作，这时我们需要用multimap

```cpp
multimap<string,string> authors;
authors.insert{ {"John Smith","Book#1"} };
authors.insert{ {"John Smith","Book#2"} };
```
此时可以无需关心insert的返回值。

- **删除元素**

关联容器提供了三种删除元素的API

|--|--|
| `c.erase(k)`| 删除key为k的元素，返回删除元素数量，类型为`size_type` |
| `c.erase(p)`| 从c中删除迭代器p指定的元素。p不能为`c.end()`，返回值为p后面的迭代器|
| `c.erase(b,e)`| 删除`b`和`e`中的元素，返回`e` |

```cpp
//map
size_t ret = word_count.erase("kate")); //ret为0或1
//multimap
auto cnt = author.erase("John Smith"); //ret>=0
```

- **下标操作**

|--|--|
| `c.[k]`| 返回关key为k的元素，如果k不在c中，则会创建一个key为k的元素，并对其初始化 |
| `c.at(k)`| 访问key为k的元素，带参数检查，如果k不在c中，则抛异常 |

对于下标操作需要注意三点：

1. `c[k]`返回的类型是`map<k,v>::mapped_type`不是`map<k,v>::value_type`。而解引用一个map迭代器会返回`value_type`类型，也就是`pair`类型，因为对`map`来说，它的都是`pair`对象，因此`value_type`类型自然是`pair`类型
	
	```cpp
	KeyObject k3 = {"Jason",33};
    m[k3] = 100;
    map<KeyObject,int>::mapped_type mt = m[k3]; //mt 是int类型
    map<KeyObject,int>::value_type vt = m[k3]; //wrong! m[k3]返回的是mapped_type,不是value_type

	//对迭代器介解引用，得到value_type类型
	auto itor = m.begin();
    map<KeyObject,int> ::value_type p = *itor; //p是pair类型
    cout<<p.first.get_name()<<endl;
	```
2. `c[k]`返回的是左值引用，可以对value进行修改。
3. 注意副作用，如果`k`不在`c`中，`c[k]`会创建一个key为k的对象

- **查找操作**

|--|--|
| `c.find(k)`| 返回key为k的元素的迭代器，如果不存在，则返回`c.end()` |
| `c.count(k)`| 返回key为k的元素个数 |
| `c.lower_bound(k)`| 返回集合中第一个key大于等于k的元素的迭代器 |
| `c.upper_bound(k)`| 返回集合中第一个key大于k的元素的迭代器 |
| `c.equal_range(k)`| 返回一个迭代器pair，表示key为k的元素范围，如果k不存在，则pair的两个值均为`c.end()` |

对于查找的API，如果是map或是set则很好理解，如果是multimap或multiset则需要考虑重复key的问题，例如我们想找出某个key对应的所有value，可以用下面几种方法

```cpp
//使用count
string author = "John Smith";
auto entries = authors.count(author);
auto itor = authors.find(author);
while(entries){
	cout<<itor->second<<endl;
	++itor;
	--entries;
}
//使用lower，upper bound
auto itor_lo = authors.lower_bound(author);
auto itor_hi = authors.upper_bound(author); 
for(auto itor =lo; itor!=hi; itor++){
	cout<<itor->second<<endl;
}
//使用equal range
auto p = authors.equal_range(author);
for(auto itor=p.first; itor!=p.second;itor++){
	cout<<itor->second<<endl;
}
```

### set/multiset

- `set`和`multiset`表示数学中集合
- 集合内的元素均是有序存储的，默认比较器为`std::less<T>`值小的元素在前面。
- `set`不允许集合中的有重复元素。`multiset`允许。
- `set`和`multiset`底层实现为BST。
- 如果`set/multiset`中保存的是自定义元素，则需要显示指定序函数

```cpp
auto comp = [](const Sales_Data& lhs, const Sale_Data& rhs){
	return lhs.isbn()<rhs.isbn();
};
set<Sales_Data,decltype(comp)> ss(comp);
```

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
//创建pair
pair<int, int> p(pair<double,double>(5.5,4.6))
p.first = 5;
p.second = 4;
pair<string,int> process(vector<string>& v){
	if(!v.empty()){
		return {v.back(),v.back().size()};
		//或者使用make_pair
		return make_pair(v.back(),v.back().size())
	}else{
		return pair<string,int>(); //隐式构造
	}
}
```

### map/multimap

- **有序的**k-v集合，元素按照`key`**从小到大**排列，缺省情况下用`less<key>`即`<`定义
- map中元素类型为`pair`模板。`first`返回key，`second`返回value，<mark>注意，返回类型为引用</mark>
- `map`支持下标访问`[]`成员函数，支持k-v赋值
- `multimap`由于支持重复元素的存放，因此不支持基于`[]`的下标访问，插入元素只能使用`insert`方法
- <mark>若没有关键字key的元素，则会往pairs里插入一个关键字为key的元素，其值用无参构造函数初始化，并返回其值的引用</mark>

```cpp
map<string,int> ages{ {"Joyce",12}, {"Austen",23} };
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

- 使用自定义对象作为`key`，需要重载`operator<(...)`

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

- 使用自定义比较函数

```cpp
//注意比较的是两个key，不是两个pari
auto comp = [](const string& p1, const string& p2){
	return p1.size() < p2.size();
};

//map只支持key的比较，不支持基于value的比较函数
map<string, int, decltype(comp)> dict(comp);
```

### 无序容器

<mark>C++ 11</mark>标准库提供了4个无序容器，这些容器不使用比较运算符类组织元素，而是使用哈希函数和key类型的`==`运算符。在key类型没有顺序要求的情况下，使用无序的容器更轻量和简单。

无序容器在存储上的组织为一组桶，每个桶保存0个或多个元素，元素通过一个哈希函数映射到某个桶中，映射过程中可能会出现哈希碰撞，导致不同的元素映射到同一个桶中，此时需要遍历桶中的元素来找到待查元素。无序容器提供了一组管理桶的函数，这些函数可以让我们查询每个桶的状态。

|---|---|
|桶接口||
| `c.bucket_count()` | 正在使用桶的数目 |
| `c.max_bucket_count()` | 容器能容纳的最多的桶的数量 |
| `c.bucket_size(n)` | 第n个桶中有多少元素 |
| `c.bucket(k)` | key为k的元素在哪个桶中 |
|桶迭代||
|`local_iterator`| 可以用来访问桶中元素的迭代器类型 |
|`const_local_iterator`| 桶迭代器的const版本 |
|`c.begin(),c.end()`| 返回桶n内元素的首尾迭代器 |
|`c.cbegin(),c.cend()`| 返回桶n内元素的首尾const类型迭代器 |
|哈希策略||
|`c.load_factor`| 返回已使用桶数量和全部桶数量的比值 |
|`c.max_load_factor`| load_factor的最大比值，超过这个值，容器将rehash，使load_factor<max_load_factor |
|`c.rehash(n)`| 重新存储，使`bucket_count>n` |
|`c.reserve(n)`| 重新存储，使c可以保存n个元素而不必rehash |

无序容器需要key实现`==`运算符和一个哈希函数来生成哈希值。因此对于一个自定义类型的key，要自己实现这两个函数。

```cpp
struct Sale{
    string data;
    int num;
    Sale(string _data, int _num):data(_data),num(_num){}
	//实现==
    bool operator==(const Sale& s) const {
        return num == s.num && s.data == data;
    }
};
//指定hash函数
namespace std{
    template<>
    struct hash<Sale>{
        size_t operator()(const Sale& s) const{
            return hash<string>()(s.data);
        }
    };
}
unordered_map<Sale,int> b(
	{ {"Jason",22},100 },
	{ {"Jacob",23},101 }
);
```
另一个常见的例子是使用`unordered_map`存放`pair<T1,T2>`类型的数据，此时由于`T1,T2`的类型未知，因此`pair`并不知道如何计算自己的哈希值，因此需要使用者提供哈希函数:

```cpp
struct pair_hash {
    template <class T1, class T2>
    std::size_t operator () (const std::pair<T1,T2> &p) const {
        auto h1 = std::hash<T1>{}(p.first);
        auto h2 = std::hash<T2>{}(p.second);

        // Mainly for demonstration purposes, i.e. works but is overly simple
        return h1 ^ h2;  
    }
};

using memo = std::pair<std::int, std::int>;
using unordered_map = std::unordered_map<memo, int, pair_hash>;

int main() {
    unordered_map um;
}
```

## 容器适配器

<mark>适配器</mark>是标准库中的一个通用概念，容器，迭代器和函数都有适配器，本质上，一个适配器是一种机制，能够使某种事物物的行为看起来像另一种事务。一个容器适配器接受一个已有的容器，通过某种操作，使其具有另外的一些行为。如果`stack`适配器可以用`vector`或者`list`实现，使其具有`stack`的性质。

- 可以用某种顺序容器实现
	- `stack` : LIFO
	- `queue`: FIFO
	- `priority_queue`: 优先级队列，最高优先级元素第一个出列
- 通用API
	- `push/pop`：添加，删除一个元素
	- `top`: 返回容器头部元素的引用

- <mark>容器适配器上**没有迭代器**</mark>
	- STL中各种排序，查找，变序等算法不适合容器适配器

### stack 

LIFO数据结构，只能插入，删除，访问栈顶元素。可用`vector`,`list`,`deque`来实现，<mark>默认情况下用`deque`实现</mark>，`vector`和`deque`实现性能优于`list`实现

```cpp
template<class T, class Cont=deque<T> >
class stack{};
```

### queue

FIFO数据结构，和`stack` 基本类似, 可以用 `list`和`deque`实现，缺省情况下用deque实现

```cpp
template<class T, class Cont = deque<T> >
class queue {};
```

### priority_queue

1. 优先级队列，默认是最大堆，最大元素在队头，可以用`vector`和`deque`实现，缺省情况下用`vector`实现
2. `priority_queue` 通常用**堆排序**实现, 保证最大的元素总是在最前面(<mark>最大堆</mark>),默认的元素比较器是 `less<T>`
3. 执行pop操作时, 删除的是最大的元素
4. 执行top操作时, 返回的是最大元素的引用

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
除了使用STL提供的比较函数之外，我们还可以自定义比较函数

```cpp
//使用lambda表达式自定义比较函数
priority_queue<ListNode*,std::vector<ListNode* >, std::function<bool(ListNode*,ListNode*)>> pq([](ListNode* l1, ListNode* l2 ){
	return l1->val < l2->val;
});

//也可以换种写法
auto compare = [](ListNode* l1, ListNode* l2){
	return l1->val < l2->val;
};
priority_queue<ListNode*,std::vector<ListNode* >, decltype(compare)>> pq(compare);
```
### Tuples

`tuple`是C++11新引入的feature，类似`pair`模板，`tuple`中的每个元素都可以有不同的类型，和`pair`不同的是，tuple可以有任意多个成员。我们可以使用两种方式来构造`tuple`对象

```cpp
//default init method, each member is set to default value
tuple<size_t, size_t, size_t> threeD;  //all members are set to 0
tuple<string, vector<double>, int, list<int>> items = {
    "constants",
    {3.14, 2.718},
    42,
    {0,1,2,3,4,5}
};
//use make_tuple
auto item = make_tuple("0-999-78345-X",3,20.00);
```
访问tuple对象有些奇怪，我们不能用`tuple.get`，而是需要用标准库函数`std::get`来访问。`std::get`是一个函数模板，我需要指定tuple成员的位置作为模板参数，此外，`std::get`返回的是tuple元素的**引用**

```cpp
auto book = get<0>(item); //返回item的第一个成员
auto cnt = get<1>(item); //返回item的第二个成员
```
当tuple的成员较多时，想要表示tuple对象的类型往往会很复杂，我们可以使用下面两个模板函数来查询tuple成员的数量和类型

```cpp
typedef decltype(item) trans;
```

## Resources

- [C++ Primer](http://www.charleshouserjr.com/Cplus2.pdf)