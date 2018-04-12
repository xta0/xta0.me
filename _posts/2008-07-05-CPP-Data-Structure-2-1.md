---
layout: post
title: Array

---

## 线性表

### 线性表的概念

- 线性表简称表，是零个或者多个元素的**有穷**序列，通常可以表示成`k0,k1,...,kn-1(n>=1)`
	- 表目：线性表中的元素
	- 索引（下标）：i称为表目ki的“索引”或“下标”
	- 表的长度：线性表中所有包含元素的个数n
	- 空表：长度为0的线性表

- 线性表特点：
	- 操作灵活，长度可以增长，缩短
	- 适合数据量较小的场景

### 线性结构

- 二元组 `B=(K,R) K={a0,a1,...,an-1} R={r}`  
	- 有一个**唯一**的**开始节点**，它没有前驱，有一个**唯一**的**直接后继**
	- 一个唯一的**终止节点**，它有一个唯一的直接前驱，没有后继
	- 其它的节点成为**内部节点**，每一个内部节点有且仅有一个前驱节点和一个后继节点：`<ai, ai+1>` `ai`是`ai+1`的前驱，`ai+1`是`ai`的后继
	- 前驱/后继关系r，具有反对称性和传递性

- 特点
	- 均匀性：虽然不同的线性表的数据元素可以是各种各样的，但对于同一个线性表的个数据元素必定是**相同**的**数据类型**和**长度**  
	- 有序性：各数据元素在线性表中都有自己的位置，且数据元素之间的**相对位置**是**线性**的

- 按复杂程度划分：
	- 简单的：线性表，栈，队列，散列表
	- 高级的：广义表，多维数组，文件...

- 按访问方式划分：
	- 直接访问：向量，数组
	- 顺序访问：栈，队列，列表，广义表，链表
	- 目录索引：字典，散列表

- 安存储划分
	- 顺序表: `vector<T>`
	- 链表: `list<T>`
- 按操作划分
	- 线性表
		- 所有表目都是同一类型节点的线性表
		- <mark>不限制操作形式</mark>
	
	- 栈(LIFO)
		- <mark>同一端操作</mark>
		- 插入和删除操作在同一端进行
			- DFS，深度搜索 
		
	- 队列(FIFO)
		- <mark>两端操作</mark>
		- 插入操作在表的一端，删除在表的另一端

### 小结

- 三个方面
	- 线性表的逻辑结构
		- 线性表长度
		- 表头
		- 表尾
		- 当前位置 
	- 线性表的存储结构
		- 顺序表
			- 按索引值从小到大存放在一片相邻区域
			- 紧凑结构，存储密度为1 
		- 链表	 
			- 单链表
			- 双链表
			- 循环链表 
	- 线性表运算 		 
		- 对表内元素的增删改查
		- 排序，检索    

- 线性表运算
	- 创建
	- 删除
	- 增，删，改，查
	- 排序
	- 检索

```cpp
//interface
template <class T> class List {
	void clear(); // 置空线性表
	bool isEmpty(); // 线性表为空时，返回 true
	bool append(const T value);
	// 在表尾添加一个元素 value，表的长度增 1
	bool insert(const int p, const T value);
	// 在位置 p 上插入一个元素 value，表的长度增 1
	bool delete(const int p);
	// 删除位置 p 上的元素，表的长度减 1
	bool getPos(int& p, const T value);
	// 查找值为 value 的元素并返回其位置
	bool getValue(const int p, T& value);
	// 把位置 p 元素值返回到变量 value
	bool setValue(const int p, const T value);
	// 用 value 修改位置 p 的元素值
};
```

## 顺序表

#### 基本概念

- 也称向量，采用<mark>定长</mark>的一维数组存储结构
- 主要特性
	- 元素类型相同
	- <mark>元素顺序的存储在连续的存储空间中，每个元素有唯一的索引值</mark>
	- 使用常数作为向量长度
- 数组存储
- 读写元素很方便，通过下标即可指定位置
	- 只要确定了首地址，线性表中任意数据元素都可以<mark>随机访问</mark>

- 元素地址计算

```
Loc(ki) = Loc(k0)+c*i, c=sizeof(ELEM)
```

#### 顺序表类定义

```cpp
class Array:public List<T>{
	private:
		T* list;
		int maxSize; //向量实际的内存空间
		int curLen; //向量当前长度，这个值小于maxSize，要预留一部分存储空间，放置频繁的内存开销
		int position;
	public:
		Array(const int size){
			maxSize = size;
			list = new T[maxSize];
			curLen = position = 0;
		}
		~MyList(){
			delete []list;
			list = nullptr;
		}
		void clear(){
			delete []list;
			list = nullptr;
			position = 0;
			list = new T[maxSize];
		}
		int length();
		bool append(const T value);
		bool insert(const int p, const T value);
		bool remove(const int p);
		bool setValue(const int p, const T value);
		bool getValue(const int p, T& value);
		bool getPos(int &p, const T value);
}

```

### 顺序表上的运算

- 插入运算

```cpp
template<class T>
bool Array<T>::insert(const int p, const T value){
	if(curLen >= maxSize){
		//重新申请空间
		//或者报错
		return false;
	}
	if(p<0 || p>=curLen){
		return false;
	}
	for(int i=curLen;i>p;i--){
		list[i] = list[i-1];
	}
	list[p] = value;
	curLen ++;

	return true;
}
```
- 删除运算

```cpp
template<class T>
bool Array<T>::delete(const int p){
	if(curLen == 0){
		return false;
	}
	if(p == 0 || p>=curLen){
		return false;
	}
	for(int i=p; i<curLen-1; i++){
		list[i] = list[i+1];
	}
	curLen--;

	return true;
}
```

- 表中元素的移动
	- 插入：移动<mark>n-i</mark>个元素
	- 删除：移动<mark>n-i-1</mark>个元素
	- <mark>时间复杂度为O(n)</mark>

### Where to Go

下一节我们讨论线性表的第二种实现方式——链表