---
layout: post
title: DataStructure Part 2 - Vector & List

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

## 链表

### 基本概念

- 通过指针把它的一串存储节点连接成一个链
- 存储节点由两部分组成
	- 数据域(data) + 指针域（后继地址,next） 
- <mark>带头结点的单链表<mark>
	- 头结点是一个虚节点：
        - `(head)->first->second->third...->NULL`
        - 本身没有值，方便操作
        - <mark>头结点本身可以代表一个链表</mark>
	- 第一个节点：`head->next, head != NULL;`
	- 空表判断：`head->next == NULL;`
	- 当前节点：fence->next(curr隐含)
- 分类
    - 单链，双链，循环链

## 单向链表

### 两部分定义

- 节点

```cpp
template <class T> class Link {
 public:
    T data; // 用于保存结点元素的内容
    Link<T> * next; // 指向后继结点的指针
    Link(const T info, const Link<T>* nextValue =NULL) {
        data = info;
        next = nextValue;
    }
    Link(const Link<T>* nextValue) {
        next = nextValue;
    }
};
```
- 链表

```cpp
template <class T> class lnkList : public List<T> {
private:
    Link<T> * head,*tail; // 单链表的头、尾指针
    Link<T> *pos(const int i); // 查找链表中第i个节点
public:
    LinkList(int s); // 构造函数
    ~LinkList(); // 析构函数
    bool isEmpty(); // 判断链表是否为空
    void clear(); // 将链表存储的内容清除，成为空表
    int length(); // 返回此顺序表的当前实际长度
    bool append(cosnt T value); // 表尾添加一个元素 value，表长度增 1
    bool insert(cosnt int p, cosnt T value); // 位置 p 上插入一个元素
    bool delete(cosnt int p); // 删除位置 p 上的元素，表的长度减 1
    bool getValue(cosnt int p, T& value); // 返回位置 p 的元素值
    bool getPos(int &p, const T value); // 查找值为 value 的元素
}
```

### 节点操作

```cpp
//查找第i个节点
template<class T>
Link<T> * LinkList<T>::pos(const int i){
    if(i == -1){
        //返回头结点
        return head;
    }
    Link<T>* node = head->next;    
    int count = 0;
    while(node && count<i){
        count++;
        node = node->next;
    }
    return node;
}


//在i的位置插入value
template<class T>
bool LinkList<T>:: insert(const int i, cosnt T value){

    //插入位置的前驱节点
    Link<T>* curr = pos(i-1);
    if(curr == NULL){
        return false;
    }
    
    //创建新节点并插入
    Link<T>* node = new Link<T>(value,curr->next);
    curr->next = node;

    //更新尾节点
    if(curr == tail){
        tail = node;
    }
    return true;
}

//删除第i个节点
template<class T>
bool LinkList<T>::delete(const int i){

    //找到待删除节点的前驱节点
    Link<T>* prev = pos(i-1);
    if(prev == NULL || p == tail){
        //待删除节点不存在
        return false;
    }
    Link<T>* node = prev->next;
    if(node == tail){
        //待删除节点是尾部
        tail = prev;
        prev->next = NULL;
    }else{
        prev->next = node->next;
    }
    
    free(node);
    return true;
}
```

### 单链表上的运算分析

- 对一个节点的操作，必须先找到它
- <mark>找单链表中的任一节点，必须从第一个点开始</mark>
- <mark>单链表操作的时间复杂度为`O(n)`</mark>
	- 定位：`O(n)`
	- 插入：`O(n) + O(1)`
	- 删除：`O(n) + O(1)`


## 双向链表

- 为弥补单链表的不足,而产生双链表
    - 单链表的`next`字段仅仅指向后继结点，不能有效地找到前驱, 反之亦然
- 增加一个指向前驱的指针`prev | data | next`

```
head-->[]<-->[a0]<-->a[1]<-...->[an](tail) 
```

### 定义

```cpp
template <class T> 
class DLink {
public:
    T data; // 用于保存结点元素的内容
    Link<T> * next; // 指向后继结点的指针
    Link<T> *prev; // 指向前驱结点的指针
    Link(const T info, Link<T>* preValue = NULL, Link<T>* nextValue = NULL) {
        // 给定值和前后指针的构造函数
        data = info;
        next = nextValue;
        prev = preValue;
    }
}
```

### 节点操作

- 插入节点

```cpp
new q; //新节点
q->next=p->next
q->prev=p
p->next=q
q->next->prev=q
```

- 删除节点

```cpp
//删除p指向节点
p->prev->next = p->next
p->next->prev = p->prev
p->next=NULL
p->prev=NULL
//是否free可根据具体情况判断
free(p)
```
 
### 循环链表	

- 将单链表或者双链表的头尾节点连起来就是一个循环链表
- 不增加额外存储花销，却给操作里带来方便
	- 从循环链表中任一节点出发，都能访问到链表中的其它节点
	- 循环链判断结束的方法
		- 计数
		- `tail -> next = head`
        
```
---------------------------------------
↓                                     |
head->[]<-->[a0]<-->[a1]<-...->[an](tail)
```


### 链表的边界条件

- 几个特殊点处理
	- 头指针处理
	- 非循环链表尾节点的指针为 `NULL`
	- 循环链表尾节点指向头结点

- 链表处理
	- 空链表的特殊处理
	- 插入或删除节点时指针勾链的顺序
	- 指针移动的正确性
		- 插入
		- 查找或遍历   

## 顺序表和链表的比较

#### 顺序表

- 没有使用指针，不用花费额外开销
- 线性表元素的读访问非常便利
- <mark>插入，删除运算时间代价`O(n)`,查找则可以常数时间完成</mark>
- 预先申请固定长度的连续空间
- 如果整个数组元素很慢，则没有结构性存储开销
- 适合静态数据结构
- 不适合：
	- 经常插入删除时，不宜使用顺序表
	- 线性表的最大长度也是一个重要因素 

#### 链表
- 无需事先了解线性表长度
- 允许线性表动态变化
- 能够适应经常插入删除内部元素的情况
- <mark>插入，删除运算时间代价`O(1)`，但找第i个元素运算时间代价`O(n)`<mark>
- 存储利用指针，动态的按照需要为表中的新元素分配存储空间
- 每个元素都有结构性存储开销
- 适合动态数据结构
- 不适合：
	- 当读操作比插入删除操作频率大时，不用用链表
	- 当指针的存储开销和整个节点内容所占空间相比较大时，应谨慎选择 


### 顺序表和链表存储密度

`n`表示线性表中当前元素的数目
`p`表示指针的存储单元大小(通常为4 bytes)
`e`表示数据元素的存储单元大小
`d`表示可以再数组中存储的线性元素的最大数目

- 空间需求
	- 顺序表的控件需求为 `d*e`
	- 链表的空间需求未`n*(p+e)`

- `n`的临界值，即 `n>d*e/(p+e)`
	- <mark>`n`越大，顺序表的空间效率就更高</mark>
	- 如果`p=e`， 则临界值为 `n = d/2`(如果顺序表有一半以上的空间是满的，那么顺序表效率更高)

### 顺序表和链表的选择

- 顺序表
	- 节点数目大概可以估计
	- 线性表中节点比较稳定（插入删除少）
	- n > de/(p+e)

- 链表
	- 节点数目无法预知
	- 线性表中节点动态变化（插入删除多）
	- n < de/(p+e)  


## Resources

- [Linked-List Practice](https://www.geeksforgeeks.org/data-structures/linked-list/)