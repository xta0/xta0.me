---
layout: post
title: Linked List
---

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

### 约瑟夫问题

- 背景：约瑟夫是一名犹太人，在罗马人占领桥塔帕特之后，39名犹太人与约瑟夫及他的朋友躲到一个山洞中，39个犹太人决定宁愿死也不要被敌人抓到，于是决定了一个自杀方式 
- 问题描述：对于任意给定的n,s和m，求按出列次序得到的人员序列
    - n：参加游戏的人数
    - s: 开始的人
    - m：间隔
- 解题思路

使用循环链表，
