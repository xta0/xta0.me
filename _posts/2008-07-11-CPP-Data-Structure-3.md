---
layout: post
title: Data Structure Part 3
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

### 查找节点

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
```

### 插入节点

```cpp
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
```




#### 接口定义

```cpp

```
#### 单链表插入

- 创建新节点
- 新节点指向右边的节点
- 左边节点指向新节点

#### 单链表的删除

- 用p指向元素x的节点的前驱节点
- 删除元素x的节点
- 释放x占据的空间

#### 单链表上的运算分析

- 对一个节点的操作，必须先找到它
- 找单链表中的任一节点，必须从第一个点开始
- 单链表操作的时间复杂度为`O(n)`
	- 定位：`O(n)`
	- 插入：`O(n) + O(1)`
	- 删除：`O(n) + O(1)`


### 环形链表

### 约瑟夫问题

## 顺序表


### 循环链表