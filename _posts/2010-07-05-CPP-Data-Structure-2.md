---
layout: post
list_title: Data Structure | 数据结构基础 | 链表 | Linked List
title: 链表
categories: [DataStructure]
---

> 本文会先介绍链表的基本概念和常用操作，以及面试中经常出现的关于链表的问题

### 基本概念

- 通过指针把它的一串存储节点连接成一个链
- 存储节点由两部分组成
	- 数据域(data) + 指针域（后继地址,next） 
- <mark>带头节点的单链表<mark>
	- 头节点是一个虚节点：
        - `(head)->first->second->third...->NULL`
        - 本身没有值，方便操作
        - <mark>头节点本身可以代表一个链表</mark>
	- 第一个节点：`head->next, head != NULL;`
	- 空表判断：`head->next == NULL;`
	- 当前节点：fence->next(curr隐含)
- 分类
    - 单链，双链，循环链

### ADT

```cpp
template <class T> class Link {
 public:
    T data; // 用于保存节点元素的内容
    Link<T> * next; // 指向后继节点的指针
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
        //返回头节点
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
    - 单链表的`next`字段仅仅指向后继节点，不能有效地找到前驱, 反之亦然
- 增加一个指向前驱的指针`prev | data | next`

```
head-->[]<-->[a0]<-->a[1]<-...->[an](tail) 
```

### 定义

```cpp
template <class T> 
class DLink {
public:
    T data; // 用于保存节点元素的内容
    Link<T> * next; // 指向后继节点的指针
    Link<T> *prev; // 指向前驱节点的指针
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
	- 循环链表尾节点指向头节点

- 链表处理
	- 空链表的特殊处理
	- 插入或删除节点时指针勾链的顺序
	- 指针移动的正确性
		- 插入
		- 查找或遍历   

### 数组和链表的比较

- **数组**
    - 没有使用指针，不用花费额外开销
    - 线性表元素的读访问非常便利
    - <mark>插入，删除运算时间代价`O(n)`,查找则可以常数时间完成</mark>
    - 预先申请固定长度的连续空间
    - 如果整个数组元素很慢，则没有结构性存储开销
    - 适合静态数据结构
    - 不适合：
        - 经常插入删除时，不宜使用顺序表
        - 线性表的最大长度也是一个重要因素 

- **链表**
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


### 数组和链表存储密度

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

## 常见的链表问题

本节会介绍面试中经常出现的链表问题，熟悉这些问题对于掌握链表的操作技巧至关重要，由于后面的问题均需要使用链表结构，这里先给出单链表的定义

```cpp
struct ListNode
{
    int val;
    ListNode *next;
    ListNode(int x) : val(x), next(NULL) {}
};
```
后面的问题将会反复用到`ListNode`，则不再重复声明

### [单链表反转](https://leetcode.com/problems/reverse-linked-list)

反转链表是链表中的常见操作，也是很多高级链表算法的基础步骤之一，其问题描述为：

```
Reverse a singly linked list.

Input: 1->2->3->4->5->NULL
Output: 5->4->3->2->1->NULL
```
翻转链表的方法有很多，常见的有迭代法和递归法，这里主要介绍基于迭代翻转算法。其思路为从第二个节点开始依次向后指向头结点，效果如下：

```
1 --> 2 --> 3 --> 4
loop#1: 1 --> null   |  2-->3-->4
loop#2: 2 --> 1 --> null | 3-->4
loop#3: 3 --> 2 --> 1 --> null | 4
loop#4: 4 --> 3 --> 2 --> 1 --> null | 
```
代码实现为:

```cpp
class Solution
{
public:
    ListNode *reverseList(ListNode *head){
       ListNode* first = nullptr;
       ListNode* second = nullptr;
       while(head){
           //保存头结点
           first = head;
           //移动头结点
           head = head->next;
           //断开链表
           first->next = second;
           second = head->next;
           //翻转后链表的头结点
           second = first;
       }
    }
};
```
不难得出，上述算法的时间复杂度为$O(N)$, 空间复杂度为$O(1)$

### [链表中环的检测](https://leetcode.com/problems/linked-list-cycle/description/)

如何判断单链表中是否有环的思路很简单，定义两个前后两个指针，前面指针走两格，后面指针走一格，如果能相遇，则表明链表中有环

```cpp
class Solution {
public:
    bool hasCycle(ListNode *head) {
        if(!head || !head->next){
            return false;
        }
        ListNode* first = head;
        ListNode* second = first->next;
        while(second != first){
            //first前进一步
            first = first->next;
            //如果second为前方为空，则链表没有环
            if(!second->next || !second->next->next){
                return false;
            }
            //second前进两步
            second = second->next;
            second = second->next;
        }
        return true;
    }
};
```

不难得出，上述算法的时间复杂度为$O(N)$, 空间复杂度为$O(1)$

### [两个单链表的交点](https://leetcode.com/problems/intersection-of-two-linked-lists/description/)

问题描述如下:

Write a program to find the node at which the intersection of two singly linked lists begins.For example, the following two linked lists:

```
A:          a1 → a2
                   ↘
                     c1 → c2 → c3
                   ↗            
B:     b1 → b2 → b3
```
begin to intersect at node c1.

**Notes**:

1. If the two linked lists have no intersection at all, return null.
2. The linked lists must retain their original structure after the function returns.
3. You may assume there are no cycles anywhere in the entire linked structure.
4. Your code should preferably run in `O(n)` time and use only `O(1)` memory.

这个题的解法关键是找到两个链表长度的差值，有了差值，我们就可以让长的链表先走完差值，然后两个链表一起走即可。为了找到这个差值，可以让两个列表各自先走到终点，计算长度差，这样算下来时间复杂度约为`O(3*N)`，具体步骤如下：

1. 两个链表各自走到终点，记录长度l1,l2,计算差值，l1-l2
2. 长的链表先走完差值
3. 两个链表一起走，观察节点的next值是否相同

```cpp
class Solution {
public:
    ListNode *getIntersectionNode(ListNode *headA, ListNode *headB) {
        if(!headA || !headB){
            return nullptr;
        }
        int la = 0; int lb = 0;
        ListNode* pa = headA;
        ListNode* pb = headB;
        //统计A,B链表长度
        while(pa&&pb){
            la++; 
            lb++;
            pa= pa ->next;
            pb= pb ->next;
        }      
        while(pb){
            lb++;
            pb = pb->next;
        }
        while(pa){
            la++;
            pa = pa->next;
        }
        //移动长的链表头部
        if(la >= lb){
            int delta = la - lb;
            for(int i =0; i<delta; ++i){
                headA = headA->next;
            }
        }else{
            int delta = lb - la;
            for(int i=0;i<delta; ++i){
                headB = headB -> next;
            }
        }
        //寻找相交节点
        while(headA && headB){
            if(headA == headB){
                return headA;
            }else{
                headA = headA -> next;
                headB = headB -> next;
            }
        }
        return nullptr;
        
    }
};
```

### [merge两个有序链表](https://leetcode.com/problems/merge-two-sorted-lists)

1. 比较两个链表节点头指针，小的前进，大的不动
2. 新链表指向小的节点
3. 新链表尾部追加两个链表中较长的一个

```cpp
class Solution {
public:
    ListNode* mergeTwoLists(ListNode* l1, ListNode* l2) {
        //创建一个dummy，用来提供初始指针
        ListNode dummy = ListNode(-1);
        ListNode* head = &dummy;
        ListNode* tmp = head;
        while(l1 && l2){
            //比较两个节点值
            if(l1->val < l2->val){
                tmp->next = l1;
                l1 = l1->next;
            }else{
                tmp->next = l2;
                l2 = l2->next;
            }
            tmp = tmp->next;
        }
        //append left
        if(l1){
            tmp->next = l1;
        }else{
            //append right
            tmp->next = l2;
        }
        return head->next;
    }
};
```
上述算法的时间复杂度为`O(N)`，空间复杂度为`O(1)`

### [删除链表倒数第N个结点 ](https://leetcode.com/problems/remove-nth-node-from-end-of-list/description/)

题目描述为：

```
Given a linked list, remove the n-th node from the end of list and return its head.

Example:

Given linked list: 1->2->3->4->5, and n = 2.

After removing the second node from the end, the linked list becomes 1->2->3->5.
Note:

Given n will always be valid.
```
解这道题的思路为

1. 使用两个指针同时移动，两个指针之间的间隔为N
2. 当第二个指针走到链表末尾时，第一个指针的下一个节点即为待删除节点

```cpp
class Solution {
public:
    ListNode* removeNthFromEnd(ListNode* head, int n) {
        ListNode* p = head;
        ListNode* runner = head;
        int count = 0;
        while(p->next){
            p = p->next;
            count ++;
            if(count > n){
                runner = runner->next;
            }
        }
        //分析三种情况
        if(count + 1 == n){
            //remove head
            return head->next;
        }else if(count+1 < n){
            return head;
        }else{
            ListNode* tmp = runner->next->next;
            runner->next = tmp;
            return head;
        }
    }
};
```

### [求链表的中间结点](https://leetcode.com/problems/middle-of-the-linked-list/description/)

这道题目的解法和上面类似，也是采用双指针走法：

1. `fast`指针走两格，`slow`指针走一格。
2. 当`fast`指针走到头或者无法前进时，`slow`指针即为中间节点。

```cpp
class Solution {
public:
    ListNode* middleNode(ListNode* head) {
        if(!head || !head->next){
            return head;
        }
        ListNode* fast = head;
        ListNode* slow = head;
        while(fast->next){
            if(fast->next->next){
                fast = fast->next->next;
            }else{
                fast = fast->next;
            }
            slow = slow->next;
        }
        return slow;
    }
};
```

## Resources

- [Linked-List Practice](https://www.geeksforgeeks.org/data-structures/linked-list/)
- [CS106B-Stanford-YouTube](https://www.youtube.com/watch?v=NcZ2cu7gc-A&list=PLnfg8b9vdpLn9exZweTJx44CII1bYczuk)
- [Algorithms-Stanford-Cousera](https://www.coursera.org/learn/algorithms-divide-conquer/home/welcome)
- [算法与数据结构-1-北大-Cousera](https://www.coursera.org/learn/shuju-jiegou-suanfa/home/welcome)
- [算法与数据结构-2-北大-Cousera](https://www.coursera.org/learn/gaoji-shuju-jiegou/home/welcome)
- [算法与数据结构-1-清华-EDX](https://courses.edx.org/courses/course-v1:TsinghuaX+30240184.1x+3T2017/course/)
- [算法与数据结构-2-清华-EDX](https://courses.edx.org/courses/course-v1:PekingX+04833050X+1T2016/course/)
- [算法设计与分析-1-北大-Cousera](https://www.coursera.org/learn/algorithms/home/welcome)
- [算法设计与分析-2-北大-EDX](https://courses.edx.org/courses/course-v1:PekingX+04833050X+1T2016/course/)
- [Reverse Linked List](https://leetcode.com/problems/reverse-linked-list/description/)