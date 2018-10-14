---
layout: post
list_title: 数据结构基础 | Data Structure | 堆 | Heap
mathjax: true
title: 堆
categories: [DataStructure]
---

## 堆

堆是<mark>完全二叉树</mark>的一种表现形式。以最小堆为例，它要求的每个父节点的值大于两个子节点的值，两个兄弟节点之间的值的大小关系没有限制。由于完全二叉树可以用数组表示，上述性质也可以表述为：

1. $K_1<=K_{2i+1}$
2. $K_1<=K_{2i+2}$

因此使用最小堆可以找出这组节点的最小值。推而广之，对于一组无序的数，可以将他们构建成堆来快速得到最大值或最小值，当有新元素进来时，堆也依然可以保持这种特性

<img src="{{site.baseurl}}/assets/images/2008/07/tree-7.jpg" style="margin-left:auto; margin-right:auto;display:block">

堆的核心操作有如下三种

1. 将一组无序的数组织成堆的形式
    - 思路1：将n个无序的数先放到数组中进行调整
    - 思路2：将n个无序的数一个一个进行入堆操作
2. 新元素入堆后如何调整堆
    - 将其放到数组左右一个元素的位置
    - 递归进行SiftUp调整
3. 堆顶元素出堆后如何调整
    - 将数组最后一个节点的值付给堆顶元素
    - 删除最后一个元素
    - 堆顶节点递归进行SiftDown调整

- SiftDown调整

所谓SiftDown调整，即将一个不合适的父节点下降到合适的位置，例如删除堆顶元素后，新的堆顶要进行SiftDown调整

```cpp
//递归调整
void sift_down(size_t position){
    //找到左右节点的index
    size_t l = left_child_index(position);
    size_t r = right_child_index(position);
    
    //叶子节点
    if(l == -1 && r == -1){
        return;
    }
    //左子树为空，说明当前已经是叶子节点
    else if(l == -1 ){
        return;
    }
    //右子树为空，比较左节点
    else if(r == -1 ){
        if(tree[i] > tree[l]){
            swap(tree[i], tree[l]);
            sift_down(l);
        }
    }else{
        //左右子树都不空
        size_t index = tree[l] < tree[r] ? l:r;
        if(tree[i] > tree[index]){
            swap(tree[i],tree[index]);
            sift_down(index);
        }
    }
}
```
- SiftUp调整

和SiftDown类似，即将一个不合适的子节点上升到合适的位置，例如新元素进入堆之后，该元素要进行SiftDown调整

```cpp
void sift_up(size_t position){
    //递归比较
    size_t p = parent_index(i);
    if(p == -1){
        return ;
    }
    if(tree[position]<tree[p]){
        swap(tree[position], tree[p]);
        SiftUp(p);
    }
}
```

- 建堆

如上文所述，建堆有两种思路，其中第二种思路较为简单，可以退化为入堆操作，第一种思路需要按下面步骤操作：

1. 将`n`个关键码放到一维数组中，整体不是最小堆
2. 由完全二叉树的特性，有一半的节点`⌊n/2⌋`是叶子节点，它们不参与建堆的过程
    - `i≥⌊n/2⌋` 时，以关键码`Ki`为根的子树已经是堆
3. 从倒数第二层最右边的非叶子节点开始（完全二叉树数组`i=⌊n/2-1⌋`的元素），依次向前，进行递归SiftDown调整。


<img src="{{site.baseurl}}/assets/images/2008/07/tree-8.jpg" style="margin-left:auto; margin-right:auto;display:block">

例如上图中，我们有一组8个数的无序序列`{72,73,71,23,94,16,05,68}`，建堆步骤为

1. 按照完全二叉树排布，形成树形结构，如上图
2. 成树后，可以看到有4个节点(`⌊4/2⌋`)已经是叶子节点，它们不需要参与建堆的过程
3. 从`23`开始（数组第`i=⌊4/2-1⌋=3`项）向前依次进行递归调整，顺序依次是
    - `23` 递归SiftDown
    - `71` 递归SiftDown
    - `73` 递归SiftDown
    - `72` 递归SiftDown

```cpp
template<class T>
void MinHeap<T>::BuildHeap(){
    for (int i=CurrentSize/2-1; i>=0; i--)
        SiftDown(i);
    } 
}
```
分析一下建堆的效率:

1. $n$个节点的堆，高度为$d=⌊\log_2^{n}+1⌋$，设根为第$0$层，第$i$层节点数为$2^i$
2. 考虑一个元素在队中向下移动的距离
    - 大约一半的节点深度为$d-1$，不移动（叶）。
    - 四分之一的节点深度为$d-2$，而它们至多能向下移动一层。
    - 树中每向上一层，节点的数目为前一层的一半，而子树高度加一。
    - 因此元素移动的最大距离的总数为：

    $$\sum_{i=1}^{\log{n}}(i-1)\frac{n}{2^i}=O(n)$$

- 堆操作时间复杂度分析
    - <mark> 建堆算法时间代价为$O(n)$</mark>
    - 堆有$\log{n}$层深
    - 插入节点、删除普通元素和删除最小元素的<mark>平均时间代价</mark>和<mark>最差时间代价</mark>都是$O(\log{n})$

- 优先队列
    - 根据需要释放具有最小（大）值的对象
    - 最大树、 左高树HBLT、WBLT、MaxWBLT
    - 改变已存储于优先队列中对象的优先权
        - 辅助数据结构帮助找到对象



### Resources

- [CS106B-Stanford-YouTube](https://www.youtube.com/watch?v=NcZ2cu7gc-A&list=PLnfg8b9vdpLn9exZweTJx44CII1bYczuk)
- [Algorithms-Stanford-Cousera](https://www.coursera.org/learn/algorithms-divide-conquer/home/welcome)
- [算法与数据结构-1-北大-Cousera](https://www.coursera.org/learn/shuju-jiegou-suanfa/home/welcome)
- [算法与数据结构-2-北大-Cousera](https://www.coursera.org/learn/gaoji-shuju-jiegou/home/welcome)
- [算法与数据结构-1-清华-EDX](https://courses.edx.org/courses/course-v1:TsinghuaX+30240184.1x+3T2017/course/)
- [算法与数据结构-2-清华-EDX](https://courses.edx.org/courses/course-v1:PekingX+04833050X+1T2016/course/)
- [算法设计与分析-1-北大-Cousera](https://www.coursera.org/learn/algorithms/home/welcome)
- [算法设计与分析-2-北大-EDX](https://courses.edx.org/courses/course-v1:PekingX+04833050X+1T2016/course/)


