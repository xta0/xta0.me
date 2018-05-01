---
layout: post
title: Data Structure Part 5
mathjax: true
---

## 二叉树

 - 二叉树 (binary tree)由<mark>结点的有限集合构成</mark>
    - 这个有限集合或者为空集 (empty)
    - 或者为由一个根结点 (root) 及两棵互不相交、分别称作这个根的左子树(left subtree)和右子树 (right subtree) 的二叉树组成的集合

<img src="/assets/images/2008/07/tree-1.jpg" style="margin-left:auto; margin-right:auto;display:block">

- 结点
    - 子结点、父结点、最左子结点
    - 兄弟结点、左兄弟、右兄弟
    - 分支结点、叶结点
        - 没有子树的结点称作 叶结点（或树叶、终端结点）
        - 非终端结点称为分支结点
- 边
    - 两个结点的有序对，称作边
    - 路径，路径长度
    - 祖先，后代
        - 若有一条由`k`到达`k(s)`的路径，则称`k`是`k(s)`的祖先，`k(s)`是`k`的子孙
- 层数
    - 根为第 0 层,其他结点的层数等于其父结点的层数加 1
    - 深度：层数最大的叶结点的层数
    - 高度：层数最大的叶结点的层数加 1

- 满二叉树和完全二叉树
    - 满二叉树
        - 结点是叶子结点
        - 内部结点有两个子节点
    - 完全二叉树
        - 若设二叉树的深度为h，除第h层外，其它各层是满的
        - 第h层如果不是满的，则子节点都在最左边

- 扩充二叉树
    - 所有叶子结点变成内部结点，增加树叶，变成满二叉树
    - 所有扩充出来的结点都是叶子节点
    - 外部路径长度`E`和内部路径长度`I`满足：`E=I+2n(n是内部结点个数)`

### 二叉树性质

1. 在二叉树中，第i层上最多有 $2i (i≥0)$ 个结点
2. 深度为 k 的二叉树至多有 $2^{(k+1)}-1 (k≥0)$ 个结点
    - 其中深度(depth)定义为二叉树中层数最大的叶结点的层数
3. 一棵二叉树，若其终端结点数为$n_0$，度为$2$的结点数为$n_2$，则 $n_0=n_2+1$
4. <mark>满二叉树定理：非空满二叉树树叶数目等于其分支结点数加1</mark>
5. 满二叉树定理推论：一个非空二叉树的空子树数目等于其结点数加1
6. 有$n$个结点$(n>0)$的完全二叉树的高度为$⌈\log_2(n+1)⌉$，深度为$⌈\log_2(n+1)- 1⌉$

### ADT表示

```cpp
template<class T>
class BinaryTreeNode{
friend class BinaryTree<T>;
private:
    BinaryTreeNode<T> *left; // 指向左子树的指针
    BinaryTreeNode<T> *right; // 指向右子树的指针
    T info;
public:
    BinaryTreeNode();
    BinaryTreeNode(const T& ele);
    T value() const;
    BinaryTreeNode<T*>leftChild() const;
    BinaryTreeNode<T*>rightChild() const;
};
```

### 深度优先遍历

- 遍历是一种将树形结构专户为线性结构的方法
- 三种深度优先遍历
    - 前序法 (tLR次序，preorder traversal)。
        - 根结点->左子树->右子树。
        - 上图：`ABDCEGFHI`
    - 中序法 (LtR次序，inorder traversal)。
        - 左子数->根结点->右子树。
        - 上图：`DBAEGCHFI`
    - 后序法 (LRt次序，postorder traversal)。
        - 左子树->右子树->根结点
        - 上图：`DBGEHIFCA`


```cpp
//递归，前序遍历
template<class T>
void BinaryTree<T>::Recursive (BinaryTreeNode<T>* root){
    if(root!=NULL) {
        //Visit(root); //前序遍历
        Recursive(root->leftchild()); // 递归访问左子树
        //Visit(root); // 中序
        Recursive(root->rightchild()); // 递归访问右子树
        //Visit(root); // 后序
    }
｝
```
递归遍历是一种简洁并很好理解的算法，而且编译器也会在递归过程中做一些优化使其效率不会太差，但是对树层次很深的情况下，容易出现StackOverflow，此时可以将递归解法转为非递归解法

```cpp
//非递归，前序遍历
template<class T>
void BinaryTree<T>::None_Recursive_1(BinaryTreeNode<T>* root){
    stack<BinaryTreeNode<T>* > ss;
    BinaryTreeNode<T>* pointer = root;
    ss.push(NULL);// 栈底监视哨
    while(pointer){
        Visit(pointer->value()); //遍历节点
        if(pointer->rightchild()!=NULL){
            ss.push(pointer->rightchild()); //如果该节点有右子树，入栈
        }
        if(pointer->leftchild()!=NULL){ //循环遍历左子树
            pointer = pointer->leftchild();
        }else{
            pointer = ss.top(); //右子树
            ss.pop();
        }
    }
}
```

也可以通过判断栈是否为空作为循环条件

```cpp
template<class T>
void BinaryTree<T>::None_Recursive_2(BinaryTreeNode<T>* root){
    stack<BinaryTreeNode<T>* > ss;
    ss.push(node);
    while(!ss.empty()){
        BinaryTreeNode<T>* top = ss.top();
        Visit(top);
        ss.pop();
        if(top->right){
            ss.push(top->right); //先入栈右子树节点
        }
        if(top->left){ 
            ss.push(top->left); //后入栈右子树节点
        }
    }
}
```

- 时间复杂度
    - 在各种遍历中，每个结点都被访问且只被访问一次，时间代价为`O(n)`
    - 非递归保存入出栈（或队列）时间
        -  前序、中序，某些结点入/出栈一次， 不超过`O(n)`
        - 后序，每个结点分别从左、右边各入/出一次， `O(n)`
- 空间复杂度
    - 栈的深度与树的高度有关
        - 最好 O(log n)
        - 最坏 O(n) ，此时树退化为线性链表

### 广度优先遍历

从二叉树的第0层（根结点）开始，自上至下 逐层遍历；在同一层中，按照 从左到右 的顺序对结点逐一访问。例如上图中，广度优先遍历的顺序为:`ABCDEFGHI`

```cpp
template<class T>
void BinaryTree<T>::LevelOrder (BinaryTreeNode<T>* root){
    queue<BinaryTreeNode<T>*> qq; //广搜使用队列
    BinaryTreeNode<T>* pointer = root;
    qq.push(pointer);
    while(!qq.empty()){
        pointer = qq.front();
        qq.pop();
        Visit(pointer->value());
        if(pointer->leftchild()){ //左子树入队
            qq.push(pointer->leftchild());
        }
        if(pointer->rightchild()){
            qq.push(pointer->rightchild()); //右子树入队
        }
    }
}
```

- 时间复杂度
    - 在各种遍历中，每个结点都被访问且只被访问一次，时间代价为O(n)
    - 非递归保存入出栈（或队列）时间    
        -  宽搜，正好每个结点入/出队一次，`O(n)`
- 空间复杂度
    - 与树的最大宽度有关
        - 最好 `O(1)`
        - 最坏 `O(n)`

### 二叉树的存储结构

二叉树的各结点随机地存储在内存空间中，结点之间的逻辑关系用指针来链接。

- 二叉链表
    - left,right两个指针指向左右两个子树
    - `left - info - right`
    
- 三叉链表
    - left,right,指向左右两个子树
    - parent指向父节点
    - `left-info-parent-right`

- 由根节点和叶子节点定位父节点

```cpp
template<class T>
BinaryTreeNode<T>* Parent(BinaryTreeNode<T>* root, BinaryTreeNode<T>* current){
    BinaryTreeNode<T>* ret = NULL;
    //前序遍历搜索
    if(root == NULL){
        return NULL;
    }
    if(root->left == current || root->right == current){
        return root;
    }else{
        tmp = Parent(root->left, current); //左子树
        if(tmp){
            return tmp;
        }
        tmp = Parent(root->right, current); //右子树
        if(tmp){
            return tmp;
        }
        return NULL;
    }
}
```

- 二叉链表的空间开销分析
    - 存储密度$\alpha$表示数据结构存储的效率
    - 结构性开销 $\gamma=1-\alpha$
        - 有效数据外的辅助信息
    
    $$
    \alpha=\frac{数据本身存储量}{整个结构占用的存储总量}
    $$

    - 以满二叉树为例，满二叉树的一半指针域为空
        - 每个节点存在两个指针，一个数据域
            - 总空间: $(2p+d)n$
            - 结构性开销: $pdn
            - 如果$p=d$，那么结构性开销为$2p/(sp+d)=2/3$
    - 可见满二叉树存储效率并不高，有三分之二的结构性开销

### 完全二叉树的顺序存储

- 由于完全二叉树的结构，可以将二叉树结点按一定的顺序存储到一片连续的存储单元，使结点在序列中的位置反映出相应的结构信息
    - 存储结构实现性的
        - 如下图的完全二叉树，其存储结构为`|3|16|7|23|37|10|21|20|`
        - 我们可以根据一维数组的下标来定位节点的位置
    - 逻辑结构上仍然是二叉树结构 

<img src="/assets/images/2008/07/tree-4.jpg" style="margin-left:auto; margin-right:auto;display:block">

- 下标公式
    - 当`2i+1<n`时，结点`i`的左孩子是结点`2i+1`，否则结点i没有左孩子
    - 当`2i+2<n` 时，结点`i`的右孩子是结点`2i+2`，否则结点i没有右孩子
    - 当`0<i<n` 时，结点`i`的父亲是结点`⌊(i-1)/2⌋`
    - 当`i`为偶数且`0<i<n`时，结点`i`的左兄弟是结点`i-1`，否则结点`i`没有左兄弟
    - 当`i`为奇数且`i+1<n`时，结点i的右兄弟是结点`i+1`，否则结点`i`没有右兄弟


### 堆

堆(heap)又被为优先队列(priority queue)，是一个经典的实现是完全二叉树。这样实现的堆成为二叉堆(binary heap)。它在完全二叉树上增加了一个要求：<mark>任意节点的优先级不小于它的子节点</mark>。

