---
layout: post
title: Data Structure Part 6 - Tree
mathjax: true
---

## 树

前一节介绍了二叉树，它是一种特殊的树形结构，对于更加广义的“树”有以下几个特性：

1. 使用集合来描述
    - 树(tree)是包括n个结点的有限集合T(n≥1)
        - 有一个特定的结点，称为根
        - 除了根之外的结点被划分成m个集合，集合之间互不相交，每个集合可以看成一课独立的子树
    - <mark>我们说树，通常指的是多个集合构成的森林，并非一棵单独的树</mark>
2. 子树是有序有向的
    - 子树自身，以及子树之间的相对次序是重要的
        - 例如，度为2的有序树并不是二叉树。因为第一子结点被删除后,第二子结点自然顶替成为第一
    - 子树是一种有向无环的图，可以用集合论中的符号表示树形结构
        - 例如，父节点A有两个子节点B，C，可以表示为`<A,B>`,`<A,C>`，两个有序对，其中根节点在逗号前面
3. 森林
    - 零棵或者多棵不相交的树的集合
        - 一棵树，删掉树根，其子树就构成了森林
        - 加入一个结点作为根，森林就转化成了一棵树

### 森林和二叉树

- 森林转二叉树

假设现在有一个森林，由三个集合组成，记作：$F={T_1,T_2,T_3}$。现在希望将这个森林转换成一颗二叉树$B(F)$，规则如下：

1. $B(F)$的根是$T_1$的根
2. $B(F)$的左子树为$T_1$除去根节点后的子树组成的二叉树3
3. $B(F)$的右子树为${T_2,T_3}$组成的二叉树。

<img src="/assets/images/2008/08/tree-10.jpg" style="margin-left:auto; margin-right:auto;display:block">


如上图所示，不难发现将上面虚线部分是一个递归合并的过程(将两棵子树合并成一课二叉树，而每棵子树又有自己的子树)。对于任意两棵子树，其合并的过程为：

1. 在森林中的所有兄弟结点之间加一连线 
2. 对每个结点，去掉除了与第一个孩子之外的其他所有连线
3. 调整位置

<img src="/assets/images/2008/08/tree-11.jpg" style="margin-left:auto; margin-right:auto;display:block">

 - 二叉树转森林

类似的，设$B$是一棵二叉树，root是$B$的根，$B_L$ 是root的左子树，$B_R$ 是root的右子树，则对应于二叉树$B$的森林或树$F(B)$的规则为:

1. 若$B$为空，则$F(B)$是空的森林
2. 若$B$不为空，则$F(B)$是一棵树$T_1$加上森林$F(B_R)$其中树$T_1$ 的根为root，root的子树为$F(B_L
)$

对于二叉树的任意一个非叶子节点，将其拆成森林的过程为：

1. 将除根结点以外的右子结点都与根节点相连
2. 去掉所有父节点和其右子结点的连线
3. 调整位置

<img src="/assets/images/2008/08/tree-12.jpg" style="margin-left:auto; margin-right:auto;display:block">

### ADT

- 树节点

```cpp
template<class T>
class TreeNode { // 树结点的ADT
public:
 TreeNode(const T& value); // 拷贝构造函数
 virtual ~TreeNode() {}; // 析构函数
 bool isLeaf(); // 判断当前结点是否为叶结点
 T Value(); // 返回结点的值
 TreeNode<T> *LeftMostChild(); // 返回第一个左孩子
 TreeNode<T> *RightSibling(); // 返回右兄弟
 void setValue(const T& value); // 设置当前结点的值
 void setChild(TreeNode<T> *pointer); // 设置左孩子
 void setSibling(TreeNode<T> *pointer); // 设置右兄弟
 void InsertFirst(TreeNode<T> *node); // 以第一个左孩子身份插入结点
 void InsertNext(TreeNode<T> *node); // 以右兄弟的身份插入结点
};
```
对于树节点，我们只需要关心两个指针，分别是第一个左孩子结点和它的右边兄弟节点，原因是当我们要将其转为二叉树时，需要用到这两个节点，同样的，当我们要删除某个兄弟结点时，也需要更新这个指针

- 树

```cpp
template<class T>
class Tree {
public:
 Tree(); // 构造函数
 virtual ~Tree(); // 析构函数
 TreeNode<T>* getRoot(); // 返回树中的根结点
 void CreateRoot(const T& rootValue); // 创建值为rootValue的根结点
 bool isEmpty(); // 判断是否为空树
 TreeNode<T>* Parent(TreeNode<T> *current); // 返回父结点
 TreeNode<T>* PrevSibling(TreeNode<T> *current); //返回前一个兄弟
 void DeleteSubTree(TreeNode<T> *subroot); // 删除以subroot子树
 void RootFirstTraverse(TreeNode<T> *root); // 先根深度优先遍历树
 void RootLastTraverse(TreeNode<T> *root); // 后根深度优先遍历树
 void WidthTraverse(TreeNode<T> *root); // 广度优先遍历树
};
```

### 遍历