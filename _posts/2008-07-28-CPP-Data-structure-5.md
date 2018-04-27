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
    - 满二叉树(左图)
        - 结点是叶子结点
        - 内部结点有两个子节点
    - 完全二叉树(右图)
        - 若设二叉树的深度为h，除第h层外，其它各层是满的
        - 第h层如果不是满的，则子节点都在最左边

<img src="/assets/images/2008/07/tree-2.png" style="margin-left:auto; margin-right:auto;display:block">

- 扩充二叉树
    - 所有叶子结点变成内部结点，增加树叶，变成满二叉树
    - 所有扩充出来的结点都是叶子节点
    - 外部路径长度`E`和内部路径长度`I`满足：`E=I+2n(n是内部结点个数)`

### 二叉树性质

- 性质1： 在二叉树中，第i层上最多有 $2i (i≥0)$ 个结点
- 性质2： 深度为 k 的二叉树至多有 $2^(k+1)-1 (k≥0)$ 个结点
    - 其中深度(depth)定义为二叉树中层数最大的叶结点的层数
- 性质3： 一棵二叉树，若其终端结点数为$n$，度为$2$的结点数为$n_2$，则 $n_0=n_2+1$
- 性质4. <mark>满二叉树定理：非空满二叉树树叶数目等于其分支结点数加1</mark>
- 性质5. 满二叉树定理推论：一个非空二叉树的空子树数目等于其结点数加1
- 性质6. 有$n$个结点$(n>0)$的完全二叉树的高度为$⌈\log_2(n+1)⌉$，深度为$⌈\log_2(n+1)- 1⌉$

### ADT

- 结点

```cpp
template<class T>
class BinaryTreeNode{
friend class BinaryTree<T>;
private:
    T info;
public:
    BinaryTreeNode();
    BinaryTreeNode(const T& ele);
    T value() const;
    BinaryTreeNode<T*>leftChild() const;
    BinaryTreeNode<T*>rightChild() const;
};
```
- 树

```cpp
template <class T>
class BinaryTree {
private:
    BinaryTreeNode<T>* root; // 二叉树根结点
public:
    BinaryTree() {root = NULL;}; // 构造函数
    ~BinaryTree() {DeleteBinaryTree(root);}; // 析构函数
    ...

    void PreOrder(BinaryTreeNode<T> *root); // 前序遍历二叉树或其子树
    void InOrder(BinaryTreeNode<T> *root); // 中序遍历二叉树或其子树
    void PostOrder(BinaryTreeNode<T> *root); // 后序遍历二叉树或其子树
    void LevelOrder(BinaryTreeNode<T> *root); // 按层次遍历二叉树或其子树
    void DeleteBinaryTree(BinaryTreeNode<T> *root); // 删除二叉树或其子树
}
```

### 遍历二叉树

- 遍历是一种将树形结构专户为线性结构的方法
- 三种深度优先遍历
    - 前序法 (tLR次序，preorder traversal)。
        - 访问根结点； 按前序遍历左子树； 按前序遍历右子树。
    - 中序法 (LtR次序，inorder traversal)。
        - 按中序遍历左子树； 访问根结点； 按中序遍历右子树。
    - 后序法 (LRt次序，postorder traversal)。
        - 按后序遍历左子树； 按后序遍历右子树； 访问根结点

![](/assets/images/2008/07/tree-3.png)