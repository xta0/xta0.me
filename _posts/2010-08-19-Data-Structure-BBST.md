---
layout: post
list_title: "" | 平衡二叉搜索树 | Balanced BST
mathjax: true
title: 平衡二叉搜索树
categories: [DataStructure]
---

## BST的平衡性

前面介绍的关于BST的操作其性能受BST的结构影响很大，例如，如果随机的给一组数据，得到的BST可能是平衡的，其高度为$\log{N}$，其检索时间会达到最好的$O(log_2{N})$；而如果有序的给一组数，则BST会退化为线性链表，其检索的时间复杂度则退化为$O(N)$。

<img src="{{site.baseurl}}/assets/images/2008/07/tree-14.jpg" style="margin-left:auto; margin-right:auto;display:block">

从平均的意义而言，在BST上的一系列操作的时间是正比于BST树的高度，而BST的树高和关键码输入次序相关，显然为了降低复杂度，我们希望树的高度越低越好。

不难发现，当节点数目固定时，兄弟子树高度越接近平衡，全树也将倾向于更低。具体来说，由$N$个节点组成的二叉树，高度不低于$\lfloor \log_2{N} \rfloor$，如果高度恰好为$\lfloor \log_2{N} \rfloor$，则称为**理想平衡**。

如果仔细回想，我们在介绍二叉树时曾提到过的完全二叉树和满二叉树，这两种树都是平衡二叉树，但实际应用中，这种树往往可遇而不可求，即是遇到，经过一系列的动态操作，平衡性也会被破坏，可以说理想平衡出现的概率极低，维护的成本过高。因此，我们要追求的是一种适度的平衡，即在渐进的意义上不超过$O(\log_2{N})$即可。适度平衡的BST，也称作**平衡二叉搜索树(Balanced BST)**。

### BST的旋转变换

如果我们使用中序遍历来恢复一棵BST，会发现其往往存在歧义性，因为其结果往往不唯一，如下图所示的两棵BST，其中序遍历序列相同，但是拓扑结构却不同

<img src="{{site.baseurl}}/assets/images/2008/07/tree-12.jpg" style="margin-left:auto; margin-right:auto;display:block">

对于上面这样的两棵BST，我们称其为等价的BST，等价的BST虽然在拓扑结构上不同，但却有规律可循：

1. 上下可变：连接关系不尽相同，承袭关系可能颠倒，在垂直方向上有一定的自由度
2. 左右不乱：中序遍历序列完全一致，全局单调非降

基于上面规律，我们怎么将一棵不平衡的BST调整为与其等价的且平衡的BST呢？答案是使用一种对节点的**旋转操作**，也叫做所谓的**ZigZag**操作，如下图所以，左图为对某节点的右旋操作(Zig)，右图为对某节点的左旋操作(Zag):

<img src="{{site.baseurl}}/assets/images/2008/07/tree-13.jpg" style="margin-left:auto; margin-right:auto;display:block">

参照上图，可以大致理解为，BST以节点`V`，进行右旋(左图`Zig`)，或者以节点`V`进行左旋(右图`Zag`)后得到的树是和原树等价的。有了这两种方式，我们即可将一棵非平衡的BST转化为BBST，但值得注意的是，这种旋转操作，最多只针对两个节点`V,C`进行操作，且时间在控制在常数。

有了这种调整的策略，我们便可对BST的各种操作进行优化，即使某些动态操作使BST变得不平衡，仍可通过上面的变换，以极小的代价（不超过$O(\log_2{N})$）将其还原为BBST，进而降低各种操作的时间复杂度。例如，后面介绍的AVL树，可将插入操作优化为$O(1)$，而红黑树更可以将插入删除均优化为$O(1)$。在接下来将要介绍的AVL树，伸展树中我们将会应用这项技术来调整BST的平衡性。

## AVL树

AVL树是一种经典的自平衡树，由Adelson-Velskii 和 Landis在1962年提出。所谓AVL树，是指树中的每个节点平衡因子（左右子树的高度差）都不超过1，所谓平衡因子的定义如下：

$$
bf(x) = height(x_{lchild}) - height(x_{rchild})
$$

节点的平衡因子可能取值为0,1（`+`）和-1（`-`）。

<img src="{{site.baseurl}}/assets/images/2008/07/tree-15.jpg" style="margin-left:auto; margin-right:auto;display:block">

可以证明，AVL树是适度平衡的，`height(AVL) = O(logn)`,即$N$个节点的AVL树，其高度是不超过$log{N}$的。

### ADT接口

```cpp
#define Balanced(node) \ //理想平衡
    ( height(node->left) == height(node->right) )
#define BF(node) \ //平衡因子
    ( height(node->left) - height(node->right) )
#define AVLBalanced(node) \ //AVL平衡条件
    ( (-2 < BF(node) && (BF(node)<2) )

//和BST不同的是插入和删除的动态操作
void insert(TreeNode* root, int target);
TreeNode* remove(TreeNode* root, int target);
```

另外，为了后续算法的便捷性，我们还需要在原来二叉树的节点中增加一个对父节点的引用

```cpp
template <class T>
struct BinaryTreeNode{
    T val;
    BinaryTreeNode<T>* left;
    BinaryTreeNode<T>* right
    //维护一个parent指针
    BinaryTreeNode<T>* parent;
}
//后续例子中将以int做为基本类型
typedef TreeNode BinaryTreeNode<int>;
```

### 失衡与重平衡

正如前面介绍的，AVL树可以自平衡，所谓自平衡就是在有数据增删的情况下，原有平衡结构被打破后，BST需要具有能重新维持平衡的能力。针对AVL树来说，考虑下面对某个节点增删的情况：

<img src="{{site.baseurl}}/assets/images/2008/07/tree-16.jpg" style="margin-left:auto; margin-right:auto;display:block">


如右图，当`M`插入后，虽然不会引起其父亲节点`K`的失衡，却会导致其祖先节点`N,R,G`相继失衡，而除了祖先之外的其它节点则不受影响。同理，左图中，当删除`Y`后，只会影响到其父节点`R`，对其它祖先节点则无影响。

一般的，AVL树中如果删除某个节点，受影响的节点最多有$O(1)$个；如果插入某个节点，受影响的节点数为$O(log{N})$。在实现上，删除操作相比插入操作则更为复杂。

### 插入

- 单旋

首先来考察待插入节点和父节点`P`，祖父节点`G`（所有失衡祖先中，最低的一个）同侧的情况，如下图所示，假设插入点在`V`的下方。这种情况我们只需要对`G`节点进行左旋操作（zag），其步骤为： 

1. 创建临时指针`Ptr`指向`P`
2. `P`的左子树`T1`成为`G`的右子树
3. `G`成为`P`的左孩子
4. 将局部子树的根从`G`替换为`P`

<img src="{{site.baseurl}}/assets/images/2008/07/tree-17.jpg" style="margin-left:auto; margin-right:auto;display:block">

`G` 经过单选调整后，恢复平衡，子树高度复原，<mark>更高的祖先也必平衡，全树复衡</mark>。zag旋转的时间复杂度为$O(1)$，由于所有节点都在右侧，因此这种左旋操作也叫做**zagzag**，同理，如果所有节点均在左侧，则需要进行右旋操作，也叫**zigzig**

- 双旋

如果`G`和`P`和`V`的朝向并不一致，则需要进行双旋操作，也叫**zigzag**或者**zagzig**，具体步骤为

1. 对`V`做zig右旋，过程与上面zag旋转相仿
2. 对`G`做zag左旋，过程参照上面单旋

<img src="{{site.baseurl}}/assets/images/2008/07/tree-18.jpg" style="margin-left:auto; margin-right:auto;display:block">

- 实现

```cpp
void insert(TreeNode* root, int target){
    ...
    //找到带插入位置
    //AVL逻辑
    TreeNode* parent = root->parent;
    while(parent){
        //发现最近祖先失衡        
        if(!AVLBalanced(parent)){
            //AVL调整
            rotate(parent);
            break; //G回复后，整棵树平衡，调整结束
        }
        parent = parent->parent;
    }
}
```

## 伸展树

BST的另一个改进变种是伸展树，伸展树是一种自适应的数据结构，它的引入主要是用来解决局部性问题，所谓局部性问题是指：刚被访问过的数据，极有可能很快的再次被访问，对应到BST上，刚刚访问过的节点，极有可能很快的再次被访问到

> 伸展树的一个经典的应用是输入法词表，输入法词表会根据用户的选择来调整词表顺序，例如用拼音输入"dasan"，可能的联想有"打伞","打散","大三"，当用户做出选择后，在下一次检索时，用户选择的词就直接出现在词表的第一项（树根位置）

由上面介绍的AVL树可知，每次查找某个节点所用的时间为$O(\log{N})$，如果连续查找$m$次，则需要$O(m\log{N})$时间。如果对查找的某些节点具有局部性，我们可以找到某种方法使得对这些节点的访问更高效。

- 自适应伸展

一种简单的自适应策略是，如果某种元素刚被访问，将这个元素至于序列或者树的最前端，则经过一段时间的使用后，所有被频繁访问的元素都将聚集于序列或者树的前端位置。

对应到BST，任何节点`v`，一旦被访问，随机转移至树根，具体策略为：

1. 如果`v`是`P`的左孩子，则对`P`进行右旋`zig(p)`
2. 如果`v`是`P`的右孩子，则对`P`进行左旋`zag(p)`

<img src="{{site.baseurl}}/assets/images/2008/07/tree-19.jpg" style="margin-left:auto; margin-right:auto;display:block">

可见不论`V`在左还是右，通过旋转都可以让`V`上升一层，因此可以对`V`进行反复旋转，直到其转移至树根，这个过程中，`V`每前进磁层都要对整棵树进行伸展，这就是所谓伸展树的来历。但是上述过程存在一种最坏情况，即当BST为单链表结构时



### Resources

- [CS106B-Stanford-YouTube](https://www.youtube.com/watch?v=NcZ2cu7gc-A&list=PLnfg8b9vdpLn9exZweTJx44CII1bYczuk)
- [Algorithms-Stanford-Cousera](https://www.coursera.org/learn/algorithms-divide-conquer/home/welcome)
- [算法与数据结构-1-北大-Cousera](https://www.coursera.org/learn/shuju-jiegou-suanfa/home/welcome)
- [算法与数据结构-2-北大-Cousera](https://www.coursera.org/learn/gaoji-shuju-jiegou/home/welcome)
- [算法与数据结构-1-清华-EDX](https://courses.edx.org/courses/course-v1:TsinghuaX+30240184.1x+3T2017/course/)
- [算法与数据结构-2-清华-EDX](https://courses.edx.org/courses/course-v1:PekingX+04833050X+1T2016/course/)
- [算法设计与分析-1-北大-Cousera](https://www.coursera.org/learn/algorithms/home/welcome)
- [算法设计与分析-2-北大-EDX](https://courses.edx.org/courses/course-v1:PekingX+04833050X+1T2016/course/)



