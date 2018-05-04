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

### 二叉搜索树

二叉搜索树（Binary Search Tree）是具有下列性质的二叉树：

1. 对于任意节点，设其值为`K`
2. 该结点的左子树(若不空)的任意一个结点的值都小于`K`
3. 该结点的 右子树(若不空)的任意一个结点的值都大于`K`
4. 该节点的左右子树也是BST

<img src="/assets/images/2008/07/tree-5.jpg" style="margin-left:auto; margin-right:auto;display:block">

- 搜索

1. 二叉搜索树的<mark>中序遍历</mark>即是节点的正序排列（从小到大排列）。
2. 假如我们要搜索`19`，每次只需要检索两个子树之一，直到`19`被被找到，或者走到叶子节点停止，类似二分法

- 插入节点

1. 从根节点开始搜索，在停止位置插入一个新叶子节点。
2. 假如我们要插入`17`，如下图搜索树，直到遇到`19`搜索停止，`17`成为`19`左叶子节点。
3. 插入新节点后的二叉树依然保持BST的性质和性能，插入时间复杂度为`O(logN)`

<img src="/assets/images/2008/07/tree-6.jpg" style="margin-left:auto; margin-right:auto;display:block">

- 删除节点

1. 如果该节点没有左子树，则使用右子树替换该节点
2. 如果该节点没有右子树，则使用左子树替换该节点
3. 如果该节点既有左子树也有右子树，则找到右子树中最小的节点，将该节点的值替换为待删除节点的值，删除该节点

```cpp
{
    BinaryTreeNode <T> * temp = rt;
    if (rt->leftchild() == NULL) {
        rt = rt->rightchild();
    }else if (rt->rightchild() == NULL) {
        rt = rt->leftchild();
    }else {
        //找到右子树中最小节点
        temp = deletemin(rt->rightchild()); //helper method
        rt->setValue(temp->value());
    }
    delete temp;
}

template<class T>
BinaryTreeNode<T>* deleteMin(BinaryTreeNode<T>* rt){
    if(rt->leftChild()!=NULL){
        return deletemin(rt->leftchild()) //在右子树中递归找到左边叶子节点
    }else{
        BinaryTreeNode<T>* temp = rt; //保留找到的叶子节点
        rt = rt->rightchild(); //将该节点替换为
        return temp;
    }
}
```

- 组织内存索引
    - 二叉搜索树是适用于内存储器的一种重要的树形索引
        - 常用红黑树、伸展树等，以维持平衡  
    -  外存常用B/B+树

### 堆

堆是<mark>完全二叉树</mark>的一种表现形式。以最小堆为例，它要求的每个父节点的值大于两个子节点的值，两个兄弟节点之间的值的大小关系没有限制。由于完全二叉树可以用数组表示，上述性质也可以表述为：

1. $K_1<=K_{2i+1}$
2. $K_1<=K_{2i+2}$

因此使用最小堆可以找出这组节点的最小值。推而广之，对于一组无序的数，可以将他们构建成堆来快速得到最大值或最小值，当有新元素进来时，堆也依然可以保持这种特性

<img src="/assets/images/2008/07/tree-7.jpg" style="margin-left:auto; margin-right:auto;display:block">

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
3. 从倒数第二层，`i=⌊n/2-1⌋` 开始进行递归SiftDown调整。例如下图中，需要从`23`开始进行递归调整，调整完毕后是`71`,`73`,`72`


<img src="/assets/images/2008/07/tree-8.jpg" style="margin-left:auto; margin-right:auto;display:block">


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
    - 大约一半的结点深度为$d-1$，不移动（叶）。
    - 四分之一的结点深度为$d-2$，而它们至多能向下移动一层。
    - 树中每向上一层，结点的数目为前一层的一半，而子树高度加一。
    - 因此元素移动的最大距离的总数为：

    $$\sum_{i=1}^{\log{n}}(i-1)\frac{n}{2^i}=O(n)$$

- 堆操作时间复杂度分析
    - <mark> 建堆算法时间代价为$O(n)$</mark>
    - 堆有$\log{n}$层深
    - 插入结点、删除普通元素和删除最小元素的<mark>平均时间代价</mark>和<mark>最差时间代价</mark>都是$O(\log{n})$

- 优先队列
    - 根据需要释放具有最小（大）值的对象
    - 最大树、 左高树HBLT、WBLT、MaxWBLT
    - 改变已存储于优先队列中对象的优先权
        - 辅助数据结构帮助找到对象

### 霍夫曼(Huffman)树

霍夫曼树是一种特殊的二叉树组织形式，用来实现霍夫曼编码，霍夫曼编码是一种不等长编码技术。所谓非等长编码，是根据字符出现的频率高低对字符进行不同位数的编码，一般希望经常出现的字符编码比较短，不经常出现的字符编码较长。

```
Z  K  F  C  U  D  L  E
2  7  24 32 37 42 42 120
```

例如上面字符中，每个下面为其对应出现的频率。我们可以对这些字符进行编码，例如`Z=111100`，`K(111101)`。每个字符的编码需要满足一个条件，即编码规则为<mark>前缀编码</mark>。所谓前缀编码要求任何一个字符的编码都不是另外一个字符编码的前缀，这种前缀特性保证了译码的唯一性。

```
Z(111100),K(111101),
F(11111),C(1110),
U(100),D(101), 
L(110),E(0)
```
假设我们已经有了上述的编码表，对于`000110`可以翻译出唯一的字符串`EEEL`。因此我们需要寻找一种编码方法使得：

1. 频率高的字符占用存储空间少
2. 编码规则符合前缀码要求

将这个问题抽象一下，对于$n$个字符$K_0$，$K_1$，…，$K_{n-1}$，它们的使用频率分别为$w_0$, $w_1$，…，$w_{n-1}$，给出它们的前缀编码，使得总编码效率最高

- 霍夫曼树

为了实现上述编码规则，我们需要设计这样一棵树：

1. 给出一个具有$n$个外部结点($n$个待编码字符)的扩充二叉树
2. 令该二叉树每个外部结点$K_i$ 有一个权 $w_i$ 外部路径长度为$l_i$ 
3. 使这个扩充二叉树的<mark>叶结点带权外部路径长度总和</mark>为$\sum_{i=0}^{n-1}l_i * w_i $ 最小

我们把具有这样性质的树叫做Huffman树，它是一种带权路径长度最短的二叉树，也称为最优二叉树。如下图中，左边树构建方式的带权路径和为：(2+7+24+32)x2 = 130, 右边树构建方式的带权路径和为：32x1+24x2 +(2+7)x3 = 107。

<img src="/assets/images/2008/07/tree-10.png" style="margin-left:auto; margin-right:auto;display:block">

- 构建Huffman树

> 贪心法

1. 将节点按照权值从小到大排列
2. 拿走前两个权值最小的节点，创建一个节点，权值为两个节点权值之和，将两个节点挂在新结点上
3. 将新节点的权值放回序列，使权值顺序保持
4. 重复上述步骤，直到序列处理完毕
5，所有待编码的节点都处在叶子位置

假设有一组节点的优先级序列为: `2 3 5 7 11 13 17 19 23 29 31 37 41`，对应的Huffman树为：

<img src="/assets/images/2008/07/tree-9.jpg" style="margin-left:auto; margin-right:auto;display:block">

- 编码

有了Huffman树，就可按路径对叶子节点(待编码字符)进行编码

```
d0 ： 1011110
d1 ： 1011111
d2 ： 101110 
d3 ： 10110
d4 ： 0100 
d5 ： 0101
d6 ： 1010 
d7 ： 000
d8 ： 001 
d9 ： 011
d10： 100 
d11 ： 110
d12： 111
```

- 译码

与编码过程相逆，从左至右逐位判别代码串，直至确定一个字符

1. 从树的根结点开始
    - `0`下降到左分支
    - `1`下降到右分支
2. 到达一个树叶结点，对应的字符就是文本信息的字符
3. 译出了一个字符，再回到树根，从二进制位串中的下一位开始继续译码

```
译码: 111101110

111 -> d12
101110 -> d0

111101110 ->d12d0
```

- 构建Huffman树

```cpp
HuffmanTree(vector<int> weight, TreeNode* root;){
    HuffmanTreeNode* root; 
    HuffmanTree(vector<int>& weight){
        priority_queue<HuffmanTreeNode*,std::vector<HuffmanTreeNode* >,comp> heap; //最小堆
        size_t n = weight.size();
        for(int i=0; i<n; ++i){
            HuffmanTreeNode* node = new HuffmanTreeNode();
            node->weight = weight[i];
            node->left = node->right = node->parent = NULL;
            heap.push(node);
        }
        for(int i=0; i<n-1; i++){ //两个节点合成一个，共有n-1次合并建立Huffman树
            HuffmanTreeNode* first = heap.top();
            heap.pop();
            HuffmanTreeNode* second = heap.top();
            heap.pop();
            HuffmanTreeNode* parent = new HuffmanTreeNode();
            first->parent = parent;
            second->parent = parent;
            parent->weight = first->weight + second->weight;
            parent->left  = first;
            parent->right = second;
            heap.push(parent);
            root = parent;
        }
}
```

- 编码效率

设平均每个字符的代码长度等于每个代码的长度$c_i$, 乘以其出现的概率$p_i$ ，即:

$$ l = c_0p_0 + c_1p_1 + … + c_{n-1}p_{n-1} $$

其中$p_i=f_i / f_T$, $f_i$为第$i$个字符的出现的次数，而$f_T$为所有字符出现的总次数。因此前面式子也可以写为

$$ c_0f_0 + c_1f_1 + … + c_{n-1}f_{n-1}) / f_T $$

则上图中的平均码长度为

$$ (3*(19+23+24+29+31+34+37+41)+4*(11+13+17)+ 5 * 7+6 * 5+7*(2+3)) / 238 = 804 / 238 ≈ 3.38 $$

对于这13个字符，如果采用等长编码每个字符需要$⌈log13⌉=4$位，显然Huffman编码效率更高，只需要等长编码$3.38/4≈ 84\% $ 的空间

- 应用
    - Huffman编码适合于字符<mark>频率不等，差别较大</mark>的情况
    - 数据通信的二进制编码
        - 不同的频率分布，会有不同的压缩比率
        - 大多数的商业压缩程序都是采用几种编码方式以应付各种类型的文件
            - Zip 压缩就是 LZ77 与 Huffman 结合
    - 归并法外排序，合并顺串


### Resources

- [How Gzip uses Huffman coding](https://jvns.ca/blog/2015/02/22/how-gzip-uses-huffman-coding/)