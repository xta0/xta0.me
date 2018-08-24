---
layout: post
list_title: Data Structure Part 5-1 | 堆与霍夫曼树 | Heap & Huffman Tree
mathjax: true
title: 堆与霍夫曼树
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

## 霍夫曼(Huffman)树

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

1. 给出一个具有$n$个外部节点($n$个待编码字符)的扩充二叉树
2. 令该二叉树每个外部节点$K_i$ 有一个权 $w_i$ 外部路径长度为$l_i$ 
3. 使这个扩充二叉树的<mark>叶节点带权外部路径长度总和</mark>为$\sum_{i=0}^{n-1}l_i * w_i $ 最小

我们把具有这样性质的树叫做Huffman树，它是一种带权路径长度最短的二叉树，也称为最优二叉树。如下图中，左边树构建方式的带权路径和为：(2+7+24+32)x2 = 130, 右边树构建方式的带权路径和为：32x1+24x2 +(2+7)x3 = 107。

<img src="{{site.baseurl}}/assets/images/2008/07/tree-10.png" style="margin-left:auto; margin-right:auto;display:block">

- 构建Huffman树

> 贪心法

1. 将节点按照权值从小到大排列
2. 拿走前两个权值最小的节点，创建一个节点，权值为两个节点权值之和，将两个节点挂在新节点上
3. 将新节点的权值放回序列，使权值顺序保持
4. 重复上述步骤，直到序列处理完毕
5，所有待编码的节点都处在叶子位置

假设有一组节点的优先级序列为: `2 3 5 7 11 13 17 19 23 29 31 37 41`，对应的Huffman树为：

<img src="{{site.baseurl}}/assets/images/2008/07/tree-9.jpg" style="margin-left:auto; margin-right:auto;display:block">

- 编码

有了Huffman树，就可按路径对叶子节点(待编码字符)进行霍夫曼编码，例如上图中的每个叶子节点的编码为：

```
d0 ： 1011110   d1 ： 1011111
d2 ： 101110    d3 ： 10110
d4 ： 0100      d5 ： 0101
d6 ： 1010      d7 ： 000
d8 ： 001       d9 ： 011
d10： 100       d11 ：110  d12： 111
```

```cpp
//前序遍历，得到每个叶子节点的霍夫曼编码
void traverse(HuffmanTreeNode* root, vector<int>& codes){
    
    if(root&&root->left==NULL&&root->right==NULL){
        for(auto num : codes){
            cout<<num<<" ";
        }
        cout<<endl;
    }
    
    if(root->left){
        codes.push_back('0');
        traverse(root->left,codes,dictionary);
        codes.pop_back(); //回溯
    }
    
    if(root->right){
        codes.push_back('1');
        traverse(root->right, codes,dictionary);
        codes.pop_back();//回溯
    }
};
```

- 译码

与编码过程相逆，从左至右逐位判别代码串，直至确定一个字符

1. 从树的根节点开始
    - `0`下降到左分支
    - `1`下降到右分支
2. 到达一个树叶节点，对应的字符就是文本信息的字符
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

- [CS106B-Stanford-YouTube](https://www.youtube.com/watch?v=NcZ2cu7gc-A&list=PLnfg8b9vdpLn9exZweTJx44CII1bYczuk)
- [Algorithms-Stanford-Cousera](https://www.coursera.org/learn/algorithms-divide-conquer/home/welcome)
- [算法与数据结构-1-北大-Cousera](https://www.coursera.org/learn/shuju-jiegou-suanfa/home/welcome)
- [算法与数据结构-2-北大-Cousera](https://www.coursera.org/learn/gaoji-shuju-jiegou/home/welcome)
- [算法与数据结构-1-清华-EDX](https://courses.edx.org/courses/course-v1:TsinghuaX+30240184.1x+3T2017/course/)
- [算法与数据结构-2-清华-EDX](https://courses.edx.org/courses/course-v1:PekingX+04833050X+1T2016/course/)
- [算法设计与分析-1-北大-Cousera](https://www.coursera.org/learn/algorithms/home/welcome)
- [算法设计与分析-2-北大-EDX](https://courses.edx.org/courses/course-v1:PekingX+04833050X+1T2016/course/)


