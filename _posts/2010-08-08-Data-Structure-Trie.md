---
layout: post
list_title: 数据结构基础 | Data Structure | 前缀树 | Trie
mathjax: true
title: Trie
categories: [DataStructure]
---

### 字符树

对于前面提到的BST，当输入是随机的情况下，可能达到理想的查询速度`O(log(N))`，但是如果输入是有序的，则BST会退化为单链表，查询速度会降为`O(N)`。因此我们需要思考，对于BST的构建，能否不和数据的输入顺序相关，而和数据的空间分布相关，这样只要数据的空间分布是随机的，那么构建出的BST查询性能就会得到保证。

Trie树又称前缀树或字典树，它最早应用于信息检索领域，所谓"Trie"来自英文单词"Retrieval"的中间的4个字符"trie"。Trie的经典应用场景是搜索提示和分词，比如当我们在搜索栏中输入关键词时，地址栏会自动提示并补全关键字。在这个例子中，与之前BST搜索不同的是Trie中的节点保存的并不是key值，而是单个的英文字符，因此使用Trie检索某个字符串是否位于某个集合中，其过程是逐一检索字符串的每个前缀字符是否在Trie的节点中，而不是像BST一样直接比较字符串内容。如下图所示，我们可以将一组单词通过Trie树的形式进行构建，其中左图为等长字符串，每个字符均不是其它字符的前缀；右边为不等长字符串，每个字符可能是其它字符的前缀，这种情况我们需要在末尾增加一个`*`表示

<img src="{{site.baseurl}}/assets/images/2010/08/trie.png" class="md-img-center">

例如当我们想要查找字符串"bad"，我们只需要从Trie树的根节点开始依次匹配`bad`中的字符，直到遇到`*`表示匹配成功。

通过字符串查询的例子，我们可以看出，Trie树的特点是可以是对输入对象进行空间分解，一个节点的所有子孙都有相同的前缀，也就是这个节点对应的字符串，而根节点对应空字符串。一般情况下，不是所有的节点都有对应的值，只有叶子节点和部分内部节点所对应的键才有相关的值。

### Trie的实现

我们以字母树为例，介绍一种简单的Trie的实现方式。首先我们需要先定一个TrieNode的数据结构

```cpp
struct TrieNode{
    bool isEnd = false; //用来标识该节点是叶子节点
    array<TrieNode*,26> children = {nullptr};
};
```

我们使用一个长度为26的静态数组来存储其孩子节点，如果某个位置不为空则说明该位置有一个孩子节点，可参考下图：

<img src="{{site.baseurl}}/assets/images/2010/08/trie-1.png" class="md-img-center">

接下来我们可以实现Trie的类

```cpp
class Trie {
    TrieNode* root;
public:
    Trie() {
        root = new TrieNode();
    }
    ~Trie(){
        delete root;
        root = nullptr;
    }
    void insert(string word) {
        TrieNode* node = root;
        for(int i =0; i<word.size(); i++){
            int index = word[i]-'a';
            TrieNode* n = node->children[index];
            if(!n){
                node->children[index] = new TrieNode();
                node = node->children[index];
            }else{
                node = n;
            }
        }
        node->isEnd = true;
    }
    bool search(string word) {
        TrieNode* node  = root;
        for(int i=0;i<word.size();i++){
            int index = word[i]-'a';
            TrieNode* n = node->children[index];
            if(!n){
                return false;
            }else{
                node = n;
            }
        }
        return node->isEnd;
    }
    bool startsWith(string prefix) {
        TrieNode* node = root;
        for(int i=0;i<prefix.size();i++){
            int index = prefix[i]-'a';
            TrieNode* n = node->children[index];
            if(!n){
                return false;
            }else{
                node = n;
            }
        }
        return true;
    }
};
```
Trie的有点是最大限度减少无效的字符串比较，其核心思想史空间换时间。利用字符串的公共前缀来降低查询时间的开销，以达到提高效率的目的。

## Resources 

- [CS106B-Stanford-YouTube](https://www.youtube.com/watch?v=NcZ2cu7gc-A&list=PLnfg8b9vdpLn9exZweTJx44CII1bYczuk)
- [Algorithms-Stanford-Cousera](https://www.coursera.org/learn/algorithms-divide-conquer/home/welcome)
- [算法与数据结构-1-北大-Cousera](https://www.coursera.org/learn/shuju-jiegou-suanfa/home/welcome)
- [算法与数据结构-2-北大-Cousera](https://www.coursera.org/learn/gaoji-shuju-jiegou/home/welcome)
- [算法与数据结构-1-清华-EDX](https://courses.edx.org/courses/course-v1:TsinghuaX+30240184.1x+3T2017/course/)
- [算法与数据结构-2-清华-EDX](https://courses.edx.org/courses/course-v1:PekingX+04833050X+1T2016/course/)
- [算法设计与分析-1-北大-Cousera](https://www.coursera.org/learn/algorithms/home/welcome)
- [算法设计与分析-2-北大-EDX](https://courses.edx.org/courses/course-v1:PekingX+04833050X+1T2016/course/)

