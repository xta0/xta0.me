---
layout: post
list_title: 数据结构基础 | Data Structure | 前缀树 | Trie
mathjax: true
title: Trie
categories: [DataStructure]
---

### 字符树

对于前面提到的BST，当输入是随机的情况下，可能达到理想的查询速度`O(log(N))`，但是如果输入是有序的，则BST会退化为单链表，查询速度会降为`O(N)`。因此我们需要思考，对于BST的构建，能否不和数据的输入顺序相关，而和数据的空间分布相关，这样只要数据的空间分布是随机的，那么构建出的BST查询性能就会得到保证。

Trie树又称前缀树或字典树，它最早应用于信息检索领域，所谓"Trie"来自英文单词"Retrieval"的中间的4个字符"trie"。Trie的经典应用场景是搜索提示和分词，比如当我们在搜索栏中输入关键词时，地址栏会自动提示并补全关键字，与BST不同，Trie中的节点保存的并不是key值，而是单个的英文字符，因此检索过程变成检索字符的每前缀字符不是比较字符串内容。

Trie树的特点是可以是对输入对象进行空间分解，一个节点的所有子孙都有相同的前缀，也就是这个节点对应的字符串，而根节点对应空字符串。一般情况下，不是所有的节点都有对应的值，只有叶子节点和部分内部节点所对应的键才有相关的值。如下图所示，我们可以将一组单词通过Trie树的形式进行构建，其中左图为等长字符串，每个字符均不是其它字符的前缀；右边为不等长字符串，每个字符可能是其它字符的前缀，这种情况我们需要在末尾增加一个`*`表示

<img src="{{site.baseurl}}/assets/images/2018/08/trie.png" class="md-img-center">

例如当我们想要查找字符串"bad"，我们只需要从Trie树的根节点开始依次匹配`bad`中的字符，直到遇到`*`表示匹配成功。

### Trie的实现


## Resources 

- [CS106B-Stanford-YouTube](https://www.youtube.com/watch?v=NcZ2cu7gc-A&list=PLnfg8b9vdpLn9exZweTJx44CII1bYczuk)
- [Algorithms-Stanford-Cousera](https://www.coursera.org/learn/algorithms-divide-conquer/home/welcome)
- [算法与数据结构-1-北大-Cousera](https://www.coursera.org/learn/shuju-jiegou-suanfa/home/welcome)
- [算法与数据结构-2-北大-Cousera](https://www.coursera.org/learn/gaoji-shuju-jiegou/home/welcome)
- [算法与数据结构-1-清华-EDX](https://courses.edx.org/courses/course-v1:TsinghuaX+30240184.1x+3T2017/course/)
- [算法与数据结构-2-清华-EDX](https://courses.edx.org/courses/course-v1:PekingX+04833050X+1T2016/course/)
- [算法设计与分析-1-北大-Cousera](https://www.coursera.org/learn/algorithms/home/welcome)
- [算法设计与分析-2-北大-EDX](https://courses.edx.org/courses/course-v1:PekingX+04833050X+1T2016/course/)

