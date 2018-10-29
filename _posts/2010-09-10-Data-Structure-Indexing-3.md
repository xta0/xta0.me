---
layout: post
list_title: 数据结构基础 | Data Structre | 索引-B/B+ 树 | B/B+ Tree
title: B/B+ 树 
mathjax: true
categories: [DataStructure]
---

## 动态索引

动态索引主要用于处理索引结构本身可能发生变化的场景，比如数据库中的primary key可能会频繁插入，删除，造成索引结构的频繁更新。建立动态索引的目的视为了保持较好的查询性能，提高检索效率。常用的动态索引结构有B/B+树，红黑树等。

### B树

B树是R.Bayer和E.MacCreight在1970年提出的一种平衡的多路查找树。所谓的多路搜索树和前面提到的二路搜索树实际上是等价的，只需要将BST的父节点和子节点进行合并即可得到B树的节点。例如下图中，将BST的根节点和两个子节点进行“两代”合并：

<img class="md-img-center" src="{{site.baseurl}}/assets/images/2010/09/b-tree-2.png" width="80%">

可见经过“父子”2代的合并，可以得到4路B树，其中每个节点有3个Key值，类似的:

1. 每`3`代合并，可得`8`路B树，每个节点有`7`个关键码
2. 每`d`代合并，可得到`m=2^d`路，每个节点有`m-1`个关键码

正如开篇提到的，B树适合对频繁操作的数据进行动态索引，其原因是什么呢？ 如果使用前面介绍的AVL树查询，由于其具有自平衡性，其性能也应该不会太差，那为什么不用AVL树做动态索引呢？我们可以看一个具体例子：

假如我们有个1G的文件记录，我们需要查找其中的某一个，如果不允许将文件读入内存，则使用AVL树需要`log(2,2^30) = 30`次I/O查询操作。如果使用B树，则一次I/O可读取一组关键码，对于多数的数据库系统，通常可以支持`m=256`，那么回到上面问题中，使用B树一次I/O可以读入256个关键码，因此只需要查询`log(256,2^30) <= 4`次I/O即可。

### B树的定义

所谓`m`阶B树，即`m`路平衡搜索树(`m>=2`)，它有如下性质

1. 所有底层的节点，深度是一致的，是一种理想平衡的搜索树
2. 如果每个节点有`n`个关键码，则每个节点最多有`n+1`个分支

也叫做`2-3`树，即每个节点有2个子节点或者3个子节点。观察发现，它有如下性质：




<img class="md-img-center" src="{{site.baseurl}}/assets/images/2008/09/B-Tree-1.png">

















## Resources 

- [CS106B-Stanford-YouTube](https://www.youtube.com/watch?v=NcZ2cu7gc-A&list=PLnfg8b9vdpLn9exZweTJx44CII1bYczuk)
- [Algorithms-Stanford-Cousera](https://www.coursera.org/learn/algorithms-divide-conquer/home/welcome)
- [算法与数据结构-1-北大-Cousera](https://www.coursera.org/learn/shuju-jiegou-suanfa/home/welcome)
- [算法与数据结构-2-北大-Cousera](https://www.coursera.org/learn/gaoji-shuju-jiegou/home/welcome)
- [算法与数据结构-1-清华-EDX](https://courses.edx.org/courses/course-v1:TsinghuaX+30240184.1x+3T2017/course/)
- [算法与数据结构-2-清华-EDX](https://courses.edx.org/courses/course-v1:PekingX+04833050X+1T2016/course/)
- [算法设计与分析-1-北大-Cousera](https://www.coursera.org/learn/algorithms/home/welcome)
- [算法设计与分析-2-北大-EDX](https://courses.edx.org/courses/course-v1:PekingX+04833050X+1T2016/course/)

