---
layout: post
list_title: 数据结构基础 | Data Structre | 索引-B/B+ 树 | B/B+ Tree
title: B/B+ 树 
mathjax: true
categories: [DataStructure]
---

## 动态索引

动态索引主要用于处理索引结构本身可能发生变化的场景，比如数据库中的primary key可能会频繁插入，删除，造成索引结构的频繁更新。建立动态索引的目的视为了保持较好的查询性能，提高检索效率。实现动态索引有很多种方式，如果使用前面提到的多分树，则节点的插入删除不是很好调整。因此我们需要一个更高效的数据结构

### B树

<img class="md-img-center" src="{{site.baseurl}}/assets/images/2008/09/B-Tree-1.png">

B树是R.Bayer和E.MacCreight在1970年提出的一种平衡的多路查找树，如上图是一种3阶B树，也叫做`2-3`树，即每个节点有2个子节点或者3个子节点。观察发现，它有如下性质：

1. 所有底层的节点，深度是一致的，是一种理想平衡的搜索树

B树所谓的多路搜索树和前面提到的二路搜索树实际上是等价的，将BST节点进行合并即可得到B树。例如下图中，将BST的根节点和两个子节点进行“两代”合并，可以得到4路B树。

<img class="md-img-center" src="{{site.baseurl}}/assets/images/2010/09/b-tree-2.png">















## Resources 

- [CS106B-Stanford-YouTube](https://www.youtube.com/watch?v=NcZ2cu7gc-A&list=PLnfg8b9vdpLn9exZweTJx44CII1bYczuk)
- [Algorithms-Stanford-Cousera](https://www.coursera.org/learn/algorithms-divide-conquer/home/welcome)
- [算法与数据结构-1-北大-Cousera](https://www.coursera.org/learn/shuju-jiegou-suanfa/home/welcome)
- [算法与数据结构-2-北大-Cousera](https://www.coursera.org/learn/gaoji-shuju-jiegou/home/welcome)
- [算法与数据结构-1-清华-EDX](https://courses.edx.org/courses/course-v1:TsinghuaX+30240184.1x+3T2017/course/)
- [算法与数据结构-2-清华-EDX](https://courses.edx.org/courses/course-v1:PekingX+04833050X+1T2016/course/)
- [算法设计与分析-1-北大-Cousera](https://www.coursera.org/learn/algorithms/home/welcome)
- [算法设计与分析-2-北大-EDX](https://courses.edx.org/courses/course-v1:PekingX+04833050X+1T2016/course/)

