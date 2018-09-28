---
layout: post
list_title: Data Structre Part 11-2 | 索引 | B/B+ 树 | Indexing-B/B+ Tree
title: B/B+ 树 
mathjax: true
categories: [DataStructure]
---

## 动态索引

动态索引主要用于处理索引结构本身可能发生变化的场景，比如数据库中的primary key可能会频繁插入，删除，造成索引结构的频繁更新。建立动态索引的目的视为了保持较好的查询性能，提高检索效率。实现动态索引有很多种方式，如果使用前面提到的多分树，则节点的插入删除不是很好调整。因此我们需要一个更高效的数据结构

### B树

<img class="md-img-center" src="{{site.baseurl}}/assets/images/2008/09/B-Tree-1.png">

B树是一种平衡的多分树(Balanced Tree)，如上图是一种3阶B树，也叫做`2-3`树，即每个节点有2个节点或者3个节点

1. 













## Resources 

- [CS106B-Stanford-YouTube](https://www.youtube.com/watch?v=NcZ2cu7gc-A&list=PLnfg8b9vdpLn9exZweTJx44CII1bYczuk)
- [Algorithms-Stanford-Cousera](https://www.coursera.org/learn/algorithms-divide-conquer/home/welcome)
- [算法与数据结构-1-北大-Cousera](https://www.coursera.org/learn/shuju-jiegou-suanfa/home/welcome)
- [算法与数据结构-2-北大-Cousera](https://www.coursera.org/learn/gaoji-shuju-jiegou/home/welcome)
- [算法与数据结构-1-清华-EDX](https://courses.edx.org/courses/course-v1:TsinghuaX+30240184.1x+3T2017/course/)
- [算法与数据结构-2-清华-EDX](https://courses.edx.org/courses/course-v1:PekingX+04833050X+1T2016/course/)
- [算法设计与分析-1-北大-Cousera](https://www.coursera.org/learn/algorithms/home/welcome)
- [算法设计与分析-2-北大-EDX](https://courses.edx.org/courses/course-v1:PekingX+04833050X+1T2016/course/)

