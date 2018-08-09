---
layout: post
list_title: Data Structre Part 10 | 索引-2 | B+ Tree
title: B树 和 B+ 树
mathjax: true
---

## 动态索引

动态索引主要用于数据库中的primary key频繁插入，删除，造成索引结构的频繁更新。如果使用前面提到的多分树，则不是很好调整。因此我们需要一个更高效的数据结构

### B树

B树是一种平衡的多分树(Balanced Tree)，它有以下性质

1. 

<img class="md-img-center" src="{{site.baseurl}}/assets/images/2008/09/B-Tree-1.png">

## 位索引