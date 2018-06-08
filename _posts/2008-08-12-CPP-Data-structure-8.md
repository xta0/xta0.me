---
layout: post
list_title: Data Structre Part 8 | Introsort | 内排序
title: 内排序算法
sub_title: Introsort Algorithm
mathjax: true
---

本章主要讨论一些常见的内排序算法，所谓内排序是指整个排序过程是在内存中完成的，衡量内排序的标准为时间与空间复杂度

## 插入排序

插入排序是最一种符合人类认知习惯的排序方法

```
void improvedInsertSort

```


### 时间复杂度

- 最佳情况：n-1次比较，2(n-1)次移动，$\Theta(n)$
- 最坏情况：$\Theta(n^2)$
    - 比较次数：$\sum{i=1}{n-1}i=n(n-1)/2$ = $\Theta(n^2)$

## 选择排序
