---
layout: post
list_title: Data Structre Part 9 | External Sort Algorithm | 外排序
title: 外排序算法
sub_title: External Sort Algorithm
mathjax: true
categories: [DataStructure]
---

外排序（External sorting）是指能够处理极大量数据的排序算法。通常来说，外排序处理的数据不能一次装入内存，因此，外排序通常采用的是一种“排序-归并”的策略。在排序阶段，先读入能放在内存中的数据量，将其排序输出到一个临时文件，依此进行，将待排序数据组织为多个有序的临时文件。尔后在归并阶段将这些临时文件组合为一个大的有序文件，也即排序结果。

## 文件的排序

对大文件的排序，通常由两个相对独立的阶段组成：

1. 文件形成尽可能长的初始顺串
2. 处理顺串，最后形成对整个数据文件的排列文件

### 置换选择排序



