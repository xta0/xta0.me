---
list_title: 笔记 | 深度学习 | Word Embeddings
title: Word Embeddings
layout: post
mathjax: true
categories: ["AI", "Machine Learning","Deep Learning"]
---

之前我们将输入文本用一个1 hot vector来表示，它是建立在一个dictionary的基础上，比如单词`man`的表示方式为

```
[0, 0, 0, ..., 1, ..., 0, 0]
```
`1`表示其在dictionary中的index，我们用$O_{index}$表示。上面例子中，`man`为${O_{5791}}$。不难发现，对任意两个不同的word，他们所对应向量之间的inner product为`0`。这说明word之间完全正交，即使他们有相关性，系统也无法generalize，例如

```
I want a glass of orange _juice___
I want a glass of apple ____
```
即使系统可以推测出`orange juice`，但是当下次遇到 `apple`时，上次的结果无法generalize，此时还需要计算得到 `apple juice`，效率非常低。因此我们可以换一种形式来表示一个word