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
`1`表示其在dictionary中的index，我们用$O_{index}$表示。上面例子中，`man`为${O_{5791}}$.