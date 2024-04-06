---
list_title: 笔记 | 深度学习 | NLP and Word Embeddings
title: NLP and Word Embeddings
layout: post
mathjax: true
categories: ["AI", "Machine Learning","Deep Learning"]
---

## Word Embeddings

之前我们将输入文本用一个1 hot vector来表示，它是建立在一个dictionary的基础上，比如单词`man`的表示方式为

```
[0, 0, 0, ..., 1, ..., 0, 0]
```
`1`表示其在dictionary中的index，我们用$O_{index}$表示。上面例子中，`man`在字典中的index为5791，则对应的表示为${O_{5791}}$。不难发现，对任意两个不同的word，他们所对应向量之间的inner product为`0`。这说明word之间完全正交，即使他们有相关性，系统也无法generalize，例如

```
I want a glass of orange juice
I want a glass of apple ____
```
即使系统可以推测出`orange juice`，但是当下次遇到 `apple`时，由于`apple`和`orange`正交，则之前的结果无法generalize到apple上面，此时还需要计算得到 `apple juice`，效率非常低。因此我们可以换一种形式来表示一个word。

我们可以给字典里的每个word关联一些feature，比如

<img class="md-img-center" src="{{site.baseurl}}/assets/images/2018/06/dl-nlp-w2-1.png">

图中我们为每个word关联了300个feature，因此每个word可以用一个`[300, 1]`的vector表示，对于单词`Man`所对应的vector，我们用 $e_{5391}$ 来表示。这样两个相似的word，他们的feature vector也是相似的，比如`Apple`和`Orange`。

回到最开始RNN的那个例子，假设我们有一个句子，我们需要识别出那些word是人名

```python
x = ["Sally", "Johnson", "is", "an", "orange", "farmer"]
y = [1, 1, 0, 0, 0, 0]
```
之前每个word使用1 hot vector来表示，现在则可以用word embedding来表示。那么word embedding从哪里来呢？我们需要自己训练model来得到每个word的embedding，当然也可以下载已经训练好的。实际上，对于每个word来说，我们可以想象将其encode成一个vector，即embedding。

## Embedding Matrix

假设我们的字典有10,000个单词，每个单词的feature vector是`[300, 1]`，那么整个embedding matrix为`[300, 10,000]`，我们的目标就是train我们的network来找到这个embedding matrix

<img class="md-img-center" src="{{site.baseurl}}/assets/images/2018/06/dl-nlp-w2-2.png">

