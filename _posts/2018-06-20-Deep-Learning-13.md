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

假设我们的字典有10,000个单词，每个单词的feature vector是`[300, 1]`，那么整个embedding matrix为`[10,000， 300]`，我们的目标就是train我们的network来找到这个embedding matrix

<img class="md-img-center" src="{{site.baseurl}}/assets/images/2018/06/dl-nlp-w2-2.png">

## Word2Vec

Word2Vec是一种相对比较高效的learn word emedding的一种算法。它的大概意思，选取一个context word，比如"orange" 和一个 target word比如 "juice"，我们通过构建neural network找到将context word映射成target word的embedding matrix。通常来说，这个target word是context word附近的一个word，可以是context word向前或者向后skip若干个random word之后得到的word。

如果从model的角度来来说，它的input是一个word，output是它周围的一个context word。

还是假定我们的字典大小为`10,000`，每个feature vector的dim是`300`，那么embedding的matrix大小为`[10,000, 300]`，我们的输入用word的1-hot vector表示，即是一个`[1000, 1]`的稀疏向量，则我们model定义如下

```
# for each input word, predict its context words surrounding it
class SkipGram(nn.Module):
    def __init__(self, n_vocab, n_embed):
        super().__init__()
        
        # complete this SkipGram model
        self.embedding = nn.Embedding(n_vocab, n_embed)
        self.fc = nn.Linear(n_embed, n_vocab)
        self.softmax = nn.LogSoftmax(dim=1)
    
    def forward(self, x):
        x = self.embedding(x)
        x = self.fc(x)
        x = self.softmax(x)
        return x
```

> [nn.Embedding](https://pytorch.org/docs/stable/generated/torch.nn.Embedding.html)

## Negtive Sampling

上面的`SkipGram`有一个性能问题是如果字典数量过大会导致softmax方法非常耗时。这里介绍另一种相对高效的network，叫做Negtive Sampling。它的大概意思是，给一组context word和target word，判断他们是否是符合语义，比如

```
x1: (orange, juice), y1:1
x2: (orange, king), y2:0
```
选取这个pair的方式和上面一样，sample一下context word，然后随机选取某一个context word周围一个word(window可以是左右10个word以内)作为target word。

因此，我们model变成了一个logistic regression model，它的input是一个pair，output是`0`或`1`用来表示这个pair是否正确。

``` python
context_embed = nn.Embedding(n_vocab, n_embed)
target_embed = nn.Embedding(n_vocab, n_embed)

P(y=1 | c,t) = sigmoid(target_embed.t() * target_embed)
```

当我们train这个model的时候，我们的traning dataset需要有negtive example，比如

```shell
context |  word | target?
--------------------------
orange  | juice | 1 
range   | king  | 0
orange  | book  | 0
orange  | the   | 0
orange  | of    | 0
```
但实际上我们train的时候，`y`不需要包含10,000个结果，而只需要`K`个，其中`K-1`个为negative example，`K`可以为4

## GloVe (global vectors for word representation)

TBD

## Sentiment Classification
