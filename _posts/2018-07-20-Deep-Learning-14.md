---
list_title: 笔记 | 深度学习 | Transformer
title: Transformer
layout: post
mathjax: true
categories: ["AI", "Machine Learning","Deep Learning"]
---

RNN在train的时候随着级数的加深，会有梯度消失的问题。GRU和LSTM虽然可以解决这个问题，但是会让model变的十分复杂。Transformer的思路是Attention + CNN。如果用传统的sequential model，每一层的输出以来前一层的输入，这种级联的方式需要将所有的token依次处理，效率极低。如果用CNN的方式，一次可以处理多个word。其中，Attention包括两个部分

- Self-Attention: 一次处理若干个word，假如一个sentence有5个word，我们会计算5个representation，$A^{\langle 1 \rangle}$, $A^{\langle 2 \rangle}$ ... $A^{\langle 5 \rangle}$
- Multi-Head Attention: a for loop over the self-attention process

## Self-Attenion Intuition

```
A(q, K, V) = attention-based vector representation of a word
```

假如我们有下面的法语句子

```
Jane visite l'Afrique en septembre
```

我们的任务是为每个单词计算出它们对应的 $A^{\langle i \rangle}$

Test!

$$
A^{\langle i \rangle}
$$
