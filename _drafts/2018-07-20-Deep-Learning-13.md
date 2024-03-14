---
list_title: 笔记 | 深度学习 | Transformer
title: Transformer
layout: post
mathjax: true
categories: ["AI", "Machine Learning","Deep Learning"]
---

<img class="md-img-center" src="{{site.baseurl}}/assets/images/2018/04/dl-rnn-1-nn-1.png">
- $x^{\langle i \rangle}$, 表示输入$x$中的第$i$个token

RNN在train的时候随着级数的加深，会有梯度消失的问题。GRU和LSTM虽然可以解决这个问题，但是会让model变的十分复杂。Transformer的思路是Attention + CNN