---
list_title: 笔记 | 深度学习 | The Transformer Architecture
title: The Transformer Architecture
layout: post
mathjax: true
categories: ["AI", "Machine Learning", "Deep Learning"]
---

## Transformer Network Intuition

We have discussed the RNN and found that it had some problems with <mark>vanishing gradients</mark>, which made it <mark>hard to capture long range dependencies and sequences</mark>. We then looked at the GRU and then the LSTM model as a way to resolve many of those problems where you make use of gates to control the flow of information.

As we move from our RNNs to GRU to LSTM, the models became more complex. And all of these models are still <mark>sequential models</mark> in that they ingested the input sentence one token at the time, as if each unit is like a bottleneck to the flow of information, because <mark>to compute the output of this final unit, we first have to compute the outputs of all the units that come before</mark>.

<mark>Transformer architecture allows you to run a lot more of these computations for an entire sequence in parallel. So you can ingest an entire sentence all at the same time</mark>, rather than just processing it one word at a time from left to right. The major innovation of the transformer architecture is <mark>combining the use of attention based representations and a CNN style of processing</mark>, which can take input a lot of pixels and can compute representations for them in parallel.

To understand the attention network, there will be two key ideas:

- **Self-Attention**: The goal of self attention is, if you have, say, a sentence of five words will end up computing five representations for these five words: $A^{\langle 1 \rangle}$, $A^{\langle 2 \rangle}$ ... $A^{\langle 5 \rangle}$. And this will be an attention based way of computing representations for <mark>all the words in your sentence in parallel</mark>
- **Multi-Head Attention**: a for loop over the self-attention process, so you end up with multiple versions of these representations

## Self-Attention

We need to calculate the attention-based representations for each of the words in your input sentence. Let's say we have the following french sentence:

```
Jane visite l'Afrique en septembre
```

Our goal will be computing an attention-based representation for each word $A^{\langle i \rangle}$. For example, one way to represent `l'Afrique` would be to just look up the word embedding for `l'Afrique`. But depending on the context, are we thinking of `l'Afrique` or Africa as a site of historical interests or as a holiday destination, or as the world's second-largest continent. Depending on how you're thinking of `l'Afrique`, you may choose to represent it differently, and that's what this representation $A^{\langle 3 \rangle}$ will do.

<mark>Self-Attention will look at the surrounding words to try to figure out what does `l'Afrique` really mean in this sentence, and find the most appropriate representation for this.

<img class="md-img-center" src="{{site.baseurl}}/assets/images/2018/10/trans-1.png">

In the formula, we have $q^{\langle 3 \rangle}$, $k^{\langle 3 \rangle}$ and $v^{\langle 3 \rangle}$, representing, query, key and value. These vectors are the key inputs to computing the attention value for each word.

$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{Q K^\top}{\sqrt{d_k}}\right) V = \sum_i \frac{\exp(q \cdot k^{\langle i \rangle})}{\sum_j \exp(q \cdot k^{\langle j \rangle})} v^{\langle i \rangle}
$$

```
A(q, K, V) = attention-based vector representation of a word
```
