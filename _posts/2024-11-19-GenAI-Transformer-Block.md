---
list_title: GenAI | The Transformer Block
title: The Transformer Block
layout: post
mathjax: true
categories: ["GenAI", "Transformer", "LLM"]
---

## The Transformer Block

The Transformer block itself contains two major components - self attention layers and a feed forward network layer. 

<img class="md-img-center" src="{{site.baseurl}}/assets/images/2024/llm-tb-1.png">

### Combine Multi-Head Attention Scores

In the previous post, we have talked about the Multi-Head attention mechanism. We know that each attention head outputs its own attention representations for the input embedding words. <mark>The question is how do we merge or deal with those outputs from different attention heads?</mark> 

In this example, we have `3` attention heads and `2` attention values per head. We end up with `6` attention values. In order to get back down to the original number (`2`) of encoded values that started with, we simply connect all the attention scores to a fully connected layer that has 2 outputs, as shown below:

<img class="md-img-center" src="{{site.baseurl}}/assets/images/2018/10/trans-17.png">

### Feed Forward Neural Network

The feed forward network layer is responsible for predicting the next token after "The Shawshank"

<img class="md-img-center" src="{{site.baseurl}}/assets/images/2024/llm-tb-2.png">

### The Transformer Block in practice

<img class="md-img-center" src="{{site.baseurl}}/assets/images/2018/10/trans-21.png">

## Resources

- [How Transformer LLMs work](https://learn.deeplearning.ai/courses/how-transformer-llms-work)
- [3 blues 1 brown: Attention in Transformers](https://www.youtube.com/watch?v=eMlx5fFNoYc)