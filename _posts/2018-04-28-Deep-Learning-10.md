---
list_title: 深度学习 | Sequence Model
title: Sequence Model
layout: post
mathjax: true
categories: ["AI", "Machine Learning","Deep Learning"]
---

### Notations

- $x^{\<i\>}$, 表示输入$x$中的第$i$个元素
- $y^{\<i\>}$, 表示输出$y$中的第$i$个元素
- $x^{(i)<t>}$，表示第$i$个输入样本中的第$t$个元素
- $y^{(i)<t>}$，表示第$i$个输出样本中的第$t$个元素
- $T_x^{(i)}$，表示第$i$个输入样本的长度
- $T_y^{(i)}$，表示第$i$个输出样本的长度

以文本输入为例，假设我们有一个10000个字的字典和一个串文本"Harry Potter and Hermione Granger invented a new spell."，则$x^{<1>}$表示"Harry", $x^{<2>}$表示"Potter"，以此类推。假设在我们的字典中，"and"这个单词排在第5位，则$x^{<1>}$的值为

$$
x^{\<1\>} = [0,0,0,0,1,0, & ... & ,0]
$$

其余的$x^{<\i>}$同理