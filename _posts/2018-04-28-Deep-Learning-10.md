---
list_title: 深度学习 | Sequence Model
title: Sequence Model
layout: post
mathjax: true
categories: ["AI", "Machine Learning","Deep Learning"]
---

### Sequence Data Notations

- $x^{\<i\>}$, 表示输入$x$中的第$i$个元素
- $y^{\<i\>}$, 表示输出$y$中的第$i$个元素
- $x^{(i)\<t\>}$，表示第$i$个输入样本中的第$t$个元素
- $y^{(i)\<t\>}$，表示第$i$个输出样本中的第$t$个元素
- $T_x^{(i)}$，表示第$i$个输入样本的长度
- $T_y^{(i)}$，表示第$i$个输出样本的长度

以文本输入为例，假设我们有一个10000个字的字典和一个串文本"Harry Potter and Hermione Granger invented a new spell."，则$x^{<1>}$表示"Harry", $x^{<2>}$表示"Potter"，以此类推。假设在我们的字典中，"and"这个单词排在第5位，则$x^{<1>}$的值为

$$
x^{<1>} = [0,0,0,0,1,0, ... ,0]
$$

其余的$x^{\<i\>}$同理

### Recurrent Neural Network

RNN的核心概念是每层的输入除了对应的$x^{\<i\>}$之外，还来自前一层的输出，如下图所示

<img class="md-img-center" src="{{site.baseurl}}/assets/images/2018/04/dl-rnn-1-nn.png">

其中$a^{<0>}> = 0$，$a^{<1>}$, $y^{<1>}$的计算方式如下

$$
a^{<1>} = g(W_{aa}a^{<0>} + W_{ax}x^{<1>} + b_a) \\
\hat y^{<1>} = g(W_{ya}a^{<1>} + b_y) \\
$$

对于$a^{\<t\>}$, 其中常用的activation函数为$tanh$或$ReLU$，对于$\hat y^{\<i\>}$，可以用$sigmoid$函数。Generalize一下

$$
a^{\<t\>} = g(W_{aa}a^{\<{t-1}\>} + W_{ax}x^{\<t\>} + b_a) \\
\hat y^{\<t\>} = g(W_{ya}a^{\<t\>} + b_y) \\
$$

简单起见，我们可以将$W_{aa}$和$W_{ax}$合并，假设，$W_{aa}$为`[100,100]`, $W_{ax}$为`[100,10000]`(通常来说$W_{ax}$较宽)，则可以将$W_{ax}$放到$W_{aa}$的右边，即$[W_{aa}\|W_{ax}]$，则合成后的矩阵$W_{a}$为`[100，10100]`。$\Omega$矩阵合并后，我们也需要将$a^{\<t-1\>}$和$x^{\<t\>}$合并，合并方法类似，从水平改为竖直 $[\frac{a^{\<t-1\>}{x^{\<t\>}}}]$得到`[10100,100]`的矩阵