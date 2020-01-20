---
list_title: 深度学习 | Sequence Model
title: Sequence Model
layout: post
mathjax: true
categories: ["AI", "Machine Learning","Deep Learning"]
---

### Sequence Data Notations

- $x^{[i]}$, 表示输入$x$中的第$i$个元素
- $y^{[i]}$, 表示输出$y$中的第$i$个元素
- $x^{(i)[t]}$，表示第$i$个输入样本中的第$t$个元素
- $y^{(i)[t]}$，表示第$i$个输出样本中的第$t$个元素
- $T_x^{(i)}$，表示第$i$个输入样本的长度
- $T_y^{(i)}$，表示第$i$个输出样本的长度

以文本输入为例，假设我们有一个10000个单词的字典和一个串文本，现在的问题是让我们查字典找出下面文本中是人名的单词

```
"Harry Potter and Hermione Granger invented a new spell."
```

我们用$x^{[i]}$表示上述句子中的每个单词，则$x^{[1]}$表示"Harry", $x^{[2]}$表示"Potter"，以此类推。假设在我们的字典中，`and`这个单词排在第5位，则$x^{[1]}$的值为

$$
x^{[1]} = [0,0,0,0,1,0, ... ,0]
$$

其余的$x^{[i]}$同理。相应的，上述句子对应的$y$表示如下，其中$y^{[i]}$表示是名字的概率

$$
y = [1,1,0,1,1,0,0,0,0]
$$

### Recurrent Neural Network

RNN的核心概念是每层的输入除了对应的$x^{[i]}$之外，还来自前一层的输出，如下图所示

<img class="md-img-center" src="{{site.baseurl}}/assets/images/2018/04/dl-rnn-1-nn-1.png">

其中$a^{[0]} = 0$，$a^{[1]}$, $y^{[1]}$的计算方式如下

$$
a^{[1]} = g(W_{aa}a^{[0]} + W_{ax}x^{[1]} + b_a) \\
\hat y^{[1]} = g(W_{ya}a^{[1]} + b_y) 
$$

对于$a^{[t]}$, 其中常用的activation函数为$tanh$或$ReLU$，对于$\hat y^{[i]}$，可以用$sigmoid$函数。Generalize一下

$$
a^{[t]} = g(W_{aa}a^{[t-1]} + W_{ax}x^{[t]} + b_a) \\
\hat y^{[t]} = g(W_y a^{[t]} + b_y) 
$$

简单起见，我们可以将$W_{aa}$和$W_{ax}$合并，假设，$W_{aa}$为`[100,100]`, $W_{ax}$为`[100,10000]`(通常来说$W_{ax}$较宽)，则可以将$W_{ax}$放到$W_{aa}$的右边，即$[W_{aa}\|W_{ax}]$，则合成后的矩阵$W_{a}$为`[100，10100]`。$W_a$矩阵合并后，我们也需要将$a^{<{t-1}>}$和$x^{[t]}$合并，合并方法类似，从水平改为竖直 $[\frac{a^{[t-1]}}{x^{[t]}}]$得到`[10100,100]`的矩阵。

<mark>因此，我们需要学习的参数便集中在了$W_a$, $b_a$和$W_y$,$b_y$上。</mark>

注意，上图中，对句子中的每个单词$x^{[t]}$都能产生一个$\hat y^{[t]}$，假设一个句子有$m$个单词，那么这一个句子 - 样本$y^{(i)}$的大小为`[m,m]`

### Loss函数

上一节中我们已经看到，对每条训练样本来说，任何一个单词产生的输出$\hat y^{(i)[t]}$是一个一维向量，形式和分类问题类似，因此对于单个单词的loss函数可以用逻辑回归的loss函数

$$
L^{[t]}(\hat y ^{[t]}, y^{[t]}) = - y^{[t]}log{y^{[t]}} - (1-y^{[t]})log{(1-y^{[t]})}
$$

则对于整个样本（句子），loss函数为每个单词loss的和

$$
L(\hat y, y) = \sum_{t=1}^{T} L^{[t]}(\hat y ^{[t]}, y^{[t]})
$$

### 不同的RNN网络

除了上面提到的一种RNN网络外，根据实际应用的不同，RNN可以衍生出不同的结构，如下图所示

<img class="md-img-center" src="{{site.baseurl}}/assets/images/2018/04/dl-rnn-1-nn-2.png">

### Language Model

Language Model有很多种，其输入为一个句子，输出为预测结果，通常以概率形式表示。假如我们的字典有10000个单词，输入文本如下

```
Cats average 15 hours of sleep a day. <EOS>
```
我们可以参考前面提到的RNN网络来构建我们的Model，如下图所示

<img class="md-img-center" src="{{site.baseurl}}/assets/images/2018/04/dl-rnn-1-nn-3.png">

1. 另$x^{[1]}$和$a^{[0]}$均为0，输出$\hat y^{[1]}$是一个softmax结果，表示字典中每个单词出现的概率，是一个`[1,10000]`的向量，由于未经训练，每个单词出现的概率均为`1/10000`
2. 接下来我们用真实$y^{[1]}$（"Cats"在字典中出现的概率）和 $a^{[1]}$作为下一层的输入，得到$\hat y^{[2]}$，其含义为当给定前一个单词为"Cats"时，当前单词是字典中各个单词的概率即 $P(?? \|Cats)$，因此$\hat y^{[2]}$也是`[1,10000]`的。注意到，此时的$x^{[2]} = y^{[1]}$
3. 类似的，第三层的输入为真实结果$y^{[2]}$，即$P(average \|Cats)$，和$a^{[2]}$，输出为$\hat y^{[2]}$，表示$P(?? \|Cats average)$。同理，此时$x^{[3]} = y^{[2]}$
4. 重复上述步骤，直到走到EOS的位置

上述的RNN模型可以做到根据前面已有的单词来预测下一个单词是什么

### GRU

不难发现，上面的RNN模型是基于前面的单词来预测后面出现的单词出现的概率，但是对于一些长句子，单词前后的联系可能被分隔开，比如英语中的定语从句

```shell
The cat, which already ate ... , was full
The cats, which already ate ..., were full
```
上面例子例子中`cat`和`was`, `cats`和`were`中间隔了一个很长的定语修饰，这就会导致当RNN在预测`was`或者`were`时，由于前面的主语信息(`cat`或者`cats`)位置很靠前，使得预测概率受到影响（如果RNN能识别出此时主语是`cat`/`cats`则`was`/`were`的预测概略应该会提高）。具体在RNN中的表现是当做back prop时，由于网络太深，会出现梯度消失的问题，也就是说我们无法通过back prop来影响到`cat`后者`cats`的weight。

GRU(Gated Recurrent Uinit)被设计用来解决上述问题，其核心思想是为每个token引入一个GRU unit - $c^{[t]}$

计算中我们用$\hat c^{[t]}$来逼近$c^{[t]}$，计算方式如下

$$
\hat c^{[t]} tanh (W_c[c^{[t-1]}, x^{[t]}] + b_c)
$$

虽然我们定义了GRU unit，但是是否要更新它的值则需要通过一个Gate来控制，定义为$\Gamma_u ^{[t]}$其中$u$表示update

$$
\Gamma_u ^{[t]} \delta (W_u[c^{[t-1]}, x^{[t]}] + b_u)
$$


## Resources

- [The Unreasonable Effectiveness of Recurrent Neural Networks](http://karpathy.github.io/2015/05/21/rnn-effectiveness/)
- [Deep Learning Specialization Course on Coursera](https://www.coursera.org/specializations/deep-learning)
- [Deep Learning with PyTorch](https://livebook.manning.com/book/deep-learning-with-pytorch/welcome/v-10/)



