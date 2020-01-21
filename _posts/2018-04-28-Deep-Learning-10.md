---
list_title: 深度学习 | Sequence Model
title: Sequence Model
layout: post
mathjax: true
categories: ["AI", "Machine Learning","Deep Learning"]
---

### Sequence Data Notations

- $x^{\langle i \rangle}$, 表示输入$x$中的第$i$个token
- $y^{\langle i \rangle}$, 表示输出$y$中的第$i$个token
- $x^{(i)\langle t \rangle}$，表示第$i$条输入样本中的第$t$个token
- $y^{(i)\langle t \rangle}$，表示第$i$条输出样本中的第$t$个token
- $n_x$，表示某个输入token向量长度
- $n_y$，表示某个输出token长度
- $x_5^{(2)\langle 3 \rangle\langle 4 \rangle}$, 表示第二条输入样本中，第三个layer中第4个token向量中的第五个元素

以文本输入为例，假设我们有一个10000个单词的字典和一个串文本，现在的问题是让我们查字典找出下面文本中是人名的单词

```
"Harry Potter and Hermione Granger invented a new spell."
```

我们用$x^{\langle i \rangle}$表示上述句子中的每个单词，则$x^{\langle 1 \rangle}$表示"Harry", $x^{\langle 2 \rangle}$表示"Potter"，以此类推。假设在我们的字典中，`and`这个单词排在第5位，则$x^{\langle 1 \rangle}$的值为一个一维向量

$$
x^{\langle 1 \rangle} = [0,0,0,0,1,0, ... ,0]
$$

注意上面的式子通常用列向量表示，即$x^{\langle i \rangle}$为`[10000,1]`。

> 在实际应用中，$x^{\langle 1 \rangle}$往往是一个2D tensor，因为我们通常一次输入$m$条训练样本(mini-batch)。我们假设`m=20`，则此时我们有20列向量，我们可以横向将它们stack成一个二维矩阵。比如上面例子中，RNN在某个时刻的输入tensor的大小是`[10000,20]`的。

相应的，上述句子对应的$y$表示如下，其中$y^{\langle i \rangle}$表示是名字的概率

$$
y = [1,1,0,1,1,0,0,0,0]
$$

### Recurrent Neural Network

RNN的核心概念是将输入数据切分为为一系列时间片，每个时间片上的数据会通过某一系列运算产生一个输出，并且该时间片上的输入除了有$x^{\langle i \rangle}$之外，还有可能来自前一个时间片的输出，如下图所示

<img class="md-img-center" src="{{site.baseurl}}/assets/images/2018/04/dl-rnn-1-nn-1.png">

图中的$T$表示时间片，$a^{\langle {T_x} \rangle}$为$T$时刻的hidden state。我们令 $a^{\langle 0 \rangle} = 0$，$a^{\langle 1 \rangle}$, 则$y^{\langle 1 \rangle}$的计算方式如下

$$
a^{\langle 1 \rangle} = g(W_{aa}a^{\langle 0 \rangle} + W_{ax}x^{\langle 1 \rangle} + b_a) \\
\hat y^{\langle 1 \rangle} = g(W_{ya}a^{\langle 1 \rangle} + b_y) 
$$

对于$a^{\langle t \rangle}$, 其中常用的activation函数为$tanh$或$ReLU$，对于$\hat y^{\langle i \rangle}$，可以用$sigmoid$函数。Generalize一下

$$
a^{\langle t \rangle} = g(W_{aa}a^{\langle {t-1} \rangle} + W_{ax}x^{\langle t \rangle} + b_a) \\
\hat y^{\langle t \rangle} = g(W_y a^{\langle t \rangle} + b_y) 
$$

简单起见，我们可以将$W_{aa}$和$W_{ax}$合并，假设，$W_{aa}$为`[100,100]`, $W_{ax}$为`[100,10000]`(通常来说$W_{ax}$较宽)，则可以将$W_{ax}$放到$W_{aa}$的右边，即$[W_{aa}\|W_{ax}]$，则合成后的矩阵$W_{a}$为`[100，10100]`。$W_a$矩阵合并后，我们也需要将$a^{ \langle {t-1} \rangle}$和$x^{\langle t \rangle}$合并，合并方法类似，从水平改为竖直 $[\frac{a^{\langle {t-1} \rangle}}{x^{\langle t \rangle}}]$得到`[10100,100]`的矩阵。

$$
a^{\langle t \rangle} = g(W_a[a^{\langle {t-1} \rangle}, x^{\langle t \rangle}] + b_a) \\
\hat y^{\langle t \rangle} = g(W_y a^{\langle t \rangle} + b_y) 
$$

<mark>因此，我们需要学习的参数便集中在了$W_a$, $b_a$和$W_y$,$b_y$上。</mark>

### Loss函数

上一节中我们已经看到，对每条训练样本来说，任何一个单词产生的输出$\hat y^{(i)\langle t \rangle}$是一个一维向量，形式和分类问题类似，因此对于单个单词的loss函数可以用逻辑回归的loss函数

$$
L^{\langle t \rangle}(\hat y ^{\langle t \rangle}, y^{\langle t \rangle}) = - y^{\langle t \rangle}log{y^{\langle t \rangle}} - (1-y^{\langle t \rangle})log{(1-y^{\langle t \rangle})}
$$

则对于整个样本（句子），loss函数为每个单词loss的和

$$
L(\hat y, y) = \sum_{t=1}^{T} L^{\langle t \rangle}(\hat y ^{\langle t \rangle}, y^{\langle t \rangle})
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

其中每个cell的结构如下图所示

<img class="md-img-center" src="{{site.baseurl}}/assets/images/2018/04/dl-rnn-1-cell.png">

1. 另$x^{\langle 1 \rangle}$和$a^{\langle 0 \rangle}$均为0，输出$\hat y^{\langle 1 \rangle}$是一个softmax结果，表示字典中每个单词出现的概率，是一个`[10000,1]`的向量，由于未经训练，每个单词出现的概率均为`1/10000`
2. 接下来我们用真实$y^{\langle 1 \rangle}$（"Cats"在字典中出现的概率）和 $a^{\langle 1 \rangle}$作为下一层的输入，得到$\hat y^{\langle 2 \rangle}$，其含义为当给定前一个单词为"Cats"时，当前单词是字典中各个单词的概率即 $P(?? \|Cats)$，因此$\hat y^{\langle 2 \rangle}$也是`[10000,1]`的。注意到，此时的$x^{\langle 2 \rangle} = y^{\langle 1 \rangle}$
3. 类似的，第三层的输入为真实结果$y^{\langle 2 \rangle}$，即$P(average \|Cats)$，和$a^{\langle 2 \rangle}$，输出为$\hat y^{\langle 2 \rangle}$，表示$P(?? \|Cats average)$。同理，此时$x^{\langle 3 \rangle} = y^{\langle 2 \rangle}$
4. 重复上述步骤，直到走到EOS的位置

上述的RNN模型可以做到根据前面已有的单词来预测下一个单词是什么

### GRU

不难发现，上面的RNN模型是基于前面的单词来预测后面出现的单词出现的概率，但是对于一些长句子，单词前后的联系可能被分隔开，比如英语中的定语从句

```shell
The cat, which already ate ... , was full
The cats, which already ate ..., were full
```
上面例子例子中`cat`和`was`, `cats`和`were`中间隔了一个很长的定语修饰，这就会导致当RNN在预测`was`或者`were`时，由于前面的主语信息(`cat`或者`cats`)位置很靠前，使得预测概率受到影响（如果RNN能识别出此时主语是`cat`/`cats`则`was`/`were`的预测概略应该会提高）。具体在RNN中的表现是当做back prop时，由于网络太深，会出现梯度消失的问题，也就是说我们无法通过back prop来影响到`cat`后者`cats`的weight。

GRU(Gated Recurrent Uinit)被设计用来解决上述问题，其核心思想是为每个token引入一个GRU unit - $c^{\langle t \rangle}$，计算方式如下

$$
\hat c^{\langle t \rangle} = tanh (W_c[c^{\langle {t-1} \rangle}, x^{\langle t \rangle}] + b_c) \\
\Gamma_u ^{\langle t \rangle} = \delta (W_u[c^{\langle {t-1} \rangle}, x^{\langle t \rangle}] + b_u) \\
c^{\langle t \rangle} = \Gamma_u ^{\langle t \rangle} * \hat c^{\langle t \rangle} + (1-\Gamma_u ^{\langle t \rangle}) * c^{\langle {t-1} \rangle}
$$

其中，$\Gamma_u ^{\langle t \rangle}$用来控制是否更新$c^{\langle t \rangle}$的值，$\delta$通常为sigmoid函数，因此$\Gamma_u ^{\langle t \rangle}$的取值为0或1；`*`为element-wise的乘法运算

回到上面的例子，假设我们`cats`对应的$c^{\langle t \rangle}$值为`0`或`1`, `1`表示主语是单数，`0`表示主语是复数。则直到计算`was/were`之前，$c^{\langle t \rangle}$的值会一直被保留，作为计算时的参考，保留的方式则是通过控制$\Gamma_u ^{\langle t \rangle}$来完成

```shell
Tha cat,    which   already   ate ...,   was    full.
    c[t]=1                               c[t]=1
    g[t]=1  g[t]=0  g[t]=0    g[t]=0 ... g[t]=0  
```
可以看到当$\Gamma_u ^{\langle t \rangle} $为1时，$c^{\langle t \rangle} = c^{\langle {t-1} \rangle} = a^{\langle {t-1} \rangle}$，则前面的信息可以被一直保留下来。

注意到$c^{\langle t \rangle}, \hat c^{\langle t \rangle}, \Gamma_u ^{\langle t \rangle}$均为向量，其中$\Gamma_u ^{\langle t \rangle}$向量中的值为0或1，则上面最后一个式子的乘法计算为element-wise的，这样$\Gamma_u ^{\langle t \rangle}$就可以起到gate的作用。

### LSTM

Long Short Term Memory(LSTM)是另一种通过建立前后token链接来解决梯度消失问题的方法，相比GRU更为流行一些。和GRU不同的是

1. LSTM使用$a^{\langle {t-1} \rangle}$来计算 $\hat c^{\langle t \rangle}$和$\Gamma_u ^{\langle t \rangle}$
2. LSTM使用两个gate来控制$c^{\langle t \rangle}$，一个前面提到的$\Gamma_u ^{\langle t \rangle}$，另一个是forget gate - $\Gamma_f ^{\langle t \rangle}$
3. LSTM使用了一个output gate来控制$a^{\langle t \rangle}$

$$
\hat c^{\langle t \rangle} = tanh (W_c[a^{\langle {t-1} \rangle}, x^{\langle t \rangle}] + b_c) \\
\Gamma_u ^{\langle t \rangle} = \delta (W_u[a^{\langle {t-1} \rangle}, x^{\langle t \rangle}] + b_u) \\
\Gamma_f ^{\langle t \rangle} = \delta (W_f[a^{\langle {t-1} \rangle}, x^{\langle t \rangle}] + b_f) \\
c^{\langle t \rangle} = \Gamma_u ^{\langle t \rangle} * \hat c^{\langle t \rangle} + \Gamma_f ^{\langle t \rangle} * c^{\langle {t-1} \rangle} \\
\Gamma_o ^{\langle t \rangle} = \delta (W_o[a^{\langle {t-1} \rangle}, x^{\langle t \rangle}] + b_o) \\
a^{\langle t \rangle} = \Gamma_o * tanh(c^{\langle t \rangle})
$$

上述LSTM式子引入了三个gate函数，虽然步骤不较复杂，但是逻辑上还是比较清晰，也容易更好的整合到RNN网络中，下图是一个引入了LSTM的RNN的计算单元

<img class="md-img-center" src="{{site.baseurl}}/assets/images/2018/04/dl-rnn-1-lstm-1.png">

如果把各个LSTM单元串联起来，则RNN的模型变为

<img class="md-img-center" src="{{site.baseurl}}/assets/images/2018/04/dl-rnn-1-lstm-2.png">

上述红线表示了$c^{\langle t \rangle}$的记忆过程，通过gate的控制，可以使$c^{\langle 3 \rangle} = c^{\langle 1 \rangle}$, 从而达到缓存前面信息的作用，进而可以解决梯度消失的问题

## Resources

- [The Unreasonable Effectiveness of Recurrent Neural Networks](http://karpathy.github.io/2015/05/21/rnn-effectiveness/)
- [Deep Learning Specialization Course on Coursera](https://www.coursera.org/specializations/deep-learning)
- [Deep Learning with PyTorch](https://livebook.manning.com/book/deep-learning-with-pytorch/welcome/v-10/)



