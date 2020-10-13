---
list_title: 笔记 | 深度学习 | Hperparameters Tuning
title: Hperparameters
layout: post
mathjax: true
categories: ["AI", "Machine Learning","Deep Learning"]
---

## Bias and Variance

如果我们在train data上面的error很低，但是在Dev set上的error很高，说明我们的模型出现over fitting，这种情况下我们说模型的**Variance**很高。如果二者错误率接近，且都很高，这是我们称为**high bias**，此时我们的模型的问题是under fitting。解决high bias可以引入更多的hidden layer来增加training的时间，或者使用一些的优化方法，后面会提到。如果要解决over fitting的问题，我们则需要更多的数据或者使用Regularization。

## Regularization

我们还是用Logistic Regression来举例。在LR中，Cost Function定义为

$$
J(w,b) = \frac{1}{m}\sum_{i=1}^{m}L(\hat{y}^{(i)}, y^{(i)})
$$

为了解决Overfitting，我们可以在上面式子的末尾增加一个Regularization项

$$
J(w,b) = \frac{1}{m}\sum_{i=1}^{m}L(\hat{y}^{(i)}, y^{(i)}) + \frac{\lambda}{2m}||{w}||{^2}
$$

上述公式末尾的二范数也称作L2 Regularization，其中$\omega$的二范数定义为

$$
||{w}||{^2} = \sum_{j=1}^{n_x}\omega^{2}_{j} = \omega^T\omega
$$

除了使用L2范数外，也有些model使用L1范数，即$\frac{\lambda}{2m}\|\|\omega\|\|_1$。

如果使用L1范数，得到的$\omega$矩阵会较为稀疏（0很多），不常用。

对于一般的Neural Network，Cost Function定义为

$$
J(\omega^{[1]}, b^{[1]},...,\omega^{[l]}, b^{[l]}) = \frac{1}{m}\sum_{i=1}^{m}L(\hat{y^{(i)}}, y^{(i)}) + \frac{\lambda}{2m}||{\omega}||{^2}
$$

其中对于某$l$层的L2范数计算方法为

$$
||\omega^{[l]}||^{(2)} = \sum_{i=1}^{n^{l}}\sum_{j=1}^{n^{(l-1)}}(\omega_{i,j}^{[l]})^2
$$

其中$i$表示row，$n^{l}$表示当前层有多少neuron(输出)，$j$表示column，$n^{(l-1)}$为前一层的输入有多少个neuron。简单理解，上面的L2范数就是对权重矩阵中的每个元素平方后求和。

新增的Regularization项同样会对反响求导产生影响，我们同样需要增加该Regularization项对$\omega$的导数

$$
d\omega = (from \ backprop) + \frac{\lambda}{m}\omega 
$$

Gradient Descent的公式不变

$$
\omega^{[l]} := \omega^{[l]} - \alpha d\omega^{[l]} 
$$

将上面$d\omega$带入，可以得到

$$
\omega^{[l]} := \omega^{[l]} - \alpha[(from \ backprop) + \frac{\lambda}{m}\omega] = (1-\frac{\alpha\lambda}{m})\omega^{[l]} - \alpha(from \ backprop)
$$

可以看到，在引入正则项后，$\omega^{[l]}$实际上是减小了，因此，L2正则也称作**weight decay**。

引入正则项为什么能减少overfitting呢？我们可以从两方面来考虑。首先通过上面的式子可以看出，如果$\lambda$很大，则$\omega$会变小，极端情况下，会有一部分weights变成0，那么我们hidden units会减少，模型将变得简单。另一个思考的方式是看activation results，我们知道

$$
z^{[l]} = \omega^{[l]}\alpha^{[l-1]} + b^{[l]}
$$

当$\omega$变小后，$z$会变小，那么输出的结果将趋于于线性

### Dropout

除了通过引入正则项来减少overfitting外，Dropout也是一种常用的手段。Dropout的思路很简单，每个hidden units将会以一定概率被去掉，去掉后的模型将变得更简单，从而减少overfitting。如下图所示 

<img src="{{site.baseurl}}/assets/images/2018/02/dp-ht-1.png">

Dropout比较流行的实现是inverted dropout，其思路为

1. 产生一个bool矩阵。以$l=3$为例，`d3 = np.random.rand(a3.shape[0], a3.shape[1]) < keep_prob`。其中`keep_prob`表示保留某个hidden unit的概率。则`d3`是一个0和1的bool矩阵
2. 计算`a3`。 `a3 = np.multiply(a3, d3)`
3. `a3 /= keep_prob`

神经网络中每层的hidden units数量可能不同，keep_prob的值也可以根据其数量进行调整。Dropout表面上看起来很简单，但实际上它也属于Regularization的一种，具体证明就不展开了。值得注意的是，Dropout会影响back prop，由于hidden units会被随机cut off，Gradient Descent的收敛曲线也将会变得不规则。因此常用的手段一般是先另keep_prop=1，确保曲线收敛，然后再逐层调整keep_prop的值，重新训练。
