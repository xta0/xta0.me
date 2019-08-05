---
list_title: 深度学习 | Deep-Layer Neural Networks
title: Deep-Layer Neural Networks
layout: post
mathjax: true
categories: ["AI", "Machine Learning","Deep Learning"]
---

<img src="{{site.baseurl}}/assets/images/2018/01/dp-w4-1.png" class="md-img-center" width="60%">

### Notations

- $n^{[l]}$: #units in layer $l$
- $a^{[l]}$: #activations units in layer $l$
    - $a^{[l]}=g^{[l]}(z^{[l]})$
    - $a^{[0]} = X$ 
- $W^{[l]}$: weights for $z^{[l]}$
- $b^{[l]}$

### Forward Propagation for Layer $l$

- Input $a^{[l-1]}$
- Output $a^{[l]}$, cache (z^{[l]})

$$
\begin{align*}
& Z^{[l]} = W^{[l]}A^{[l-1]} + $b^{[l]}
& A^{[l]} = g^{[l]}(Z^{[l]})
\end{align*}
$$

其中，$W^{[l]}$矩阵的维度为$(n^{[l]}, n^{[l-1]})$, $b^{[l]}$的维度为$(n^{[l]},1)$，$Z^{[l]}$和$A^{[l]}$均为$(n^{[l]},m)$ （m为训练样本数量）

### Backward Propagation for layer $l$

- Input $da^{[l]}$
- Output $da^{[l-1]}$, $dW^{[l]}$, $db^{[l]}$

$$
\begin{align*}
& dz^{[l]} = da^{[l]} *  g^{[l]'}(z^{[l]}) \quad (element-wise \ product)
& dw^{[l]} = dz^{[l]}a^{[l-1]}
& db^{[l]} = dz^{[1]}
& da^{[l-1]} = w^{[l]^T}dz^{[l]}
\end{align*}
$$

