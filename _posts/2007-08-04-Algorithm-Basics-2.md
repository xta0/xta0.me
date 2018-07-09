---
layout: post
list_title: Algorithms-2 | 算法分析基础 | Algorithm Analysis Basics
title: 算法分析基础
mathjax: true
---


### 递推方程

递推思想是人们常用的一种思维方式，它是一种正向的思考的过程，目的是为了找寻到某种规律，从而进行归纳和演绎，有点类似我们学过的数学归纳法。如果说前面提到的递归是自顶向下的逆向思路，那么递推就是自底向上的正向思路。

递推最重要的是建立递推方程，所谓递推方程，指的是对于序列$a_0,a_1,...,a_n,...$简记为$\\{a_n \\}$，一个把$a_n$与某些个$a_i$联系起来的等式叫做关于$\\{a_n \\}$的<mark>递推方程</mark>。

一个比较有名的递推方程的例子是Fibonacci数列，这个数列我们曾经在上一篇介绍动态规划的时候提到过，它是一个很好的具有递推关系例子，由前面文章可知它的递推方程为：$f_n = f_{n-1} + f_{n-2}$，其中$f_0=1, f_1 = 1$。但是当时我们并没有对这个这个递推方程进行求解，我们只是通过函数曲线观察到当$n$增大时，它是一个呈指数增长的函数。而这个方程的解是什么呢？不难求出，将递推式进行序列求和，得到

$$
f_n=\frac{1}{\sqrt 5}(\frac{1+\sqrt 5}{2})^{n+1} - \frac{1}{\sqrt 5}(\frac{1- \sqrt 5}{2})^{n+1}
$$

### 序列求和

上面的关于Fibonacci数列的递推方程是怎么求出来的呢？ 它实际上是一个序列求和的结果。序列求和，是用来计算迭代问题复杂度的有效工具，由于迭代问题的不同，它们可产生的数列类型也不同，总的来说有下面几种：

1. 算术级数，等差数列求和
    -  $T(n) = 1+2+...+n = \frac{n(n+1)}{2} = O(n^2)$
    -  $\sum_{k=1}^n a_k=\frac{n(a_1+a_n)}{2} = O(n^2)$ 
2. 几何级数，等比数列求和
    - $T(n) = 1+2+4+...+2^n = 2^{n+1}-1 = O(2^{n+1}) = O(2^n)$
    - $\sum_{k=0}^n aq^k = \frac{a(1-q^{n+1})}{1-q}$, $\sum_{k=0}^n aq^k = \frac{a}{1-q}(q<1) = O(a^n)$
3. 调和级数
    - $h(n) = 1+1/2+1/3 +...+ 1/n = \Theta(\log{n})$
    - $\sum_{k=1}^n\frac{1}{k} ={\ln n}+{O(1)} $
4. 对数级数
    - $\log1 + \log2 + ... + \log{n} = \log(n!) = \Theta(nlogn)$

### 递归树

1. 递归树是迭代计算的模型
2. 递归树的生成过程与迭代过程一致

### 主定理的证明和引用

形如下面的递推方程

$$
T(n) = a T(n/b) + f(n)
$$

其中$a$为归约后的子问题个数，$n/b$为归约后子问题规模，$f(n)$为归约过程及组合子问题的解的工作量。该类递推方程在实际引用中非常广泛，比如二分检索:$T(n) = T(n/2)+1$，二分归并排序$T(n) = 2T(n/2)n-1$，大部分算法形式包含递归。

求解上述递推方程，可分如下几种类型

1. 若$f(n) = O(n^{\log_b{a-\epsilon}}),\epsilon > 0, \thinspace 那么 \thinspace T(n) = \Theta(n^{\log_b{a}})$

2. 若$f(n) = \Theta(n^{\log_b{a}})，那么 \thinspace T(n) = \Theta(n^{\log_b{a}}\log{n})$

3. 若$f(n) = \Omega(n^{\log_b{a+\epsilon}}),\epsilon > 0, \thinspace，且对于某个常数$

### Resources

- [CS106B-Stanford-YouTube](https://www.youtube.com/watch?v=NcZ2cu7gc-A&list=PLnfg8b9vdpLn9exZweTJx44CII1bYczuk)
- [Algorithms-Stanford-Cousera](https://www.coursera.org/learn/algorithms-divide-conquer/home/welcome)
- [算法与数据结构-1-北大-Cousera](https://www.coursera.org/learn/shuju-jiegou-suanfa/home/welcome)
- [算法与数据结构-2-北大-Cousera](https://www.coursera.org/learn/gaoji-shuju-jiegou/home/welcome)
- [算法与数据结构-1-清华-EDX](https://courses.edx.org/courses/course-v1:TsinghuaX+30240184.1x+3T2017/course/)
- [算法与数据结构-2-清华-EDX](https://courses.edx.org/courses/course-v1:PekingX+04833050X+1T2016/course/)
- [算法设计与分析-1-北大-Cousera](https://www.coursera.org/learn/algorithms/home/welcome)
- [算法设计与分析-2-北大-EDX](https://courses.edx.org/courses/course-v1:PekingX+04833050X+1T2016/course/)