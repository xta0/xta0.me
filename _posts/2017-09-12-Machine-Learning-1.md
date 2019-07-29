---
layout: post
list_title: 机器学习 | Machine Learning | Overview
title: Machine Learning Overview
meta: Coursera Stanford Machine Learning Cousre Note, Chapter1
categories: [Machine Learning,AI]
mathjax: true
---

> 文中所用到的图片部分截取自Andrew Ng在[Cousera上的课程](https://www.coursera.org/learn/machine-learning)

### Macine Learning definition

机器学习的定义：

* Arthur Samuel(1959). Machine Learning: Field of study that gives computers the ability to learn without being explicitly programmed.
* Tom Mitchell(1998). Well-posed Learning Problem: A computer program is said to learn from experience E with respect to some task T and some performance measure P, if its performance on T, as measured by P, improves with experience E.

### Machine Learning Algorithms:

* 监督学习：Supervised learning
* 非监督学习：Unsupervised learning
* 其它： - Reinforcement learning - recommender systems

### Supervised Learning

**监督学习**: "Define supervised learning as problems where the desired output is provided for examples in the training set." 监督学习我们对给定数据的预测结果有一定的预期，输入和输出之间有某种关系，监督学习包括"Regression"和"Classification"。其中"Regression"是指预测函数的预测结果是连续的，"Classification"指的是预测函数的结果是离散的

### Unspervised Learning

**非监学习**: "Define unsupervised learning as problems where we are not told what the desired output is." 非监督学习我们对预测的结果没有预期,我们可以从数据中得出某种模型，但是却无法知道数据中的变量会带来什么影响，我们可以通过将变量之间的关系进行聚类来推测出预测模型。非监督学习没有基于预测结果的反馈，两个例子：

* 聚类：取 1,000,000 种不同的基因，找到一种自动将其分类的方法，可以细胞间不同变量的相似程度进行分类，比如生命周期，位置，功能等
* 非聚类：经典的"Cocktail Party Algorithm"，从吵杂的环境中确定某个个体的声音

### Model Representation

为了后面课程使用方便，我们先来定义一些术语：

* 我们使用<math xmlns="http://www.w3.org/1998/Math/MathML"> <msup> <mi>x</mi> <mrow class="MJX-TeXAtom-ORD"> <mo stretchy="false">(</mo> <mi>i</mi> <mo stretchy="false">)</mo> </mrow> </msup> </math> 来表示输入的特征样本，使用 <math xmlns="http://www.w3.org/1998/Math/MathML"> <msup> <mi>y</mi> <mrow class="MJX-TeXAtom-ORD"> <mo stretchy="false">(</mo> <mi>i</mi> <mo stretchy="false">)</mo> </mrow> </msup> </math> 表示我们想要得到的预测结果
* 我们使用 <math xmlns="http://www.w3.org/1998/Math/MathML"> <mo stretchy="false">(</mo> <msup> <mi>x</mi> <mrow class="MJX-TeXAtom-ORD"> <mo stretchy="false">(</mo> <mi>i</mi> <mo stretchy="false">)</mo> </mrow> </msup> <mo>,</mo> <msup> <mi>y</mi> <mrow class="MJX-TeXAtom-ORD"> <mo stretchy="false">(</mo> <mi>i</mi> <mo stretchy="false">)</mo> </mrow> </msup> <mo stretchy="false">)</mo> </math> 来表示一组训练样本，通常我们的数据集中有多个训练样本，数据集用<math xmlns="http://www.w3.org/1998/Math/MathML"> <mrow class="MJX-TeXAtom-ORD"> <mo stretchy="false">(</mo> <msup> <mi>x</mi> <mrow class="MJX-TeXAtom-ORD"> <mo stretchy="false">(</mo> <mi>i</mi> <mo stretchy="false">)</mo> </mrow> </msup> <mo>,</mo> <msup> <mi>y</mi> <mrow class="MJX-TeXAtom-ORD"> <mo stretchy="false">(</mo> <mi>i</mi> <mo stretchy="false">)</mo> </mrow> </msup> <mo stretchy="false">)</mo> <mo>;</mo> <mi>i</mi> <mo>=</mo> <mn>1</mn> <mo>,</mo> <mo>.</mo> <mo>.</mo> <mo>.</mo> <mo>,</mo> <mi>m</mi> </mrow> </math> 表示，注意上角标`(i)`表示数据样本的 index
* 我们使用`X`表示输入样本空间，也可以理解为输入矩阵，`Y`表示输出样本空间或者输出矩阵，有 X = Y = ℝ.

在监督学习中，对输入的样本`X`我们使用预测函数（hypothesis）`h(x)` 来求解预测结果`y`，即`h : X → Y`，如下图所示

![Altext](/assets/images/2017/09/ml-1.png)

> Regression Analysis 是一种统计学上分析数据的方法，目的在于了解两个或多个变数间是否相关、相关方向与强度，并建立数学模型以便观察特定变数来预测研究者感兴趣的变数。更具体的来说，回归分析可以帮助人们了解在只有一个自变量变化时因变量的变化量。一般来说，通过回归分析我们可以由给出的自变量估计因变量的条件期望。

回归在数学上来说是建立因变数 <math><mi>Y</mi></math> 与自变数 <math><mi>X</mi></math>之间关系的模型，给定一个点集，能够用一条曲线去拟合之，如果这个曲线是一条直线，那就被称为线性回归，如果曲线是一条二次曲线，就被称为二次回归，回归还有很多的变种，如 locally weighted 回归，logistic 回归，等等。如果得到的预测函数得出的结果是离散的，我们把这种学习问题叫做**分类问题**

回归的最早形式是[最小二乘法](https://zh.wikipedia.org/wiki/%E6%9C%80%E5%B0%8F%E4%BA%8C%E4%B9%98%E6%B3%95)，由 1805 年的勒让德(Legendre)[1]，和 1809 年的高斯(Gauss)出版[2]。勒让德和高斯都将该方法应用于从天文观测中确定关于太阳的物体的轨道（主要是彗星，但后来是新发现的小行星）的问题。 高斯在 1821 年发表了最小二乘理论的进一步发展[3]，包括高斯－马尔可夫定理的一个版本。