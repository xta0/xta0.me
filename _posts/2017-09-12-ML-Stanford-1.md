---
layout: post
title: Machine Learning - Chap1
meta: Coursera Stanford Machine Learning Cousre Note, Chapter1
categories: [ml-stanford,course]
mathjax: true
---

> 所有文章均为作者原创，转载请注明出处

## Chapter1

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
