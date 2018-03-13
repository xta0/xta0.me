---
layout: post
title: Data Science Overview
---

## A Crash Course in Data Science

> [Course Link](https://www.coursera.org/learn/data-science-course)

### Statistics

Statistics is the discipline of analyzing data. As such it intersects heavily with data science, machine learning and, of course, traditional statistical analysis. These are:

- Descriptive statistics
- Inference
- Prediction
- Experimental Design

**Descriptive statistics** includes exploratory data analysis, unsupervised learning, clustering and basic data summaries. Descriptive statistics have many uses, most notably helping us get familiar with a data set. Descriptive statistics usually are the starting point for any analysis. Often, descriptive statistics help us arrive at hypotheses to be tested later with more formal inference.

**Inference** is the process of making conclusions about populations(群体) from samples. Inference includes most of the activities traditionally associated with statistics such as: estimation, confidence intervals, hypothesis tests and variability. Inference forces us to formally define targets of estimations or hypotheses. It forces us to think about the population that we're trying to generalize to from our sample.

**Prediction** overlaps quite a bit with inference, but modern prediction tends to have a different mindset. Prediction is the process of trying to guess an outcome given a set of realizations of the outcome and some predictors. <mark>Machine learning, regression, deep learning, boosting, random forests and logistic regression are all prediction algorithms</mark>. If the target of prediction is binary or categorical, prediction is often called classification. In modern prediction, emphasis shifts from building small, parsimonious, interpretable models to focusing on prediction performance, often estimated via cross validation. Generalizability is often given not by a sampling model, as in traditional inference, but by challenging the algorithm on novel datasets. Prediction has transformed many fields include e-commerce, marketing and financial forecasting.

**Experimental design** is the act of controlling your experimental process to optimize the chance of arriving at sound conclusions. The most notable example of experimental design is randomization. In randomization a treatment is randomized across experimental units to make treatment groups as comparable as possible. Clinical trials and A/B testing both employ randomization. In random sampling, one tries to randomly sample from a population of interest to get better generalizability of the results to the population. Many election polls try to get a random sample.

### Machine Learning

Machine learning has been a revolution in modern prediction and clustering. <mark>Machine learning has become an expansive field involving computer science, statistics and engineering.</mark> Some of the algorithms have their roots in artificial intelligence (like neural networks and deep learning).
For data scientists, we decompose two main activities of machine learning. (Of course, this list is non-exhaustive.) These are are
1. **Unsupervised learning** - trying to uncover unobserved factors in the data. It is called "unsupervised" as there is no gold standard outcome to judge against. Some example algorithms including <mark>hierarchical clustering, principal components analysis, factor analysis and k-means<mark>.
2. **Supervised learning** - using a collection of predictors, and some observed outcomes, to build an algorithm to predict the outcome when it is not observed. Some examples include: <mark>neural networks, random forests, boosting and support vector machines</mark>.

We give a famous early example of unsupervised clustering in the computation of the <mark>g-factor</mark>. This was postulated to be a measure of intrinsic intelligence. Early factor analytic models were used to cluster scores on psychometric questions to create the g-factor. Notice the lack of a gold standard outcome. There was no true measure of intrinsic intelligence to train an algorithm to predict it.

For supervised learning, we give an early example, the development of regression. In this, Francis Galton wanted to predict children's heights from their parents. He developed linear regression in the process. Notice that having several children with known adult heights along with their parents allows one to build the model, then apply it to parents who are expecting.

It is worth contrasting modern machine learning and prediction with more traditional statistics. Traditional statistics has a great deal of overlap with machine learning, including models that produce very good predictions and methods for clustering. However, <mark>there is much more of an emphasis in traditional statistics on modeling and inference, the problem of extending results to a population</mark>. Modern machine learning was somewhat of a revolution in statistics not only because of the performance of the algorithms for supervised and unsupervised problems, but also from a paradigm shift away from a focus on models and inference. Below we characterize some of these differences.和传统统计学视角不一样的地方：

For this discussion, I would summarize (focusing on supervised learning) some characteristics of ML as:

- the emphasis on predictions;
- evaluating results via prediction performance;
- having concern for overfitting but not model complexity per se;
- emphasis on performance;
- obtaining generalizability through performance on novel datasets;
- usually no superpopulation model specified;
- concern over performance and robustness.

<mark>In contrast, I would characterize the typical characteristics of traditional statistics as</mark>:

- emphasizing superpopulation inference;
- focusing on a-priori hypotheses;
- preferring simpler models over complex ones (parsimony), even if the more complex models perform slightly better;
- emphasizing parameter interpretability;
- having statistical modeling or sampling assumptions that - connect data to a population of interest;
- having concern over assumptions and robustness.

In recent years, the distinction between both fields have substantially faded. ML researchers have worked tirelessly to improve interpretations while statistical researchers have improved the prediction performance of their algorithms.

- 更多关于统计学和机器学习之间的关系，可阅读：
    - [Rise of the Machines](http://www.stat.cmu.edu/~larry/Wasserman.pdf)
    - [Statistical Modeling: The Two Cultures](http://www2.math.uu.se/~thulin/mm/breiman.pdf)
    - [Classifier Technology and the Illusion
of Progress](https://arxiv.org/pdf/math/0606441.pdf)

### The Structure of a Data Science Project

- 5 Phases
    - Question
        - 搞清楚要解决的是哪类问题，是统计问题还是机器学习问题
        - 拿到数据
    - EDA(Exploratory Data Analysis)
        - Are the data suitable for the question
        - Sketch the solution
    - Formal Modeling
        - 建模，通过各种数据集验证
    - Interpertation
        - 结果的评估，哪些预测结果是合理的，哪些不合理
    - Communication
        - 公布结果