---
layout: post
title: Stanford Algorithm Course-1
mathjax: true
---

### Course Topics
- Vocabulary for design and analysis of algorithms
    - "Big-O" notation
    - "Sweet spot" for high-level reasoning about algorithms
- Divide and conquer algorithm design paradigm
    - Will apply to : Integer multiplication, sorting, matrix multiplication, closet pair.
    - General analysis methods
- Randomization in algorithm design
    - Will apply to : Quick sort, primality testing, graph partitioning, hashing
- Primitives for reasoning about graphs
    - Connectivity information, shortest paths, structure of information and social networks
- Use and implementation of data structures
    - Heaps, balanced binary search trees, hashing and some variants

### Merge Sort: Motivation and Example

- Why Merge Sort
    - Good introduction to divide & conquer
        - Improves over Selection, Insertion, Bubble sorts
    - Calibrate your preparation
    - Motivates guiding principles for algorithm analysis
        - worst - case
        - asymptotic analysis
    - Analysis generalizes to "Master Method"

- Merge Sort Pseudocode

```
C = output [lenght = n]
A = 1st sorted array[n/2]
B = 2nd sorted array[n/2]
i = 1
j = 1

for k=1 to n
    if A[i]<B[j]
        C[k] = A[i]
        i++
    else
        C[k] = B[j]
        j++
```

- Running Time of Merge Sort
    - 归并排序需要的代码执行次数：`6nlog2n + 6n`, `n`为数组维度
    - 推导
        - 使用`recursion tree`
        - 对递归树的每一层`j=0,1,2...,log2n`，有
            - <math><msup><mi>2</mi><mi>j</mi></msup></math/>个节点，每个节点代表一个需要继续递归的子数组
            - 每个子数组的大小为<math><mi>n</mi><mo>/</mo><msup><mi>2</mi><mi>j</mi></msup></math>
            - 由伪代码可以推导出，合并需要的执行次数为`6m`
        - 对第`j`层，一共需要的执行次数为：
            - 所有子数组的个数 x 每个子数组合并需要的次数，即<math><msup><mi>2</mi><mi>j</mi></msup><mo> * </mo><mn>6</mn><mo stretchy="false">(</mo><mi>n</mi><mo>/</mo><msup><mi>2</mi><mi>j</mi></msup><mo stretchy="false">)</mo><mo>=</mo><mn>6</mn><mi>n</mi></math/>
        - 总共的执行次数为：<math><mn>6</mn><mi>n</mi><mo stretchy="false">(</mo><msubsup><mo>log</mo><mn>2</mn><mi>n</mi></msubsup><mo>+</mo><mn>1</mn><mo stretchy="false">)</mo><mo>=</mo><mn>6</mn><mi>n</mi><msubsup><mo>log</mo><mn>2</mn><mi>n</mi></msubsup><mo>+</mo><mn>6</mn><mi>n</mi></math>

- 评估算法的代码执行次数
    - 使用"Worst Case"，不对输入做限制
    - 忽略掉常数项(如`+6n`)和常数系数:`6nlog2n`中的`6`
    > 代码的执行次数会因为编程语言的实现方式不同而有所差异，因此严格定义次数基本做不到，
    - 使用渐进分析，关注N为无穷大时，算法消耗的时间
        
### Big O notation

