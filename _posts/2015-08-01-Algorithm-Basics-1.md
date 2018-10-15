---
layout: post
list_title: 算法基础 | Algorithms | 概述 | Overview
title: 算法概述 | Algorithm Overview
mathjax: true
categories: [Algorithms]
---

## 算法分析

所谓计算机算法，是信息处理的一种手段，它可以借助某种工具，按照一定规则，以明确而机械的形式进行


### 算法特性

- 通用性
	- 对参数化输入进行问题求解
	- 保证计算结果的正确性
- 有效性
	- 算法是有限条指令组成的指令序列
	- 即由一系列具体步骤组成  
- 确定性
	- 算法描述中的下一步应该执行的步骤必须明确
- 有穷性
	- 算法的执行必须在有限步内结束
	- 算法不能含有死循环 

```
算法A解问题P：

1. 把问题P的任何实例作为算法A的输入
2. 每部计算结果是确定的
3. A能够在有限步得到计算记过
4. 输出正确的解
```

### 算法的伪码表示

- 基本符号

```
赋值语句：<- / = 
分支语句：if...then...[else...]
循环语句：while, for, repeat until
转向语句：goto
输出语句：return
调用：直接写过程的名字
注释：//...
```

- 求解最大公约数

```
//输入：非负整数m,n，其中m与n不全为0
//输出：m与n的最大公约数

Euclid(m,n){
	while m>0 do
		r <- n mod m
		n <- m
		m <- r
	return n
}
```

### 复杂度分析的主要方法

- 迭代： 级数求和
- 递归： 递归跟踪 + 递推方程

## 算法设计

- 建模
    - 优化目标
    - 约束条件

### NP-Hard问题

- 类似问题数千个，大量存在与各个应用领域
- 至今没有找到有效算法
    - 现有算法的运行时间是输入规模的指数或更高阶的函数
- 至今没有人能证明对于这类问题不存在多项式时间的算法
- 从是否存在多项式时间算法的角度看，这些问题彼此是等价的。<mark>这些问题的难度处于有效计算的边界</mark>




### Resources

- [CS106B-Stanford-YouTube](https://www.youtube.com/watch?v=NcZ2cu7gc-A&list=PLnfg8b9vdpLn9exZweTJx44CII1bYczuk)
- [Algorithms-Stanford-Cousera](https://www.coursera.org/learn/algorithms-divide-conquer/home/welcome)
- [算法基础](https://www.coursera.org/learn/suanfa-jichu)
- [算法与数据结构-1-北大-Cousera](https://www.coursera.org/learn/shuju-jiegou-suanfa/home/welcome)
- [算法与数据结构-2-北大-Cousera](https://www.coursera.org/learn/gaoji-shuju-jiegou/home/welcome)
- [算法与数据结构-1-清华-EDX](https://courses.edx.org/courses/course-v1:TsinghuaX+30240184.1x+3T2017/course/)
- [算法与数据结构-2-清华-EDX](https://courses.edx.org/courses/course-v1:PekingX+04833050X+1T2016/course/)
- [算法设计与分析-1-北大-Cousera](https://www.coursera.org/learn/algorithms/home/welcome)
- [算法设计与分析-2-北大-EDX](https://courses.edx.org/courses/course-v1:PekingX+04833050X+1T2016/course/)