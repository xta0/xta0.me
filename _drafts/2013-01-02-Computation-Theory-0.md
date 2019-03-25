---
list_title: 计算概论（一）| Computation Theory | Finite Automata
title: 有限自动机
layout: post
mathjax: true
---


## Finite automata
 
### Informal introduction to FA

有限状态机可以用来描述变量之间的状态转移，它包含：

1. Information represented by its **state**. 
2. State changes in response to **inputs*.
3. Rules that tell how the state changes in response to inputs are called **transitions**.

举一个网球比赛的例子，假设有一场五局三胜制的网球比赛，每局有六盘，每盘至少得4分才能赢，平分的情况下，一方连续得两份才能赢，我们希望用一种语言来记录每盘比赛比分的过程，改怎么做呢？由于每盘比赛可以产生的比分状态数量是有限的，因此我们可以设计一个有限状态机来描述一场网球比赛比分的所有过程。如下图

<img src="{{site.baseurl}}/assets/images/2013/01/FA-01-tenis.jpg" style="display:block; margin-left:auto; margin-right:auto"/>

我们以0-0处为起始点，到发球方胜利或者失败为终点，枚举了有所比分状态转移的情况，FA的任务就是处理一系列输入的字符来还原一场网球比赛的比分过程。加入我们有下面字符串

<code text-align="center">
s o s o s o s o s o s s
</code>

我们从`s`开始令FA依次读取上面字符，按照箭头走向即可以还原出整场比赛的过程，在FA中使用`*`表示当前状态。因此，对于FA的输入是一串字符串，FA的输出是一个终止状态。

### DFA(Deterministic finite automta)

1. **Alphabets**
    - 字母表是一组有限符号的集合，记作alphabet$\sum$
        - ASCII码集合，Unicode集合
        - `{0,1}`二进制集合
        - `{a,b,c}`，上面网球例子中的`{s,o}`集合等等
2. **string** 
    - 是由alphabets集合 $\sum$ 中一系列字符组成的list
        - $\sum{*}$ 表示alphabets集合中能表示的所有字符串<mark>集合</mark>
        - 字符串的长度表示字符个数
        - $\epsilon$ 表示空串集合（字符串长度为0）
        - 子串
        - 子序列
    - Example
        - `{0,1}* = { ϵ,0,1,00,01,10,11,000, 001, ... }`
3. **Language**
    - A language is a subset of $\sum{*}$ for some alphabet $\sum$


DFA的定义如下

1. 一个确定的状态集合，用 $Q$ 表示
2. 一组输入的字符，用 $\sum$ 表示
3. 一个状态转移函数（正则表达式），用 $\delta$ 表示
4. 一个初始状态，用 $q_0 表示，$q_0$ 属于 $Q$ 的一部分
5. 一组最终状态，用 $F$ 表示，$F \subseteq Q$
    - 叫Final State，也可以叫Accepting State

- 状态转移函数
    - 两个参数
        - a state 
        - an input symbol
    - $\delta(q,a) $ 
        - 表示当前状态Id为`q`，输入为`a`时
        - 下一个状态id的值


### Resource

- [Automata Theory, Languages and Computation](http://infolab.stanford.edu/~ullman/ialc.html)
- [Foundations of Computer Science](http://i.stanford.edu/~ullman/focs.html)