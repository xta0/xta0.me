---
layout: post
title: PL-Haskell-2
tag: Haskell
categories: PL

---

<em> 所有文章均为作者原创，转载请注明出处 </em>

> http://www.cs.nott.ac.uk/~gmh/book.html

##Chap1:Introduction

###Functional Programming Language

- 以数学公式的方式描述问题

- 例子:

对1到10求和，用Java：

```
total = 0

for(i=0;i<10;i++)
{
	total = total+i
}

```
累计total的计算是通过 variable assignment完成的。

这种style属于imperative programming

计算的过程是对total状态的不断修改，没有返回值

同时，它也不是First-Class的，没有办法将计算过程作为参数传递，代码组织是松散的。

同样的例子，用Haskell实现：

```
sum[1...10]

```
这种style属于declarative programming

计算过程是由不同表达式compose而成的：

首先创建一个list = 1...10

然后将list作为sum的输入，带入到sum中计算，得到结果

**从表达式的角度来说：可以把函数型语言理解为不同表达式的组合，同时每个表达式是独立的，是有输入输出的函数，是可以被单元测试的，将这些表达式compose到一起就是函数型编程**

###Functional Programming background

- 1930s: Alonzo Church develops the ***lamda calculus***, a simple but powerful theory of functions

- 1950s: John McCarthy develops ***Lisp***, the first functional language, with some influences from the lamda calculus, but retaining variable assignments .

> 注：使用assignment是非常诱人的，因为CPU决定了计算过程就是不断向寄存器写入的过程，也就是不断更新寄存器中值的过程，编译器也是为这个过程而设计，因此允许直接赋值是许多编程语言都支持的，但是它却违背了Pure Functional的思路，这种直接赋值也被称为为<em>side effect</em>

- 1960s: Peter Landin develops ***ISWIM***, the first pure functional language, based strongly on the lambda calculus, with no assignments!

- 1970s: John Backus develops ***FP***, a functional language that emphasizes *high-order functions* and *reasoning about programs*.

- 1970s: Robin Milner and other develop ***ML***, the first modern functional language, which introduced *type inference* and *polymorphic types(generic)*.

- 1970s ~ 1980s: David Turner develops a number of *lazy* functional languages, culminating in the ***Miranda*** system.

- 1987s: ***Haskell***



###A Taste of Haskell

qucik sort:

```
f[] = []

f (x:xs) = f ys ++ [x] ++ f zs
			
			where
			
				ys = [a | a <- xs, a<= x]
				zs = [b | b <- xs, b > x]


```

##Chap2:First Step






