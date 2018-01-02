---
layout: post
title: Scala
tag: Scala
categories: 编程语言

---

<em> 所有文章均为作者原创，转载请注明出处 </em>


##0. Prepare

###初始化Scala环境

- MacOS:


###Eclipse

- Hello.java

- Sheet.sc : REPL，类似交互式命令行，Run on the fly.


##1-1：Prgramming Paradigms

- ***Paradigm***: In science, a *paradigm* describes distinct concepts or thought patterns in some scientific discipline.

Main programming paradigms:

- imperative programming
	
- functional programming
	
- logic programming
	
	
Orthogonal to it:

- Object-oriented programming


###Review: Imperative Programming

Imperative programming is about:

- modifying mutable varaibles

- using assignments

- and control structure such as if-then-else, loops, break, continue, return.

The most common informal way to understand imperative programs is as instruction sequences for a Von Neumann computer：

Processer <---Bus---> Memory

###Imperative Programs and Computers

There's a strong correspondence between

- mutable variables <-> memory cells

- variable dereferences <-> load instructions

- variable assignments <-> store instructions

- control structures <-> jumps

*Problem*: Scaling up. How can we avoid conceptualizing programs word by word ? 可扩展性太差，单纯的堆砌指令。

*Reference*: John Backus, Can Programming Be Liberated from the von Neumann Style? Turing Award Lecture 1978. 1950s 发明Fortran，20年后意识到Fortran的局限性。

###Scaling up

In the end, pure imperative programming is limited by the "Von Neumann" bottleneck:

*One tends to concepualize data structures word-by-word.*

We need other techniques for defining high-level abstractions such as collections, polynomials, geometric shapes, strings, documents,...

Ideally: Develop *thories* of collections, shapes, strings...

###What is a Theory?

A theory consists of :

- one or more data types 数据类型

- operations on these types 数据操作

- laws that describe the relationships between values and operations 它们之间的关系

**Normally, a theory does not describe mutations!**

###Theories without Mutation

For instance the theory of polynomials defines the sum of two polynomials by laws such as:

`( a * x +b ) + (c * x +b ) = (a+c)* x + (b+d)`

But it does not define an operator to change a coefficient while keeping the polynomial the same ! 

但是Theory中并没有关于mutation的定义，一个多项式的系数如果用代码表示:

```
class Polynomial { double[] coefficient; }

Ploynomial p = ...;

p.coefficient[0] = 42;

```

p的系数可以从外部被修改而变化，而p却不变，这显然违背了数学公式的唯一性：当系数确定时，多项式p是唯一的，当系数变化时，产生新的多项式p。因此，数学函数是没有mutation的，但是将这个模型映射到程序中，却很难做到。


###Consequences for Programming

If we want to implement high-level concepts following their mathematical theories, there's no place for mutation.

承接上面的问题，如果我们想把数学模型映射到程序中，是不允许mutation存在的。

- The theories do not admit it.

- Mutation can destroy useful laws in the theories

Therefore, let's :

- concentrate on defining theories for operators expressed as functions

- avoid mutations

- have powerful ways to abstract and compose functions.

###Functional Programming

- In a *restricted* sense, FP means programming without mutable variables, assignments, loops, and other imperative control structures.没有变量，赋值，循环，过程性控制语句

- In a *wider* sense, functional programming means focusing on the functions

- In particular, functions can be values that are produced, consumed, and composed.functions in a FP language are first-class citizens.

	- they can be defined anywhere, including inside other functions
	
	- like any other value,they can be passed as parameters to functions and returned as results
	
	- as for other values, there  exists a set of operators to compose functions.

###Some functional programming lanugages

- In the restriced sense:

	- Pure Lisp, XSLT, XPath, XQuery, FP
	- Haskell(without I/O, Monad, or UnsafePerformIO)
	
- In the wider sense:

	- Lisp, Scheme, Racket, Clojure
	
	- SML, OCaml, F#
	
	- Haskell(full language)
	
	- Scala
	
	- Smalltalk Ruby(!) 支持block的OOP
	
### History of FP lauguages

- 1959 Lisp

- 1975-77 ML,FP,Scheme

- 1978 Smalltalk

- 1986 SML

- 1990 Haskell, Erlang

- 1999 XSLT

- 2000 OCaml

- 2003 Scala, XQuery

- 2005 F#

- 2007 Clojure

###Recommended Book

- <SICP>: A classic. Many parts of the course and quizzes are based on it, but we change the language from Scheme to Scala.

- <Programming in Scala>: The standard language introduction and reference

###Why Functional Prgramming?

- simpler reasoning principles

- better modularity

- good for exploiting parallelism for multicore and cloud computing.

To find out more see the video of my 2011 Oscon Java keynote:

[Working Hard to Keep it Simple](https://www.youtube.com/watch?v=3jg1AheF4n0)

###Keynote of <Working Hard to Keep it Simple>

- 多核对并行计算有要求，但是现代的编程语言缺乏对这方面的支持

- Parallel Programming vs Concurrent Programming

	- 并行编程主要利用CPU多核特性，使任务处理变的快速，但是任务处理可以使有序的
	
	- 并发编程指的是同一时刻需要处理许多任务
	
	- Both too hard to get right!
	
- The Root of The Problem

	- 多线程引起的不确定性
	
	- 不确定性 = 平行计算 + mutable state
	
	- 如果要保证并行计算的确定性，首先要avoid mutable state
	
	- Avoiding mutable state 意味着 programming functionally


- Scala:

	- Agile, with lightweight syntax
	
	- OO
	
	- Functional
	
	- Safe and performant, with strong static typing

- Different Tools for Different Purposes

	- Parallelism:
	
		- Collections : Parallel Collections
		
		- Collections : Distributed Collections
		
		- Parallel DSLs
	
	- Concurrency:
	
		- Actors
		- Software transactional memory
		- Futures




##1-2:Elements of Programming

Every non-trivial programming language provides:

- primitive expressions representing the simplest elements

- ways to combine expressions

- ways to abstract expressions, which introduce a name for an expression by which it can then be referred to

###REPL

- 进入scala命令行:`scala` / `sbt console`

- 退出scala:`:quit`

###Evaluation

A non-primitive expression is evaluated as follows/

1. Take the leftmost operator
2. Evaluate its operands (left before right)
3. Apply the operator to the operands

A name is evaluated by replacing it with the right hand side of its definiation

The evaluation process stops once it results in a value

- example:

```java

scala> def square(x:Double) = x*x
square: (x: Double)Double

scala> def sumOfSquares(x:Double,y:Double)  = square(x)+square(y)
sumOfSquares: (x: Double, y: Double)Double

```

###Parameter and Return Types

Function parameters come with their type, which is given after a colon

```Scala

def power(x: Double, y:Int) :Double = ...

```
If a return type is given, it follows the parameter list.

Primitive types are as in Java, but are written capitalized

`Int` : 32-bit integers

`Double` : 64-bit floating pointer numbers

`Boolean` boolean values `true` and `false`

###Evaluation of Function Applications

- Evaluate 函数参数，从左到右

- 将函数按照等号右边展开

- 将函数内部的参数替换成传入的参数

- Example:

```Scala

sumOfSquare(3,2+2)

sumOfSquare(3,4)

square(3) + square(4)

3*3 + square(4)

9 + 4*4

9 + 16

25

```

###The substitution model

- This scheme of expression evaluation is called *substitution model* （这种规则被称为替换模型）

- the idea underlying this model is that all evaluation does is *reduce an expression*  to a value. （思想是所有对表达式的求值都会reduce出一个value，类似于代数求值，也可以理解为所有表达式都会最终有一个输出）

- It can be applied to all expressions, as long as they have no side effects.（他可以被应用到所有的表达式中，但是表达式中不允许出现side-effect：不符合替换模型的表达式，如 a++;）

- The substitution model is formalized in the *lamda-calculus*, which gives a foudation for function programming.（这种替换模型被标准化为lamda表达式，奠定了FP的基础。）

 
###Termination

- Does every expression reduce to a value (in a finite number of steps)?

- No. Here is a counter-example:

```
def loop: Int = loop

loop;

```

###Chaning the evaluation strategy

The interpreter reduces function arguments to values before rewriting the function application.

One could alternatively apply the function to unreduced arguments.

example:

```Scala

sumOfSquare(3,2+2)

square(3) + square(2+2)

3*3 + square(2+2)

9 + (2+2)(2+2)

9 + 4*4

25

```


Scala的解释器在处理函数的参数时，一种方式是将参数先进行求值后代入，另一种方式是lazy的形式，参数一直带入到最后再求值。

- 前一种称为 *call-by-value* ， 它的优势是对入参表达式只进行一次求值

- 后一种称为 *call-by-name* ， 它的优势是如果参数在函数中不会使用，那么就不会进行求值运算



##1-3: Evaluation Strategies and Termination


Scala中通常使用CBV，但是如果参数用`=>`修饰则代表该参数用CBN求值

- Example:

```
def constOne(x:Int, y: => Int) = 1

```

##1-4 Conditionals and Value Definiation

### Conditional Expressions

To express choosing between two alternatives, Scala使用条件表达式：

```
if - else

```

它和Java中的`if-else`很像，但是在imperative programming中，`if-else`称为statement，函数型语言中没有statement，只有expression，而根据Dan的课程，expression要具备三点：

- syntax

- type checking rules

- evaluation rules

具体可以参考ML中关于`if-else`的描述

- Example:

```
def abs(x: Int) = if ( x >= 0 ) x else -x

```

### Value Definitions

上面已经看到，函数的参数可以通过CBV和CBN两种方式求值，同样的规则也适用于定义函数。

- `def` 是CBN，等号右边在函数被调用的时候再进行求值

- `val` 是CBV，等号右边在函数定义的时候进行求值

```
val x = 2
val y = square(x)

```
Afterwards, the name refers to the value.

For instance, y above refers to 4, not square(2)

### Value Definitions and Termination

`val`和`def`的区别当等号右边的表达式为死循环时，便可以体现的很清楚：

```
def loop : Boolean = loop

```

- A definition:

```
def x = loop

```
没问题，等号右边不会求值。

- Value:

```
val x = loop

```
死循环，等号右边定义的时候进行了求值。 


## 1-5: Example

对于Recursive的函数，函数的返回值需要显式声明，否则返回值是optional的。

## 1-6: Blocks and Lexical Scope

###Nested functions

一般来说，一个任务可以分解为多个函数来实现，但有些函数只是helper，并只会被调用一次，它不需要暴露出来，这时候需要考虑使用block：

```Scala

def cal(x:Double) = 
{
	def func1(x:Double):Double = x+1
	
	def func2(x:Double):Double = x*x
	
	func1(x) + func2(x)

}

```

###Blocks in Scala

- block用{...}表示

```Scala

{
	val x = f(3)
	x*x
}

```
- 它里面包含一系列definition 或者 expression

- The last element of a block is an expression that defines its value(最后一个值为block的返回值)

- This return expression can be preceded by auxiliary definitions

- Blocks are themselves expressions; a block may appear everywhere an expression can. (意思是block是first class的)

###Blocks and Visibility

```
val x = 0

def f(y:Int) = y+1

val result = {

	val x = f(3)
	x*x

}

```

- The definitions inside a block are only visible from within the block.

- the definitions inside a block *shadow* definitions of the same names outside the block


###Lexical Scoping

Definitions of outer blocks are visible inside a block unless they are shadowed.

Therefore, we can simplify `sqrt` by eliminating redundant occurrences of the x parameter, which means everywhere the same thing.


###Semicolons

Scala中分号是optional的，但是多个表达式在一行时，分号是不能省的:

```Scala

val y = x+1; y*y

```

###Summary

You have seen simple elements of functional programming in Scala

- arithmetic and boolean expressions

- conditional expressions if-else

- functions with recursion

- nesting and lexical scope

- CBN and CBV

- You have learned a way to reason about program execution: reduce expressions using the substitution model.

## 1-7: Tail Recursion


###Review: Evaluating a Function Application

One simple rule: One evaluates a function application `f(e1,e2,...,en)`

1. 对`e1...en`进行求值，得到`v1...vn`

2. 把等号右边的函数体展开

3. 用`v1,...,vn`替换函数内的参数`e1,...,en`

###Application Rewriting Rule

This can be formalized as *rewriting of the program itself*

```Scala

//定义一个函数f
def f(x1,...,xn) = B; ...f(v1...vn)

//调用了这个函数
f(v1,...,vn)

//将f展开
[v1/x1,...,vn/xn]B

```

- Here, `[v1/x1,...,vn/xn]` 意味着:

	- The expression `B` in which all occurrences of `xi` have been replaced by `vi`
	- `[v1/x1,...,vn/xn]` is called a *substitution*


### 例1:

```Scala

def gcd(a:Int, b:Int) : Int = 
	if (b == 0) a else gcd(b,a%b)

```

`gcd(14,21)` is evaluated as follows:

`gcd(14,21)`

- -> `if(21 == 0) 14 else gcd (21,14%21) `
- -> `if(false) else gcd (21,14%21) `
- -> `gcd (21,14%21)`
- -> `gcd (21,14)`
- -> `if(14) == 0 21 else gcd (14,21%14) `
- -> ...
- -> `gcd (14,7)`
- -> ...
- -> `gcd (7,0)`
- -> `if (0==0) 7 else gcd(0,7%0)`
- -> `7`

### 例2:

```Scala

def factorial(x:Int):Int = 

	if (x == 0 ) 1 else x*factorial(x-1) 

```

```
factorial(4)

```

- -> `if (4==0) 1 else 4*factorial(4-1)`
- -> ...
- -> `4*factorial(3)`
- -> `4*3*factorial(2)`
- -> `4*3*2factorial(1)`
- -> `4*3*2*1*factorial(0)`
- -> `4*3*2*1*1`
- -> 24


###Tail Recursion

***Implementation Consideration***: if a function calls itself as its last action, the function's stack frame can be reused. This is called *tail recursion*

意思是如果一个函数，在最后递归调用自己，则它的stack可以被复用。它的性能变成和使用`for`循环相同。

例1中，最后一个`else`会递归，由于它后面没有其它指令了，因此他是*Tail Recursive*的，例2中，`else`执行递归后还有一条指令是和`x`相乘，因此它不满足*Tail Recursive*

In general, if the last action of a function consists of calling a function（maybe the same, maybe some other functions）, one stack frame would be sufficient for both functions. Such calls are called `tail-calls`

如果一个函数的最后一条指令是调用另一个函数，那么这种调用叫`tail-calls`，这种情况下一个stack frame可以满足两个函数公用。

但是并不是所有的函数都需要使用尾部递归，JVM允许的递归深度为几百，如果超过这个层次就会stack
 overflow，因此对于递归层次很深的函数，要使用这种技巧。
 
 针对例2，来实现一种满足*Tail Recursive*的方法。
 
```scala
object exercise {
  
def factorial(x:Int):Int = 
{
	def loop(acc:Int, x:Int):Int = 
  			
	if ( x== 0 ) acc
	else
		loop(acc*x,x-1)
		
	loop(1,x) 
}
  
factorial(4)
  
}

```
 
 


# Chap 2

## 2-1:HOF

- first-class values

- like any other value, a function can be passed as a parameter and returned as a result

- This provides a flexible way to compose programs

- Function takes other functions as parameters or that return functions as results are called *HOF*

Let's define:


```Scala

def sum(f: Int=>Int, a:Int, b:Int):Int = 

	if (a>b) 0
	else f(a) + sum(f,a+1,b)
 
```
We can then write:

```Scala
def sumInts(a:Int, b:Int) 		= sum(id,a,b)
def sumCubes(a:Int, b:Int) 		= sum(cube,a,b)
def sumFactorials(a:Int,b:Int) = sum(fact,a,b)

```

where:

```
def id(x:Int):Int = x
def cube(x:Int):Int = x * x * x
def fact(x:Int):Int = if(x==0) 1 else fact(x-1)

```

###Function Types

The type A => B is the type of a *function* that takes an argument of type A and returns a result of type B


###Anonymous Functions

上面的例子中，将函数最为参数的一个副作用是要创建许多子函数。有些时候定义这些子函数有些冗余，这时候可以使用匿名函数


###Anonymous Function Syntax

- Example: A function that raises its argument to a cube:

```scala

(x:Int) => x * x * x

```
Here,`(x:Int) => x * x * x`是匿名函数

- 参数的类型可以省略，他可以被编译器推断

多个参数:

```scala

(x:Int, y:Int) => x+y

```

###Anonymous Functions are Syntactic Sugar

所有的匿名函数都可以用def表示:

```scala

(x1:T1,...,xn:Tn) => E

//等价于

{
	def f(x1:T1,...,xn:Tn) = E;
	f
}

```
因此匿名函数实际上函数的语法糖 = {} + def f

- 例子：

```scala

 //return a function using block
 def sum(f: Int => Int) : (Int,Int)=>Int = 
  
  {
		def sumF(a:Int, b:Int):Int =
		
			if(a>b) 0
			else f(a) + sumF(a+1,b)
		sumF
	
  } 


```
Using  anoymous functions:

```scala

def sumInt(a:Int, b:Int) = sum(x => x, a, b)
def sumCubes(a:Int,b:Int) = sum(x=>x*x*x,a,b)

```
##2-2：Currying

###Motivation

Look again at the summation functions:

```scala

def sumInts(a:Int, b:Int) = sum(x=>x ,a,b)

```

- Question

Note that a and b get passed unchanged from `sumInts` and `sumCubes` into `sum`.

Can we be even shorter by getting rid of these parameters?

###Functions Returning Functions

Let's rewrite sum as follows:

```scala

def sum(f: Int => Int) : (Int,Int)=>Int = 
{
	def sumF(a:Int, b:Int):Int = 
		
		if(a>b) 0
		else f(a) + sumF(a+1,b)
	sumF
	
}

```
`sum` is now a function that returns another function.

The returned function `sumF` applies the given function parameter `f` and sums the result

```scala

def sumIncrease = sum( x=>x+1 )

val result = sumIncrease(2,4)

//也可以直接计算:

val result = sum (x => x+1 )(2,4)

```

Generally,function application associates to the left:

代码中，函数的运算是左结合：

```scala

sum(x=>x+1)(1,10) = (sum(x=>x+1))(1,10)

```

对于一个返回函数的函数来说，scala提供了一种syntax, 以上面的sum函数为例:

```scala

def sum(f:Int=>Int)(a:Int,b:Int) : Int = 
	if(a>b) 0 else f(a) + sum(f)(a+1,b)

```

相当于将参数直接代入进来，使`sum (x => x+1 )(2,4)`这种写法更直观

###Expansion of Multiple Parameter Lists

In general, a definition of a function with multiple parameter lists

```scala

//带有参数列表的函数
def f(args1)...(argsn) = E

```

where `n>1`, 等价于 

```scala

def f(args1)...(argsn-1) = {def g(argsn)=E; g}

```

理解:

- arg1,args2,...,argsn-1实际上都表示匿名函数:
	
	- `def ((f(g)h)z) ... = {def g(x) = E; g}`
	
	- 其中g(x)是所有这些匿名函数代入后的最终函数
	

whare `g` is a fresh identifier. Or for short:


```scala

def f(args1)...(argsn-1) = (argsn => E)

```
上面这个式子使用匿名函数表达：

等号左边是匿名函数的求解，等号右边是最终的函数：输入为argsn， 输出为E

也可定义为：

```scala

def f = (args1 => (args2 => ... (argsn => E)...))

```

这种将匿名函数的返回值作为下一个函数的输入，这种风格叫做currying，命名来自Haskell的创使人：Haskell Brooks Curry(1900-1982)

- 分析`sum`

```scala

def sum(f:Int=>Int)(a:Int,b:Int):Int = E

```

what is the type of sum?

```scala

(Int => Int) => (Int, Int) => Int

- 输入是(Int => Int) 

- 输出是(Int,Int)=>Int

- 因此，上面的式子等价于 `(Int => Int) => ((Int, Int) => Int)`

	- 但是函数是右结合的，因此左后的括号可以免掉


###Summation with Anonymous Functions

使用匿名函数，我们能简化`sum`函数

```scala

def sumInts(a:Int,b:Int) = sum(x=>x, a,b)
def sumCubes(a:Int,b:Int) = sum(x * x * x, a,b)

```

###MapReduce

- map,reduce这种函数的基础就是currying


```Scala

def product(f:Int => Int)(a:Int, b:Int):Int =
  
  		if (a > b) 1
  		else f(a) * product(f)(a+1,b)     //> product: (f: Int => Int)(a: Int, b: Int)Int
  		
def fact(n:Int) = product(x => x)(1,n)          //> fact: (n: Int)Int
  
fact(3)                                         //> res1: Int = 6


def mapReduce(map:Int=>Int, combine:(Int,Int)=>Int,zero:Int)(a:Int, b:Int):Int =
	
		if (a>b) zero
		else combine(map(a), mapReduce(map,combine,zero)(a+1,b))
                                                  //> mapReduce: (map: Int => Int, combine: (Int, Int) => Int, zero: Int)(a: Int, 
                                                  //| b: Int)Int
		
	mapReduce(x=>x,(x,y)=>x*y,1)(2,4)         //> res2: Int = 24
	
	
	def new_product(f:Int=>Int)(a:Int,b:Int) = mapReduce(f,(x,y)=>x*y,1)(a,b)
                                                  //> new_product: (f: Int => Int)(a: Int, b: Int)Int
	def new_fact(n:Int) = new_product(x=>x)(1,n)
                                                  //> new_fact: (n: Int)Int
 	new_fact(4)                               //> res3: Int = 24
 	

```


##2-3: Finding a fixed point of a function

A number `x` is called a *fixed point* of a function `f` if

```
f(x) = x

```

For some functions `f` we can locate the fixed points by starting with an initial estimate and then by applying `f` in a repetitive way.

```
x, f(x), f(f(x)), f(f(f(x))),....

```

until the values does not vary anymore(or the change is sufficiently small).

This leads to the following function for finding a fixed point:

```scala

 	val tolerance = 0.0001                    //> tolerance  : Double = 1.0E-4
 
 	def isCloseEnough(x:Double,y:Double) = abs((x-y)/x)/x < tolerance
                                                  //> isCloseEnough: (x: Double, y: Double)Boolean
 
 	def fixedPoint(f:Double => Double)(firstGuess:Double):Double =
	{
			def iterate(guess:Double):Double = {
			
				val next = f(guess)
				
				if (isCloseEnough(guess,next)) next
				else
						iterate(next)
			}
			
		iterate(firstGuess)
	
	} 



```
###Summary

- 上周我们知道了Function是first-class的

- 这周我们知道HOF可以将function combine起来，可以做入参也可以做返回值

- As a programmer, one must look for opportunities to abstract and reuse.

- The highest level of abstraction is not always the best, but it is important to know the techniques, so as to use them properly.


##2-4:Scala syntax summary

We have seen language elements to express types, expressions and definations.

Below, we give their context-free syntax in Extended Backus-Naur form(EBNF), where:

- `|` denotes an alternative

- `[...]` an option(0 or 1)

- `{...}` a repetition (0 or more)

###Types

Type         = SimpleType | FunctionType

FunctionType = SimpleType `=>` Type

			 | `( [Types] )` `=>` Type
			 
SimpleType 	 = Ident

Types        = Type {',' Type} 

A type can be:

- A *numeric type* : Int, Double, Byte, Short, Char, Long, Float

- *Boolean type* : true,false

- *String type* 

- *function type*, like `Int => Int`, `(Int,Int) => Int`

###Expressions

An expression can be:

- An *identifier* such as` x, isGoodEnough`

- An *literal*, like `0,1.0,"abc"`

- An *function application*, like `sqrt(x)`

- An *operator application*, like `-x,y+x`

- An *selection* , like `math.abs`

- An *conditional expression*, like `if(x<0) -x else x`

- A *block* like `{val x= math.abs(y) ; x+2}`

- An *anonymous function*, like `x => x+1`

###Definiations:

**Definition:**

- A *function definition* like `def sum(x:Int,y:Int) = x+y`

- A *value definitions* like `val y = square(2)`


**Parameter**

- A *call-by-value param*, like`(x:Int)`

- A *call-by-name para* like `(y: =>  Double)`



##3-1: Functions an Data

In this section, we'll learn how functions create and encapsulate data structures

###Example

Rational Numbers 有理数

We want to design a package for doing rational arithmetic 有理数运算.

A ration number `x/y` is reresented by  two integers:

- 分子`x`
- 分母`y`

###Classes

In Scala, we do this by defining a *class* :

```scala

class Rational(x:Int, y:Int)
{
	def numer = x;
	def denom = y

}

```

This definition introduces two entitiles:

- A new *type*, name Rational.

- A *constructor* Rational to create elements of this type.

Scala keeps the names of types and values in *different namespaces*

So there's no confict between the two definitions of Rational

###Objects

We call the elements of a class type *objects*.

We create an object by prefixing an application of the constructor of the class with the operator new.

- Example:

```scala

new Rational(1,2)

```

###Members of an Object

Objects of the class Rational have two *members*, numer and denom.

We select the members of an object with the infix operator'.'(like in Java).

- Example

```Scala

val x = new Rational(1,2)                 //> x  : week3.Rational = week3.Rational@2c641e9a
x.numer                                         //> res0: Int = 1
x.denom                                         //> res1: Int = 2


```

###Methods

One can go further and also package functions operating on a data abstraction in the data abstraction itself.

Such functions are called *methods*.

- Example

Rational numbers now would have, in addition to the functions *numer* and *denom*, the functions *add,sub,mul,div,equal,toString*

```scala

class Rational(x:Int, y:Int){

	def numer = x
	def denom = y

	def add(that:Rational) = new Rational(numer*that.denom + denom*that.numer,
	 denom*that.denom)
	 
	 def neg:Rational = new Rational(-numer,denom)
	 
	 def sub(that:Rational) = add(that.neg)
	 
	 override def toString = numer + "/" + denom
}

```


##2-6: More Fun With Rationals

###Rationals with Data Abstraction

```scala

class Rational(x:Int, y:Int){

	def numer = x/g
	def denom = y/g
	
	private def gcd(a:Int, b:Int) : Int = if (b == 0) a else gcd(b,a%b)
	 
	private val g = gcd(x,y)
	 
	...
	
}
```

定义私有成员`g`和私有函数`gcd`

It is equally possible to turn number and denom into vals, so that they are computed only once:


```Scala

class Rational(x:Int, y:Int){

	val numer = x/gcd(x,y)
	val denom = y/gcd(x,y)
	
	...
}

```

这种优化和改动都在这个类的里面，并不影响调用者，这种对数据的封装成为*data abstraction*

###Self Reference

On the inside of a class, the name `this` represents the object on which the current method is executed.

- Example:

```scala

class Rational(x:Int, y:Int){

...

def less(that:Rational) = numer * that.denom < that.numer * denom

def max(that:Rational) = if(this.less(that)) that else this

...	 
	 
}
	

```

###Precondition

可以在类中加入对参数的合法性校验:

```scala

class Rational(x:Int, y:Int){

	require(y != 0, "denominator can not be 0")
	
	...

}

```

`require` is a predefined function.

It takes a condition and an optional message string.

if the condition is false than `IllegalArgumentException` is thrown with the given message string.


###Assertions

Besides `require`, there is also `assert`

Assert also takes a condition and an optional message:

```scala

val x = sqrt(y)
assert(x >= 0)

``` 

和require一样，assert在条件为false的时候也会抛出异常，但是异常类型为`AssertionError`。

assert和require的目的不同：

- `require` is used to enforce a precondition on the caller of a function.

- `assert` is used as to check the code of the function itself.


###Constructor

默认情况下，Scala的类有默认的构造函数，如果想增加构造函数，需要将`this`作为函数使用：

```scala

class Rational(x:Int, y:Int){

	 require(y != 0, "denominator can not be 0")

	 def this(x: Int) = this (x,1)
	 
	 ...

}

val z = new Rational(3)

```

##2-7: Evaluations and Operations

### Classes and Substitutions

我们之前可以使用substitution规则去分析函数，这节将这个规则用于class object

- *Question:* How is an instantiation of the class `new C(e1,...,em)`

- *Answer:* The expression arguments `e1,...,em` 参数和函数函数一样求值，得到：`new C(v1,...,vm)`


我们现在定义这样一个类:

```
class C(x1,...,xm){ ... def f(y1,...,yn) = b ...}

```

where:

- 类的构造函数入参为：`x1,...,xn`

- 类中定义个了一个函数`f`,参数为`y1,...,yn`

- *Question:* How is the following expression evaluated?

```
new C(v1,...,vm).f(w1,...,wn)

```

###Operators

上面的例子中，求两个有理数的和，代码为`x.add(y)`，这种没有`x + y `直观，因此，scala提供一种方式使方法调用变的更直观：

它由两部分syntax构成：

- Step1: Infix Notation:

Any method with a parameter can be used like an infix operator.

it is therefore possible to write:

```
r add s  	=> 		r.add(s)

r less s 	=> 		r.less(s)

r max s		=>		r.max(s)

```

- Step2: 定义操作符

Scala可以重载操作符：


```scala

 def < (that:Rational) = numer * that.denom < that.numer * denom
 
```

调用变为：

```
x.less(y)  等价于 x < y

```


##2-8: Class Hierarchies

###Abstract Classes

Consider the task of writing a class for sets of integers with the following operations

定义一个可以表示整数集合的类

```scala

abstract class IntSet{

	def incl(x:Int):InSet
	def contains(x:Int):Boolean


}


```

抽象类:

IntSet is an *abstract class*.

抽象类可以定义接口，但不需要实现。
抽象类不可以直接用*new*创建


###Class Extensions

Let's consider implementing sets as binary trees.

There are two types of possible trees: a tree for the empty set, and a tree consisting of an integer and two sub-trees.

Here are their implementations

```scala

class Empty extends IntSet
{
	def contains(x:Int):Boolean = false
	
	def incl(x:Int):IntSet = new NonEmpty(x,new Empty, new Empty)

}

class NonEmpty(elem: Int, left: IntSet, right:IntSet) extends IntSet
{
	def contains(x:Int):Boolean = 
	
		if (x<elem) left.contains(x)
		else if (x>elem) right.contains(x)
		else true

	def incl(x:Int):IntSet = 
		
		if(x<elem) new NonEmpty(elem,left.incl(x), right)
		else if(x>elem) new NonEmpty(elem, left, right.incl(x))
		else this

}

```

###Base Classes and SubClasses

- `IntSet` is called the *superclass* of Empty and NonEmpty.

- Empty and NonEmpty are *subclasses* of `IntSet`.

- In Scala, any user-defined class extends another class.

- If no superclass is given , the standard class object in the Java package `java.lang` is assumed

- The direct or indirect superclasses of a class C are called *baseclasses* of C

- So, the base classes of NonEmpty are IntSet and Object

###Implementing and Overriding

如果父类中的方法没有implementation，那么子类可直接实现，如果父类方法中有implementation，那么子类需要override：

```scala

abstract class Base
{
	def foo = 1
	
	def bar:Int
}

class Sub extends Base
{
	override def foo = 2
	
	
	def bar = 3

}

```

###Object Definitions

上面的例子中，Empty类其实没有变化，但是每次都要重新创建，如果把它定义为一个单例，则能省去许多重复创建的instance，在Scala中，使用`object`:

```scala

object class Empty extends IntSet
{
	def contains(x:Int):Boolean = false
	
	def incl(x:Int):IntSet = new NonEmpty(x,Empty,Empty)
	
	override def toString = {"."}

}

```

- This defines a *singleton object* named Empty.

- No other Empty instances can be created.

- Singleton objects are valeus , so Empty evaluates to itself

###Programs

之前的代码都是使用REPL或workSheet运行，这一节将会创建一个Scala Object：

```scala

package week4

object week4 {

  def main(args:Array[String]) = println("hello")
 
}

```

###Dynamic Binding

Object-oriented languages(including Scala) implement *dynmaic method dispatch*.

This means that the code invoked by a method call depends on the runtime type of the object that contains the method.

- Example

```

Empty contains 1

-> [1/x][Empty/this] false

= false

```

###Something to Ponder

Dynamic dispatch of methods is analogous to calls to higher-order functions.

- Question:

Can we implement one concept in terms of the other?

- Objects in terms of higher-order functions?

- HOF in terms of objects?


##2-9: How class are organized

默认import的包：

- All members of package scala

- All members of package java.lang

- All members of the singleton object scala.Predef.


Here are the fully qualified names of some types and functions which you have seen so far:

```
Int 		=> scala.Int
Boolean		=> scala.Boolean
Obejct		=> java.lang.Object
require 	=> scala.predef.require
assert		=> scala.predef.assert

```

###Scala Doc

www.scala-lang.org/api/current


###Traits

In Java, as well as in Scala, a class can only have one superclass.

But what if a class has several natural supertypes to which it conforms or from which it wants to inherit code?

Here, you could use traits.

A trait is declared like an abstract class, just with `trait` instead of `abstract` class.

Trait类似Java中的interface

```scala

trait Planar
{
	def height: Int
	def width : Int
	def surface = height*width
}

```

OBject只能继承一个class，但可以实现多个trait


```scala

class Square extends Shape with Plannar with Movable...

```
Traits resemble interfaces in Java, but are more powerful because they can contains fields and concrete methods.

On the other hand, traits cannot have parameters, only classes can

Trait中不能定义property

###Top Types

Scala有几个基本类型：

- Any : the base type of all types, Method: '==', '!=', 'equals', 'hashCode', 'toString'

- AnyRef : The base type of all reference types; Alias of 'java.lang.Object'

- AnyVal : The base type of all primitive types.

###The Nothing Type

Nothing is at the bottom of Scala's type hierarchy. It is a subtype of every other type.

There is no value of type Nothing.

Why is that useful?

- To signal abnormal termination

- As an element type of empty collections

###Exceptions

Scala's exception handling is similar to Java's

The expression:

```scala
def error(msg:String) = throw new Error(msg) 

```

###The Null Type

Every **reference class type** also has null as a value.

The type of `null` is `Null` .

`Null` is a subtype of every class that inherits from `Object`; it is incompatible with subtypes of `AnyVal`

```scala

val x = null //x:Null = null

val y:String = null // y:String = null

val z:Int = null //error: type mismatch


```

```scala

if (true) 1 else false                          //> res0: AnyVal = 1

```

if的第一个分支返回的类型为Int，第二个分支返回的类型为boolean，那么表达式的类型应该是他们共同的类型：AnyVal

##2-10: Polymorphism

以构建一个`Con-Lists`为例，`Con-List`是一个immutable linked list

它可以通过两种方式构建:

- `Nil` : 构建一个empty列表

- `Cons` 一个包含一个element的cell和其余的list部分

###Cons-Lists in Scala

`Con-List`可以通过如下数据结构标示:

```scala

trait IntList
{
  
}

class Cons(val head:Int, val tail:IntList) extends IntList
{
  

}

class Nil extends IntList
{
  
}

```

`Cons`的构造函数中参数用`val`声明了，这种情况相当于自动给成员变量赋值:

```scala

class Cons(_head:Int, _tail:IntList) extends IntList
{
  	val heal = _head;
  	val tail = _tail;
}

```


###Type Parameters

仅仅定义int类型的list太单一了，希望将其扩展到double，string类型，这时候需要使用泛型:

```scala

trait List[T]

class Cons[T](val head:T, val tail:List[T]) extends List[T]

class Nil[T] extends List[T] 

```

定义`trait`:

```scala

trait List[T]
{
  def isEmpty:Boolean
  
  def head:T
  
  def tail:List[T]
  
}

```

定义`Cons`,`Nil`

```

class Cons[T](val head:T, val tail:List[T]) extends List[T]
{
  def isEmpty:Boolean = false

}

class Nil[T] extends List[T] 
{
	def isEmpty:Boolean = true
	
	def head:Nothing = throw new NoSuchElementException("Nil.head")
	
	def tail:Nothing = throw new NoSuchElementException("Nil.head")
}

```

###Generic Functions

和类一样，函数也支持泛型参数[T], 例如下面这个函数创建一个泛型List

```scala

def singleton[T](elem:T) = new Cons[T](elem, new Nil[T])

singleton[Int](1)
singleton[Boolean](true)

```
###Type Inference

事实上，编译器可以对参数类型进行推断,[T]可以省略：

```
singleton(1)
singleton(true)

```

### Types and Evaluation

类型并不影响Scala进行参数替换，根据Substitution规则，参数被代入后，在执行表达式求值前，所有的类型都被remove掉。

这一特性叫做:`type erasure`

使用这一特性的语言有: Java,Scala,Haskell,ML,OCaml.

有些语言在runtime中仍然带着类型：C++,C#,F#

### Polymorphism

这次来自希腊，意思是多种形式，In programming, 它表示:

- 函数的参数可以是多种类型的

- 某一种类型可以有很多子类型

我们目前已经看到过这两个特性:

- 子类: 可以将一个父类型的指针指向一个子类型

- 泛型: 函数参数和类的参数可以声明成泛型

###excise

已知一个list: List[T]，要求返回第n个元素:

```scala

def nth[T](n:Int,xs:List[T]):T = 
	
	if (n==0) xs.head 
	else nth(n-1,xs.tail)

```

## 2-11 Functions as Objects

这个section讲object和function之间的关系：

###Functions as Objects

在Scala中，函数被认为是object：

例如`A => B`这种函数类型其实是`scala.Function1[A,B]`的缩写，定义为：

```scala

package scala

trait Function1[A,B]{

	def apply(x:A):B

}

```

所以，函数只是object的`apply`方法，同样的，对于多个函数入参的`trait`还有`Function2`,`Function3`,...

### Expansion of Function Values

匿名函数如下：

```scala

(x:Int) => x*x

```
被展开如下:

```scala

{
	class AnonFun extends Function1[Int,Int]
	{
		def apply(x:Int):Int = x*x
	}
	
	new AnonFun
}

```

或者更简单的，使用匿名类：

```scala

new Function1[Int,Int]
{
	def apply(x:Int) = x*x
}

```

### Expansion of Function Calls

这样的函数调用`f(a,b)`会扩展为：

```scala

f.apply(a,b)

```

```scala

val f = (x:Int) => x*x

f(7)

```

被扩展为:

```scala

val f = new Function1[Int,Int]{ def apply(x:Int) = x*x }
f.apply(7)

```

所以：**Functions are just Objects**

### Functions and Methods

下面的方法：

```scala

def f(x:Int):Boolean = ...

```

由上一小节可知`f`本身并不是函数类型，而是一个对象的`apply`方法。

但是如果`f`被放到了等号右边，需要成为一个函数类型的时候，会自动转换:

```scala

(x:Int) => f(x)

```

上面的匿名函数再展开：

```scala

new Function1[Int,Boolean]
{
	def apply(x:Int) = f(x)
}

```
## Objects Everywhere

到目前，我们
