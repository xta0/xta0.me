---
list_title: 编程语言原理 | Programming Language | SML
title: SML
layout: post
---

> Programming Language课程笔记

### Course Content

- Essential concepts relevant in any programming language
- Use ML,Racket,Ruby
- Big Focus on Functional Programming 
- **3 parts (100-200 hours),11 weeks**
	- Part A
		- Syntax vs. semantics vs. idioms vs. libraries vs. tools
		- ML basics (bindings, conditionals, records, functions)
		- Recursive functions and recursive types
		- Benefits of no mutation
		- Algebraic datatypes, pattern matching
		- Tail recursion
		- Higher-order functions; closures
		- Lexical scope
		- Currying
		- Syntactic sugar
		- Equivalence and effects
		- Parametric polymorphism and container types
		- Type inference
		- Abstract types and modules
	- Part B
		- Racket basics
		- Dynamic vs. static typing
		- Laziness, streams, and memoization
		- Implementing languages, especially higher-order functions
		- Macros
		- Eval
	- Part C
		- Ruby basics
		- Object-oriented programming is dynamic dispatch
		- Pure object-orientation
		- Implementing dynamic dispatch
		- Multiple inheritance, interfaces, and mixins
		- OOP vs. functional decomposition and extensibility
		- Subtyping for records, functions, and objects
		- Class-based subtyping
		- Subtyping
		- Subtyping vs. parametric polymorphism; bounded polymorphism



### Environment Setup

- SML Interpreter
	- `brew install sml`
	- [Commandline reference](http://pages.cs.wisc.edu/~fischer/cs538.s08/sml/sml.html)
- Code Editor
	- [VSCode](https://code.visualstudio.com/)
	- [Standard ML extension](https://github.com/freebroccolo/vscode-sml) 
	
# Week 1

## Variable Bindings and Expressions

- "Let go" of all programming languages you already know"
- <mark>Treat "ML" as a "totally new thing"</mark>
	- Time later to compare / contrast to what you konw
- Start from a blank file 

```Haskell
(* This is a comment. This is our first program. *)
val x = 34; 

(* static enviroment: x : int *)
(* dynamic enviroment: x --> 34 *)
val y = 17;

(* static enviroment: x:int, y:int *)
(* dynamic enviroment: x --> 34, y --> 17 *)
val z = (x+y) + (y+2);

(* static enviroment: x:int, y:int, z:int *)
(* dynamic enviroment: x--> 34, y-->17, z --> 70 *)
val q = z+1

(* static enviroment: x:int, y:int, z:int, q:int *)
(* dynamic enviroment: x--> 34, y-->17, q --> 71 *)
val abs_of_z = if z<0 then 0-z else z;(* bool *)(* int *)

(* dynamic enviroment: ..., abs_of_z --> 70 *)
val abs_of_z_simpler = abs(z)
```

- Static environment: 静态环境是指代码在运行时环境之前（执行前），ML会对变量做类型推断，签名检查，比如`x:int`。
- Dynamic environment: 程序的运行时环境，保存变量当前状态

### A Variable Binding

- 定义
	- `val x = e;`

- Syntax(语法):
	- **Syntax** is just how you write something
	- `val`, `=` , `;` 
	- variable `x`
	- Expression `e`

- Semantics(语义):
	- Syntax is just how you write something
	- **Semantics** is what that something means
		- **Type Checking** (before program runs)
		- **Evaluation** (as program runs)
	
	- For variable bindings:
		- Type-check expresson and extend **static environment**
		- Evaluate expression and extend **dynamic environment** 
		
> 在函数型语言里，没有赋值(assign)的概念，而是叫做binding。一个变量(符号)被bind一个value后，这个变量是不允许再去bind其它的value。
		
## Rules for Expressions

### Expressions

- We have seen many kinds of expressions:
	- `34 true false x e1+e2 e1>e2`
	- `if e1 then e2 else e3`
	
- <mark>Every kind of expression has</mark>
	- **Syntax** 
		- 语法
	- **Type-checking rules** 
		- 类型检查
		- Produces a type or fails(with a bad error message)
		- types so far: `int` `bool` `unit`
		
	- **Evaluation rules**(used only on things that type-check) 
		- 求值规则
		- Produces a value(or exception or infinite-loop)

### 分析

- `a=3`
	- Syntax:
		- sequence of letters,digits,_,not starting with digit
	- Type-checking:
		- Look up type in current **static enviroment**. if not there, fail
	- Evaluation:
		- look up value in current **dynamic enviroment**
	
- `+`
	- Syntax:
		`e1+e2` where `e1` and `e2` are expressions
	- Type-checking:
		- if `e1` and `e2` have type `int`,
		- then `e +e2` has type `int`
	- Evaluation:
		- if `e1` evaluates to `v1` and `e2` evaluates to `v2`,
		- then `e1+e2` evaluates to sum of `v1` and `v2`

- `if-else`
	- Syntax: 
		- if `e1` then `e2` else `e3`
		- where `if`,`then`,`else` are keywords and `e1`,`e2`,`e3` are subexpressions
	- Type-checking:
		- first `e1` must have type `bool`.
		- `e2` and `e3` can have any type `t`, but they must have the same type `t`.     
		- the type of the entire expression is also `t`
	- Evaluation rules:
		- first evaluate `e1` to a value call it `v1`, if the result is ture, then evaluate `e2` as the result of whole expression. else, evaluate `e3` and that result is the whole expression's result.

### Values

- All values are expressions
- Not all expressions are values
- Every value **"evaluates to itself"** in "zero steps"
- Examples:
	- `34`,`17`,`42` have type `int`
	- `true`, `false` have type `bool`
	- `()` has type `unit`

### The REPL and Erros

- 使用命令行解释执行单条语句要加`;`
	- `val x=1;` 
- 读文件`use "foo.sml";`

- Error
	- syntax:
		- what you wrote means nothing or not the construct you intended.
	- Type-checking:
		- What you wrote does not type-checked
	- Evaluation: 
		- It runs but produces wrong answer, or an exception, or an infinite loop
	- common error:
		- `if` - `then` - `else`
		- if takes a `bool` type value
		- `then` and `else` must return the same type of result
		- 负号用`~`表示：`~5`
		- 除号用`div`表示 `10 div 5`

## Shadowing

### Multiple binding of same variable

<mark>shadowing</mark>指的是add a variable to the environment，但是这个variable在environment中已经存在了，下面代码：

```haskell
val a = 10
(* a:int a -> 10 *)

val b = a*2
(* b -> 20 *)

val a = 5 (*  this is not an assignment statement *)
(* a -> 5, b-> 20 *)
```

上面代码中，`a=5`并没有改变原来的`a`值，在ML中是没有办法修改原先内存中的值的。因此这里得到的`a`是一个新的environment中的`a`,后面的变量都注册到这个enviroment中，因此原来的a被shadow掉了。


```haskell
val c = b
(* a -> 5, b -> 20 , c -> 20  *)
```

此时b不是前面的b了，而是新environment中的b

```haskell
val d = a
(* ...,in the current envrioment a -> 5, d->5 *)

val a = a + 1
(*create a new envrioment, a -> 6*)
```
和上面原理相同，在当前的envrionment中`a+1 = 6`，此时需要增加一个variable，也叫`a`，又产生了shadowing。系统会再创建一个新的environment保存新的`a`。

```haskell
val a = <hidden-value> : int
val b = 20 : int
val a = <hidden-value> : int
val c = 20 : int
val d = 5 : int
val a = 6 : int
val it = () : unit
```
查看当前环境的转台，可以看到之前的两个被shadow掉的`a`提示`<hidden-value>`

## Functions(informally)

### Function

- the most important build block in the whole course
	- Like Java methods,have arguments and result
	- But no classes,`self`,`this`,`return`

```haskell
(* val pow = fn : int * int -> int *)
fun pow(x:int, y:int) = 
	if y=0 
	then 1 
	else x*pow(x,y-1)
```

- 函数签名
	- Example: `int * int -> int` 表示有两个参数，类型都是`int`,返回值也是`int`
	- In expressions, `*` is multiplication: `x*pow(x,y-1)`

- Cannot refer to later function bindings
	- That's simply ML's rule
	- Helper functions must come before their uses
	- Need spection construct for mutual recursion(later)
	
> 上面例子可知ML中Function是first-class object, 函数的类型就是它的签名

### Recursion 

- “Makes sense” because calls to same function solve "simpler problems"
- Recursion more powerful than loops
	- We won't use a single loop in ML
	- Loops ofter(not always)obscure simple, elegant solutions
	
<mark>Everything you can do in loop, you can do it in recursion</mark>，使用递归可以取代循环，简化代码

> 由于`for`或`while`是一种过程性的表达式，它表达的是如何完成循环，还需要引入一些状态变量。在函数型语言中，这种做法不直观，是一种过程式的风格


## Functions（formally)

### Function binding

- Syntax : `fun x0（x1:t1,...,xn:tn）= e`
	- `x0`是函数名
	- (will generalize in later lecture)
		
- Type-Checking: 
	- 首先将`x0`的类型绑定为`(t1 *...* tn)->t`
	- 对函数的body`e`进行type-checking, 使用static environment中已有的信息，比如之前创建的binding（函数body中可能使用以前的binding）
	- 对函数的参数类型检查和函数自身的类型检查
		- 函数body中可能出现递归，因此对函数自身也要进行type-checking

- Evaluation : 
	- 运行时求值，对函数的body不进行提前Evaluation
	- Add `x0` to dynamic enviroment so later expression can call it. 
	- Funcation call semantics will also allow recursion		

### More on type-checking

<div style="text-align:center">
	<code>fun x0（x1:t1,...,xn:tn）= e</code>
</div>

- New kind of type : `(t1 *...* tn ) -> t`
	- Result type on right
	- 在static environment中，只有函数对象`x0`的类型。ML函数也是在运行时求值的，因此函数体中使用的binding是在dynamic environment中寻找的
	- 参数只能在函数体内部`e`中使用
	- `x0`的返回值类型是函数体`e`的类型，Type-checker可以根据`e`推断出返回值类型`t`
		
### Function Calls

A new kind of expression:

- Syntax : `e0 (e1,...,en)`
	- 如果只有一个参数，则可以省略括号。ML中不支持可变参数
	- `pow(2,3)`
- Type-Checking:
	- `e0` has some type `(t1 *...* tn ) -> t`，then `e0 (e1,...,en)` has type `t`
		- Example:`pow(x,y-1)` in previouse example has type `int`
	- `e1` has type `t1`, ... , `en` has type `tn`
- Evaluation:
	1. 在当前运行时环境(<mark>current dynamic enviroment</mark>)中对`e0`进行求值
		- `e0`求值之后是一个函数，类型为 :`(t1 *...* tn ) -> t`
	2. 在当前运行时环境中求解参数 `v1,...,vn`
		- 比如参数中有`2+2`，这种情况下需要对其进行求值后再继续evaluate function
	3. Result is evaluation of `e` in an enviroment extended to map `x1` to `v1`, ... `xn` to `vn`
	 	- ("An envrioment" is actually the enviroment where the function was defined, and includes `x0` for recursion)
	
	
## Tuples and Pairs

### Paris

- Syntax : 
	- `(e1,e2)`
- Type-checking: 
	- if `e1` has type `ta` and `e2` has type `tb` then the expression has type `ta * tb`
	- A new kind of type
- Evaluation:
	- Evaluate `e1` to `v1` and `e2` to `v2`; result is `(v1,v2)`
	- A pair of values is a value

#### Access

- Syntax：
	- `#1 e` and `#2 e`
- Evaluation: 
	- Evaluate e to a pair of values and return first or section piece
	- Example: if `e` is a variable x then look up x in enviroment
- Type-checking:
	- if `e` has type `ta * tb` then `#1 e` has type `ta` and `#2 e` has type `tb`

```haskell
fun swap(pr : int*bool) = (#2 pr, #1 pr)

(* (int*int) * (int*int) -> int *) 
fun sum_two_pairs (pr1 : int * int, pr2 : int * int) = (#1 pr1)+(#2 pr1)+(#1 pr2)+ (#2 pr2)

(* int*int -> int *int *)
fun div_mod (x:int,y:int) = (x div y , x mod y)

fun sort_pair(pr:int*int) = 
    if (#1 pr) < (#2 pr)
    then pr 
    else (#2 pr, #1 pr)
```


### Tuples

Actually , you can have tuples with more than two parts.

- (e1,e2,...,en)
- ta * tb * ... * tn
- "#1 e, #2 e, #3 e ..."

### Nesting

Pairs and tuples can be nested however you want

```haskell
val x1 = (7,(true,9)) (* int * (bool*int) *)

val x2 = #1 (#2 x1) (* bool *)

```

## Lists

- Despite nested tuples, tye type of variable still "commits" to a particular "amount" of data 

In contrast, a list:

- Can have any number of elements
- But all list elements have the same type

### Building Lists

- The empty list is a value `[]`

- In general, a list of values is a value; elements separated by commas:`[v1,v2,...,vn]`

> 在SML中list中每个元素的类型是相同的

- if `e1` evaluates to `v` and `e2` evaluates to a list `[v1,...,vn]`,then `e1::e2` evaluates to `[v,..,vn]`

` e1::e2 (*pronounced "cons"*) `example:

```
- val list = 1::[1,2];
val list = [1,1,2] : int list

```

### Accessing Lists

Until we learn pattern-matching, we will use three standard-library functions

- `null e` evaluates to `true` if and only if `e` evaluates to []

```
- null list;
val it = false : bool

```


- if `e` evaluates to `[v1,v2,...,vn]` then `hd e` evaluates to `v1`
	- raise exception if e evaluates to []
	
- if `e` evaluates to `[v1,v2,...,vn]` then `tl e` evaluates to `[v2,...,vn]`
	- raise exception if e evaluates to []
	- Notice result is a list
	

### Type-checking list operations

Lots of new types： For any type `t`, the type `t list` describes lists where all elements have type `t`

数组的类型为`t list`，`t`为任意类型

```
- [1,2,3];
val it = [1,2,3] : int list

```

Examples:
	
`int list` `bool list` `int list list` `(int * int) list` `(int list*int) list`

- So [] can have type t list of any type
	- SML uses type `·a list` to indicate this("quote a" or "alpha")
	
- For `e1::e2` to type-check, we need a `t` such that `e1` has type `t` and `e2` has type `t list`. Then the result type is `t list`

- null : `.a list -> bool`

Takes a alpha list(any type of list)(·a list) and returns a boolean value


- hd : `.a list -> .a`

- tl : `.a list -> .a list`


## 1 - 9:List Function

Gain experience with lists and recursion by writing several functions that process and/or produce lists...

example:

```
fun sum_list (xs:int list) = 
	if null xs then 0
	else hd xs + sum_list(tl xs)
	
	
fun countdown(x:int) = 
	if x=0 then []
	else x::countdown(x-1)
	
fun append(xs : int list, ys : int list) =
	if null xs then ys
    else (hd xs) :: append((tl xs),ys)

```
Functions over lists are usually recursive

- Only way to "get to all the elements"

- what should the answer be for the empty list?

- what should the answer be for a non-empty list?

	- Typically in terms of the answer for the tail of the list !
	
	
Similarly, functions that produce lists of potentially any size will be recursive

- You create a list out of smaller list

> 基本上函数型语言关于数组的处理都是递归运算。


## Let Expression

###Review

Huge progress already on the core pieces of ML:

- Types: `int bool unit t1...tn t list t1...tn->t`

	- Types are "nest"(each t above can be itself a compound type)
	
- Variables, environments, and basic expressions

- Functions

	- Build: `fun x0(x1:t1,....,xn:tn) = e`
	
	- Use: e0(e1,e2...en)
	
- Tuples

	- Build: `(e1,...,en)`
	 
	- Use: `#1 e, #2 e, ....`
	
- Lists

	- Build: `[]`, `e1::e2`
	
	- Use: `null e`, `hd e`, `tl e`


###Now...

引入局部变量

The big thing we need: local bindings

- For style and convenience

This segment:

- Basic let-expressions

Next segments:

- A big but natural idea: nested function bindings

- For efficiency

The construct to introduce local bindings is ***just an expressions****,
so we can use it anywhere an expression can go.


###Let - expressions

3 questions:

- Syntax: `let b1 b2...bn in e end`

	- Each `bi` is any *binding* and **e** is any expression
	
- Type-checking: Type-check each `bi` and `e` in a static environment that includes the previous bindings.Type of whole let-expression is the type of `e`.

- Evaluation: Evaluate each `bi` and `e` in a dynamic environment that includes the previous binding.Result of whole let-expression is result of evaluating `e`.

example:

```
fun silly1(z:int) = 

	let 
		val x = if z>0 then z else 34
		val y = x + z + 9
	in
		if x>y then x*2 else y*y
	end
	
```

```
func silly2() = 

	let 
		val x = 1
	in 
		//这里的x是在新的environment里面，上面的x会被shadow掉
		(let val x = 2 in x+1 end) + 
		
		//这里的x值为1
		(let val y=x+2 in y+1 end)
	end
	
```

###What's new

上面`let in end`语法实际上是对***scope***的描述：

- What's new is scope: where a binding is in the enviroment

	- In later bindings and body of the let-expression
	
		- (Unless a later or nested binding shadows it)
		
	- Only in later bindings and body of the let-expression

- Nothing else is new:

	- Can put any binding we want, event function bindings
	
	- type-check and evaluate just like at "top-level"


## 1-11：Nested Functions

###Any binding

According to our rules for let-expressions, we can define functions inside any let-expression

```
let b1 b2 ... bn in e end

```
This is a natural idea, and often good style

example:

```

fun countup_from1(x:int) = 
    let
	fun count (from: int) = 
	    if from = x
	    then x::[]
	    else from :: count(from+1)
    in 
	count(1)
    end
    
    
```

由于函数是first class的，可以在函数中定义函数，并且在scope中生效。

- Functions can use bindings in the environment where they are defined:

	- Bindings from "outer" environments
	
		- Such as parameters to the outer function
		
	- Earlier bindings in the let-expression
	
- Unnecessary parameters are usually bad style

	- Like to in previous example
	
###Nested functions: style

- Good style to define helper functions inside the functions they help if they are:
	
	- Unlikely to be useful elsewhere
	- Likely to be misued if available elsewhere
	- Likely to be changed or removed later
	
	
- A fundamental trade-off in code design:reusing code saves effort and avoids bugs, but makes the reused code harder to change later.

## 1-12 Let Expressions to Avoid Repeated Comupatation

example:

```
fun bad_max(xs:int list) = 
    if null xs then 0
    else if null (tl xs) then (hd xs)
    else if hd xs > bad_max(tl xs) then hd xs
    else bad_max(tl xs)
    
let x = bad_max [50,49,...,1]
let y = bad_max [1,2,...,59]  

```
Consider this code and the recursive calls it makes

- Don't worry about calls to `null`,`hd` and `tl` because they do a small constant amount of work

上面代码的效率：

- 对于第一种情况[50,49,...,1]执行bad_max的次数为50。

- 对于第二种情况[1,2,...,50]每一次执行bad_max又会递归执行两次bad_max，执行次数为2^50方


一种解法是缓存bad_max的结果：

```
fun good_max(xs: int list) = 
    if null xs then 0
    else if null (tl xs) then hd xs
    else
	let val tl_ans = good_max(tl xs)
	in
	    if hd xs > tl_ans then hd xs
	    else tl_ans
    end
```

上面的代码只调用good_max一次。

### Math never lies

The key is not to do repeated work that might do repeated work that do ...

- Saving recursive results in local bindings is essential.
         
  
##Options

Motived：

```
fun get_ max(xs: int list) = 
    if null xs then 0
    else if null (tl xs) then hd xs
    else
	let val tl_ans = good_max(tl xs)
	in
	    if hd xs > tl_ans then hd xs
	    else tl_ans
    end
```

### Motivating Options

Having `max` return 0 for the empty list is really awful

- Could raise an exception

- Could return a zero-element or one-element list

	- That works but is poor style because the built-in support for options expresses this situation directly  

###Options

类似Swift中的optional

- `t option` is a type for any type t

	- (much like `t list`，but a different type, not a list)
	
- Building:

	- `NONE` has type `.a option` (much like[] has type `.a list`)
	
	- `SOME e` has type `t option` if `e` has type `t`(much like `e::[]`)
	
- Accessing:

	- `isSome`  has type `.a option -> bool`
	
	- `valOf` has type `.a option -> .a ` (exception if given NONE)
 
改写后的max方法

```
fun max1(xs:int list) = 
    if null xs then NONE
    else
	let val tl_ans = max1(tl xs)
	in if isSome tl_ans andalso valof tl_ans > hd xs
	   then tl_ans
	   else SOME(hd xs)
    end

```

 


## 1-13 More Boolean and Comparison Expressions


SOme "odds and ends" that haven't come up much yet:

- Combining Boolean expressions(and,or,not)
- Comparison operations

###Boolean operations
 
`e1 andalso e2` => "&&"

`e1 orelse e2` => "||"

`not e1` => "!"

###Comparisions

`= <> > < >= <=`


## 1.14 A Key Benefit of Immutable Data

### A valuable non-feature: no mutation

Have now covered all the features you need.

Now learn a very important *non-feature*:

- Huh??How could the lock of a feature be important?
	
- When it lets you know things other code will not do with your code and the results your code produces

A major aspect and contribution of functional programming:

Not being able to assign to (a.k.a. mutate) variables or parts of tuples and lists

***This is a Big Deal***	

意思是SML或者函数型语言在设计的时对数据的操作就是immutable的,这种看似的缺陷反而成了它的优点。

###Cannot tell if you copy

比较下面两个方法:

```
fun sort_pair (pr: int * int) = 

	if #1 pr < #2 pr then pr
	else (#2 pr, #1 pr)

```


```
fun sort_pair (pr: int * int) = 

	if #1 pr < #2 pr 
	then (#1 pr, #2 pr)
	else (#2 pr, #1 pr)

```

In ML, there two implementations of `sort_pair` are *indistinguishable*

- But only beacause tuples are immutable

- The first is better style: simpler and avoids making a new pair in the then-branch

- In langauges with mutable compound data, these are different!

从实现上看，这两种方法是有区别的，前者是直接返回了入参对象，后者则是新创建了一个`pr`的`copy`出来:

```
val p = (3,4)

val x = sort_pair p

val y = x


```

对于第一种情况，y是p的alias，对于第二种情况y是p的copy

但是对于使用者来说，在ML中这两种方法是没有区别的，因为ML中数据结构是immutable的，也就是说p无法改变自己的值。

因此无论是引用还是copy, 都不会影响y。

那么，如果ML是mutable的会怎么样呢?

假如：

```

//假如p中的value可以被修改，变为(5，4)
 #1 p = 5 
 
 val z = #1 y

```

那么 z的值是多少呢？

对于第一种情况，y是p的alias，那么z的值为5

对于第二种情况，y是p的copy，那么z的值仍为3

这样你就必须不停的去关注y是p的copy还是alias。























































