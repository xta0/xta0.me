---
list_title: 编程语言原理(一) | Programming Language Part 1 | Functions
title: 编程语言原理(二)
layout: post
---


### Functional Programming 

"*Functional Programming*" can mea a few different things:

- Avoid mutation in most cases
- Using functions as values
- Style encouraging recursion and recursive data structures
- Style closer to mathematical definitions
- Programming idioms using laziness
- Anything not OOP or C?(not a good definiation)

Not sure a definiation of "functional language" exits beyond makes functional programming easy / default / required


### First-class Functions

- First-Class functions: Can use them wherever we use values
- Functions are values too
- Arguments,results parts of tuples, bound to variable, carried by datatype constructors exceptions...

```java
fun double x = 2*x
fun incr x = x+1
val a_tuple = (double,incr,double(incr 7))
val eighteen = (#1 a_tuple) 9
```

- Most common use is as an argument / result of another function
    - Other function is called a *higher-order-function*
	- Powerful way to factor out common functionality

### Function Closures

- Function closure: Functions can use bindings from outside the function defination(in scope where function is defined)
    - Makes first-class functions much more powerful
	- Will get to this feature in a bit, after simpler examples
	
- Distinction between terms first-class functions and function closure is not universally understood
	- Important conceptual distinction even if terms get muddled 

### Functions As Arguments

- We can pass one function as an argument to another function 
	- Not a new feature,just never thought to do it before

- Elegant strategy for factoring out common code

	- REplace N similar functions with calls to 1 function where you pass in N different(short) functions as arguments. 

- 数学描述 : f(g(x))


##  Polymorphic Types and Functions As Arguments

### The Key point

- HOF are often so "generic" and "reusable" that they have polymorphic types, i.e, types with type variables

- But there are HOF that are polymorphic

- ALways a good idea to understand the type of a a function, especially a HOF

### Types

函数`n_times`的类型为: `val n_times = fn : ('a -> 'a) * int * 'a -> 'a`，其中`'a`为泛型；fn的返回值类型要和n_times的返回值类型相同；我们可以把`n_times`严格的限制成某一类型:`(int -> int) * int * int -> int`，但是这种限制没有意义

> This *polymorphism* makes `n_times` more useful
	
### Anonymous Functions

以这个函数为例：

```
fun n_times(f,n,x) = 
    if n=0 then x
    else f(n_times(f,n-1,x))
    
fun triple x = 3*x

fun triple_n_times(n,x) = n_times(triple,n,x)

val x3 = triple_n_times(2,3)

```

接下来的问题是如何将`triple`函数变为匿名函数代入到`n_times`中

- 使用`let-in-end`:

```
fun triple_n_times(n,x) = n_times(let triple x = x*3 in triple end ,n,x)


```

- 使用匿名函数:

```
fun triple_n_times(n,x) = n_times(（fn x => 3*x),n,x)

```
匿名函数只提供一个函数原型

- Like all expression forms, can appear anywhere

- Syntax:

	- `fn` not `fun`
	- `=>` not `=`
	- no function name,just argument pattern
	


### Using anonymous functions

- Most common use : Argument to a HOF

	- Don't need a name just to pass a function
	
- But : cannot use an anonymous function for a recursive function

	- Beacause there is no name for making recursive calls
	
	- If not for recursion, `fun`bindings would be syntactic sugar for `val` bindings and anonymous functions
	
	```
	fun triple x = 3 * x
	
	val triple = fn y => 3*y
	
	```
	第一个triple实际上是第二个triple的语法糖。
	
	同样，`triple_n_times`也可以写成这样，但是并不是一种好的表述方式
	```
	( * poor style * )
	fun triple_n_times(n,x) = fn(n,x) => n_times(（fn x => 3*x),n,x)

	
	```
	

	
## 2-5 Unnecessary Function Wrapping


看这个例子：

```ml

fun n_times(f,n,x) = 
    if n=0 then x
    else f(n_times(f,n-1,x))
    
fun nth_tail(n,xs) = n_times((fn y => tl y), n , xs)

```

根据上一节学到的，我们可以把`n_times`中的`f`用匿名函数代替:`fn y => tl y`

但是这种情况下使用匿名 函数是不正确的，正确的是:

```ml

fun nth_tail(n,xs) = n_times(tl, n , xs)

```

由于`fn y => tl y`表述的意思和`tl`一样，因此直接使用`tl`更有效率。

抽象来说当一个函数满足:

```ml

fn x => f x

```

不需要使用匿名函数，直接使用`f`，在看一个例子

```ml

fun rev xs = List.rev xs

val rev = fn xs => List.rev xs

val rev = List.rev

```

## 2-6 Map and Filter

###Map

```ml

fun map(f,xs) = 
    case xs of 
	 [] => []
      | x:xs' => (f,x)::map(f,xs') 


```
- map的类型为：

```ml

val map = fn : ('a -> 'b) * 'a list -> 'b list

```

- 使用map:

```ml

val result = map(fn x => x+1 , [1,2,3])

```

- Map is without doubt 是最有名的 HOF:
	
 - The name is standard 
	
 - You use it all the time once you know it: saves a little space,但是更找你更要的是，保持运算的连续性
	
 - Similar predefined function：`List.map`
 
 	- But it uses currying 


###Filter

```ml

fun filter(f,xs) = 
    case xs of 
	[] => []
     | x::xs' => if f x
                 then x::(filter(f,xs'))
		 else filter(f,xs')

```

- filter类型为：

```ml

val filter = fn : ('a -> bool) * 'a list -> 'a list

```

- 使用filter

```ml

val filter_result = filter(fn x => x>1,[1,2,3])

```

- Filter也是很有名的HOF


## 2-7 Generalizing Prior Topics

Our examples of first-class functions so far have all:

- Take one function as an argument to another function

- Processed a number or a list

But first-class functions are usefule anywhere for any kind of data

- Can pass serveral functions as arguments

- Can put functions in data structures

- Can return funtions as results

- Can wirte HOF that traverse your own data structures

Useful whenever you want to abstract over "What to compute with"

- No new language features


### Return function as result

```ml

fun double_or_triple f = 

	if f 7 
	then fn x => 2*x
	else fn x => 3*x

```

`double_or_triple`的签名为：`(* (int -> bool) -> (int -> int) * )`

使用：

```ml

val double = double_or_triple(fn x => x-3 = 4)

```

但是REPL输出的`double_or_triple`的签名为:`(* (int -> bool) -> int -> int * )`

Because it never prints uncessary parentheses and 

`t1->t2->t3->t4` 等价于 `t1 -> (t2 -> (t3 -> t4))`

## 2-8 Definition of Lexical Scope

- We know function bodies can use any bindings in scope

- But now that functions can be passed around: In scope where?

- This semantics is called `lexical scope`

- There are lots of good reasons for this semantics

	- Discussed after explaining what the semantics is 
	
	- Later in course: implementing it 
	
- Must "get this" for competent programming

将的是变量的作用域:

```ml

val x = 1

(* f maps to a function that adds 1 to its argument *)
fun f y = x + y

(* x maps to 2 , shadows 1 *)
val x = 2 

val y = 3

(* call the function defined on line2 with 5 *)
val z = f (x+y)

(* z maps to 6 *)

```
The semantics of functions has two parts:

- The `code` part

- The `environment` that was current when the function was defined

- This is a "pair" but unlike ML pairs, you cannot access the pieces

- All you do is call this "pair"

- This pair is called `function closure`

- A call evaluates the code part in the environment part


### coming up:

- demonstrate how the rule works with HOF

- why the other natural rule, dynamic scope is a bad idea

- powerful `idioms` with HOF use this rule

	- Passing functions to iterators like `filter`
	
	- Several more idims




## 2-9 Why lexical scope

- Lexical Scope : use environment where function is defined

- Dynamic Scope : use environment where function is called


Decades ago, both might have been considered reasonable, but now we know lexical scope makes much more sense


- Here are three reasions:

 - Function meaning does not depend on variable names used
 
 - Function can be type-checked and reasoned about where defined
 
 - Closures can easily store the data they need 
 
 
 ### Does dynamic scope exist?
 
 - Lexical scope for variables is definitely the right default
 
 	- Very common across languages
 	
 - Dynamic scope is occasionally convenient in some situations
 
 	- Some languages(e.g. Racket)have special ways to do it
 	
 	- BUt most do not bother
 	
 - if you squint some, exception handling is more like dynamic scope:
 
 	- `raise e` transfers control to the current innermost handler
 	
 	- Does not have to by syntactically inside a handl expression



## 2-10 Closure and Recomputation

### When things 

Things we know:

- A function body is not evaluated until the function is called

- A function body is evaluated every time the function is called

- A variable binding evaluates its expression when the binding is evaluated, not every time the variable is used.

With closures, this means we can avoid repeating computations that do not depend on function arguments

- Not so worried about performance but good example to emphasize the semantics of functions

这一节的Dan想表达的意思是: closure可以capture value，这个value可以只计算一次，这样就避免重复计算。

## 2-11 Fold and More Closure

### Another famous function: Fold

`fold`(and synonyms/close relatives reduce, inject,etc.) is another very famous iterator over recursive structures

Accumulates an answer by repeatedly applying `f` to answer so far

- `fold(f,acc,[x1,x2,x3,x4])` computes `f(f(f(f(acc,x1),x2),x3),x4)`

```ml

fun fold (f,acc,xs) = 
    case xs of [] => acc
	    | x::xs =>fold(f,f(acc,x),xs) 

```

- This version "folds left"; another version "folds right"
- Whether the direction matters depends on `f`(often not)

`fold`的签名为:

```ml

val fold = fn : ('a * 'b -> 'a) * 'a * 'b list -> 'a

```

### Why iterators again?

- These "iterator-like" functions are not build into the language
	
	- Just a programming pattern
	
	- Though many languages have built-in support, which often allows stopping early without resorting to exceptions

- This pattern separates recursive traversal from data processing

	- Can reuse same traversal for different data processing
	
	- Can reuse same data processing for different data structures
	
	- In both cases, using common vocabulary concisely communicates intent
	
这里Dan想表达的意思是：如果我们能抽象一个迭代器出来，那么对于复杂的数据结构，一个人可以复杂实现迭代器，另一个人可以实现复杂的计算，这两者均可复用。

- 数组求和：

```ml

fun f_sum xs = fold ((fn(x,y) => x+y), 0, xs)

val sum = f_sum [1,2,3]

```

### Iterators made better

- Function like `map,filter` and `fold` are much more powerful thanks to closures and lexical scope

- Function passed in can use any "private" data in its environment

- Iterator "doesn't event know the data is there" or what type it has

 
 
## 2-12 : Another Closure Idiom: Combining Functions

### More idioms

- We know the rule for lexical scope and function closures

	- Now what is it good for
	
A partial but wide-ranging list:

- Pass functions with private data to iterators: Done

- Combine functions(e.g., composition)

- Currying(multi-arg functions and partial application)

- Callbacks (e.g., in reactive programming)

- Implementing an ADT with a record of functions

### Combing Functions

Canonical example is function composition:

```
fun compose(f,g) = fn x => f(g x)

```

- Creates a closure that "remembers" what `f` and `g` are bound to

- 它的签名为:

```
val compose = fn : ('a -> 'b) * ('c -> 'a) -> 'c -> 'b

```

- 在ML中，这种function compose有特殊的符号表示:

```
f o g 表示 f (g x)

```

- 例子：

```
fun sqirt_of_abs i = Math.sqart (Real.fromInt (abs i ))

```
也可以写成:

```
fun sqrt_of_abs i = ( Math.sqrt o Real.fromInt o abs ) i

```

### Left-to-right or right-to-left

```
val sqrt_of_abs  =  Math.sqrt o Real.fromInt o abs 

```

As in math, function composition is "right to left"

- "take absolute value, convert to real, and take square root"

- "square root of the conversion to real of absolute value"

"Pipelines" of functions are common in functional programming and many programmers prefer left-to-right

 
## 2-13:Another Closure Idiom:Currying

###Currying

- Recall every ML function takes exactly one argument

- Previously encoded `n` arguments via one `n-tuple`

- Another Way: Take one argument and return a function that takes another argument and ...

	- Called "currying" after famous logican Haskell Curry
	
	
	
- 之前处理多个入参的方式使用tuple：
	
```
fun sorted3_tupled (x,y,z) = z >= y andalso y >= x

val t1 = sorted3_tupled(7,9,11)

```
- 引入currying:

```
val sorted3 = fn x => fn y => fn z => z >=y andalso y >= x

fun sorted x = fn y => fn z => z >= y andalso y>=x 

```

```
val t2 = (((sorted3 7) 9) 11)

```

- Calling (sorted3 7) returns a closure with:
	
	- Code `fn y => fn z => z >= y andalso y>= x`
	- Environment maps `x` to `7`
	
- Calling that closure with 9 returns a closure with

	- Code `fn z => z>=y andalso y >= x`
	- Environment maps `x` to `7`, `y` to `9`
	
- Calling that closure with `11` returns `true`


### Syntactic sugar, part1

```
val t2 = (((sorted3 7) 9) 11)

```

- In general , e1, e2, e3, e4, ... means (((e1, e2), e3), e4,)

- So instead of `(((sorted3 7) 9) 11)` can just write `sorted3 7 9 11`

- Callers can just think "multi-argument function with spaces instead of tuple expression"

	- Different than tupling; caller and callee must use same technique


- Wrong:

```
val wrong1 = sorted3_tupled 7 9 11
val wrong2 = sorted3 (7,9,11)

```

### Syntactic sugar, part2

```
val sorted3_old = fn x => fn y => fn z => z >=y andalso y >= x

fun sorted3_nicer x y z = z >= y andalso y >= x

val t4 = sorted3_nicer 7 9 11

val t5 = (((sorted3_nicer 7) 9 ) 11)

```

## 2-14: Partial Application

### Too Few Arguments

- Previously used currying to simulate multiple arguments
 
- But if caller provides "too few" arguments, we get back a closure "waiting for the remaining arguments"

	- Called partial application
	
	- Convenient and useful
	
	- Can be done with any curried function
	
- No new semantics here: a pleasant idiom

- 例子:

```

val sum_partial = fold (fn (x,y) => x+y) 0 

```

上面这个例子：`fold`应该有三个参数: 一个算数表达式，一个初始值，一个数组，显然上面的`sum`缺一个参数，这也成了一个好处，相当于`sum`只接受一个数组参数:

```
val sum_partial = fn : int list -> int

```

使用`sum_partial`更方便:

```
val sum_value = sum_partial [1,2,3]

```

另一个例子:

```

fun exist f xs = 
	case xs of 
		[] => false
		| x::xs => f x orelse exist f xs

```

```
val no = exist (fn x => x=7) [4,11,23]

val hasZero = exist (fn x => x=0 )

```


## 2-15 : Currying Wrapup

### More combining functions

- What if you want to curry a tupled function or vice-versa

- What if a function's arguments are in the wrong order for the partial application you want?

Naturally, it is easy to write higher order wrapper functions

- And their types are neat logical formulas


将的时对于一些参数不满足要求的函数，如何使用curry使其满足条件:

- 例子：

```
fun range (i,j) = if i>j then [] else i :: range(i+1,j)

```

如果这么调用:

```

val countup = range 1 

```
肯定是不满足条件的，入参不是tuple，这种情况我们可以使用curry function：

先定义个curry 函数:

```
fun curry f = fn x => fn y => f (x,y)

```

根据语法糖，展开为:

```
fun curry f x y  = f (x,y)

```

这时将`countup`函数改写为:

```
val countup = curry range 1 

val xs = countup 7 (* [1,2,3,4,5,6,7] * )

```

同样我们可以定义`uncurry`

```
fun uncurry f (x,y) = f x y

```

### Efficiency

So which is faster: tupling or currying multiple-arguments?

- They are both constant-time operations, so it doesn't matter in most of your code

- For the small part where efficiency matters:

	- It turns out SML/NJ compiles tuples more efficiently
	
	- But many other FP implementation do better with currying(OCaml, F#, Haskell)
	
		- So currying is the "normal thing" and programmers read `t1->t2->t3->t4` as a 3-argument function that also allows partial applications.



## 2-16 : Mutation

### ML has(separate) mutation

- Mutable data sturcture are okay in some situations

	- When "update to the state of world" is appropriate moel
	
	- But want most language constructs truly immutable
	
- ML does this with a separate construct: references


### References

- New types: `t ref` where `t` is a type

- New expressions:

	- `ref e` to create a reference with initial contents `e` 指针
	
	- `e1 := e2` to update contents 赋值
	
	- `!e` to retrieve contents 取值


### Reference example

```
val x = ref 42
val y = ref 42
val z = x
val _ = x := 43
val w = (!y) + (!z) （* 85 * ）

```

- x binds to 一个指向42的指针，因此，x+1会报错，因为x的类型是 `int ref`即int型指针

- x 自身的值是不可以改变的


## 2-17 : Callbacks

### Callbacks

A common idiom: Library takes functions to apply later, when an event occurs - examples：
	
- When a key is pressed, mouse moves, data arrives...
	
- When the program enters some state
	
A library may accept multiple callbacks

- Different callbacks may need different private data with different types

- Fortunately, a function's type does not include the types of bindings in its environment

- (In OOP,objects and private fields are used similiaryly, e.g, Java Swing's event-listeners) 

### Mutable State

We really do want the "callbacks registered" to change when a function to register a callback is called

### Example call-back library

Library maintains mutable state for "What callbacks are there" and provides a function for accpting new ones

- A real library would all support removing them,etc.

- In example, callbacks have type `int -> unit` 

So the entire public library interface would be the function for registering new callbacks:

```
val onKeyEvent : (int -> unit) -> unit

```

unit 是无用的返回值

Because callbacks are executed for side-effect, they may also need mutable state


### Library implementation

关于callback的实现


## 2-18 : Standard-Library Doc

ML, like many languages, has a standard library

- For things you could not implement on your own
	
	- Examples: Opening a file, setting a timer
	
- For things so common, a standard definition is appropriate

	- Examples: `List.map`, string concatentation
	
- Where to look: http://www.standardml.org/Basis/manpages.html
	

## 2-19 : Implementing ADT using closure

略

## 2-20 : Closure idioms without Closures

### Higher order programming

- HOF e.g: with `map` and `filter` , is great

- Without closures, we can still do it more manually

	- In OOP (e.g., Java) with one-method interfaces
	
	- In procedural(e.g., C)with explicit environment arguments
	
- Working through this:

	- Shows connections between languages and features
	
	- Can help you understand closures and objects

如何在Java/C中实现closure

### Outline

- This segment:

	- Just the code we will "port" to Java or C

	- Not using statndard library 

- Next segments:

	- The code in Java and C
	
	- What works well and what is painful



## 2-21 : Closure in Java

### Java

- Java 8 scheduled to have closures(like C#, Scala, Ruby)

	- Write like `xs.map((x) => x.age).filter((x) => x>21).length()`
	
	- Make parallelism and collections much easier
	
	- Encourage less mutation
	
- But how could we program in an ML style without help

	- Will not look like the code above
	
	- Was even more painful before Java had generics 
	
### One-method interfaces

```
interface Func<B,A> { B m(A x); }

interface Pred<A> { boolean m(A x); }

```

- An interface is a named type

- An object with one method can serve as a closure

	- Different instances can have different fields (possibly different types) like different closures can have different environments(possibly different types)
	
- So an interface with one method can serve as a function type


## 2-22 : Closure in C


略

# Chap3

## 3-0 Course Motivation

- Why learn the fundamental concepts that appear in all languages?

- Why use languages quite different from C, C++, Java, Python?

- Why focus on functional programming?

- Why use ML, Racket, and Ruby in particular?

### Caveats

Will give some of my reasons in terms of this course

- My reasons: more personal opinion than normal lectures

	- Other may have equally valid reasons
	
- Partial list: surely other good reasons

- In term of course: Keep discussion informal

	- Not rigorous proof that all reasons are correct
	
- Will not say one language is "better" than other

### Summary

- No such thing as a "best" PL

- Fundamental concepts easier to teach in some PLs

- A good PL is a relevant, elegant interface for writing software 

	- There is no substitute for precise understanding of PL semantics
	
- Functional languages have been on the leading edge for decades

	- Ideas have been absorbed by the mainstream, but very slowly
	
	- First-class functions and avoiding mutation increasingly essential
	
	- Meanwhile, use the ideas to be a better C/Java/PHP hacker
	
- Many great alternatives to ML, Racket, and Ruby, but each was chosen for a reason and for how they complement each other. 


## 3-1 Why Study General PL Concepts


- Semantics:

	- Correct reasoning about programs, interfaces, and compilers requires a precise knowledge of semantics
	
		- Not "I feel...."
		- Not "I like curly braces more than parenthses"
		- Much of software development is designing precise interfaces;
		
- Idioms make you a better programmer

	- Best to see in multiple settings, including where they shine
	
	- See Java in a clearer light even if I never show you Java


## 3-2 Are All PLs the Same?


- Yes
	
	- Any input-output behavior implementable in language X is implementable in language Y [Church-Turing thesis]
	
	- Java, ML, and a language with one loop and three infinitely-large integers are the same
	
- Yes:

	- Same fundamentals reappear: variables, abstraction, one-of types, recursive definitions...
	
- No:

	- The human condition vs. different cultures(trave to learn more)
	
	- The default in one language is awkward in another 
	
	- Beware "the Turing tarpit"
	

## Why FP?

Why spend 60%-80% of course using FP:

- mutation is dicouraged

- HOF are very convenient

- One of types via constructs like datatypes

Because:

- These features are invaluable for correct, elegant, efficient

- Functional languages have always been ahead of their time

- Functional languages well-suited to where computing is going



### Ahead of their time

All these were dismissid as "beautiful, worthless, slow things PL professors make you learn"

- Garbage collection(Java didn't exist in 1995, PL courses did)

- Generics(`List<T>` in Java, `C#`), much more like SML than C++

- XML for universal data representation(like Racket/Scheme/LISP/...)

- HOF(Ruby,Javascript,C#...)

- Type inference(C#,Scala...)

- Recursion(a big fight in 1960 about this - I'm told)

- ...


### The future may resemble the past

Somehow nobody notices we are right ... 20 years later

- "To conquer" vs. "to assimilate"

- Societal progress takes time and muddles "take credit"

- Maybe pattern-matching, currying, hygienic macros, etc. will be next


### Recent Surge part1

- Clojure

- Erlang

- F#

- Haskell

- OCaml

- Scala

In general , see http://cufp.org

### Recent Surge part2

Popular adoption of concepts:

- C#, LINQ (closures, type inference,...)

- Java 8 (closures)

- MapReduce / Hadoop

	- Avoiding side-effects essential for fault-tolerance here
	
- ..

### Why a surge?

My best guesses:

- Concise, elegant, productive programming

- Javascript, Python, Ruby helped break the Java/C/C++ hegemony

- Avoiding mutation is the easiset way to make concurrent and parallel programming easier

	- In general, to handle sharing in complex systems
	
- Sure, functional programming is still a small niche, but there is so much software in the world today even niches have room


## Why ML,Racket,Ruby?

### The languages together

SML, Racket, and Ruby are a useful combination for us

|  | dynamically typed | statically typed |
| ------------ | ------------- | ------------ |
| functional | Racket  | SML |
| object-oriented | Ruby | Java/C#/Scala|

- ML: polymorphic types, pattern-matching, abstract types & modules

- Racket : dynamic typing, "good" macros, minimalist syntax, eval

- Ruby: classes but not types, very OOP, mixins

- ...


Really wish we had more time:

- Haskell: laziness, purity, type classes, monads

- Prolog: unification and backtracking

- ...

### But why not...

Instead of SML, could use similar languages easy to learn after:

- OCaml: yes indeed but would have to port all my materials

	- And a few small things...
	
- F#: yes and very cool, but needs a .Net platform

	- And a few more small things...
	
- Haskell: more popular, cooler types, but lazy semantics and type classes from day 1

Admittedly, SML and its implementations are showing their age, but it still makes for a shine foundation in statically typed , eager functional programming


Instead of Racket, could use similar languages easy to learn after:

- Scheme, List, Clojure,...

Racket has a combination of 

- A modern feel and active evolution

- "Better" macros, modules, structs, constracts,...

- A large user base and community(not just for education)

- An IDE tailored to education

Could easily define our own language in the Racket system

- Would rather use a good and vetted design

Instead of Ruby, could use another language:

- Python,Perl, Javascript are also dynamically typed, but are not as "fully" OOP, which is what I want to focus on 

	- Python also does not have closures
	
	- Javascript also does not have classes but is OOP
	
- Smalltalk serves my OOP needs

	- But implementation merge language / environment
	
	- Less modern syntax, user base, etc


###Is this real programming?

- The way we use ML/Racket/Ruby can make them seem almost "silly" precisely because lecture and homework focus on insteresting langaguge constructs


- "Real" programming needs IO, string operations, floating-point, graphics, project managers, testing framework, threads, build systems...

	- Many elegant languages have all that and more
	
		- Including Racket and Ruby
		
	- If we used Java the same way, Java would seem "silly" too




# Chap4

## 4-0: Remaining ML Topics

### Remaining Topics:

- Type Inference

- Mutual Recursion

- Module System

- Equivalence

## 4-1: Type Inference

### Type-checking

- (Static)type-checking can reject a program before it runs to prevent the possibility of some errors

	- A feature of statically typed languages
	
- Dynamically typed languages do little such checking 

	- So might try to treat a number as a function at run-time
	
- Will study relative advantages after some Racket

	- Racket, Ruby (Python, Javascript...) dynamically typed
	
- ML(and Java,C#,Scala,C,C++)is statically typed

	- Every binding has one type, determined "at complie-time"


### Impicitly typed

- ML is statically typed

- ML is implicitly typed: rarely need to write down types

```
fun f x =  (* infer val f : int -> int *)
	
	if x > 3
	then 42
	else x * 2

```

- Statically typed: Much more like Java than Javascript


### Type infernce

- Type inference problem: Give every binding/expression a type such that type-checking succeeds

	- Fail if and only if no solution exists
	
- In principle, could be a pass before the type-checker

	- But often implemented together
	
- Type inference can be easy, difficult or impossible

	- Easy: Accept all programs
	
	- Easy: Reject all programs
	
	- Subtle, elegant, and not magic: ML



## 4-1 ML Type Inference











#Chap 5: Summary of SML

