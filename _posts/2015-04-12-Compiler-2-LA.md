---
layout: post
list_title: 编译原理 | Compiler Theory | Lexical Analysis & Finate Automata
title: Lexical Analysis & Finate Automata
mathjax: true
categories: [Compiler]
---

## Lexical Analysis

- 词法分析
    - 将代码分解成token.
- Token Class（Token的类型）: 
	- Identifier:以英文字母开头, `A1,Foo,B17`,....
	- Integer:一串非空的数字, `0,12,001`,...
	- keyword:`if,else,begin`,...
	- whitespace: 空格，换行，tab等
	
- Token格式：
    - `<class:string>`

- LA的输入是一串字符串，LA输出的是一系列token的数组
	- 例如输入是: foo=42;
	- 输出是:`<identifier,"foo">,	<op,"=">, <int,"42">`
    - 类似的例子可以参考[LLVM](http://akadealloc.github.io/blog/2013/03/20/LLVM-1.html)

- 总结一下LA要做两件事：
	- 检查substring中包含的token
	- 每个token生成`<class:string>`的格式 


###  Lexical Analysis example

- FORTRAN :
    - 空格不记入token
	    -  `VA R1` = `VAR1`
        - `DO 5 I = 1.25`的意思是变量`DO51`的值为1.25
	- 在FORTRAN中，循环：`DO 5 I = 1,25` 意思是在DO符号开始处和符号5(类似goto的label)之间的语句是循环体,变量从I的值从1到25 
	- 问题: LA是如何判断等号左边是变量赋值语句还是循环语句？
	    - LA检测token的顺序是从左到右，这时它涉及到 look ahead,当LA发现DO时，它会look ahead去看等号后面是逗号还是点。类似的情况还有`if,else`等这种关键字，`=,==`这种操作符，当LA检测到首字母时并不能确定它是变量名称还是关键字或操作符，因此LA需要look ahead。
	    - 因此在设计LA的时候要尽量避免这种look ahead，会影响性能。
	- FORTRAN有这种funny rule，是因为那个时候很容易不小心打出空格

- C++: 
	- C++的模版语法：`Foo<Bar>` 
	- C++的IO输入语法：`cin>>var`
	- 如果碰到:`Foo<Bar<Bazz>>`怎么办？, 最后的`>>`怎么判断呢？
	- C++的LA解决办法是，让developer手动加一个空格：`Foo<Bar<Bazz> >`(现在的编译器已经修复了该问题)

## Regular Languages

- Lexical structure = token classes
- We must say what set of strings is in a token class:当拿到了n多个token之后，我们首先要匹配出token class,然后才能拿到每个token class对应的string
	- 寻找token class的方法一般是使用<em>regular langauages</em>

- 什么是Regular Language:
	- 定义Regular Language一般需要使用正则表达式(Regular Expression):
		- 每一种正则表达式代表一种字符集，有两种基本的字符集：
			- Single character: 
				- `'c' = {"c"}`:表示只含有这个字符的字符集
			- Epsilon: 
				- `ε = {""}`: 表示只包含一个空字符串的字符集
				- 注意`ε`不等于`o(empty)`

		- 除了这两种基本的字符集外，还有三种组合型的字符集:
			- Union
				- `A+B = {a|a<A} or {b|b<B}`
			- Concatenation
				- `AB = {ab|a<A and b<B}`
			- Iteration
				- `A*` = `AAA....A`(重复i次)
				- 如果`i=0`，那么上面式子变成`A0`，`A0 = e(Epsilon) = {""}` 
				
		- 正则表达式：在某个字符集(假如为Z)上，有一个最小的表达式集合，通过这个集合可以表征Z上任意字符的组合（可能的出现情况）。正则表达式的y语法（grammar）如下
			- Baes cases
				- R = `ε` = `{""}`(epsilon)
				- R = {`c`}, c<Z
			- Three compound expressions
				- Union:  `R = R+R`
				- Concatenation: `R = RR`
				- Iteration: `R = R*`
			
		- 正则表达式的例子:
			- `Z = {0,1}`
				- `1* ` = `""` + `1` + `11` + `111` + `111...1`(i次) = all strings of 1's，意思是`1*`这个正则表达式可以表示所有1的字符。
				- `(1+0)1`  = `{ab|a<1+0 and b<1} ` = `{11,01}`，意思是`(1+0)1`这个正则表达式可以表示`{11,01}`这两种情况的字符串
				- `(0+1)*` = `"" + (0+1) + (0+1)(0+1) + ... + (0+1)...(0+1) ` = all string of 1's and 0's。意思是这个正则表达式可以表示字符集Z的任意字符组合。
				- 表征同一字符集的正则表达式有不止一个，例如第1个例子`1*`也可以写作`1* +1 `第2个例子也可以写作`11+10`。
- In Summary 小结
	- Regular expressions specify regular languages 正则表达式是定义正则语言的基础
	- 正则表达式的基础语法有5条
		- Two base cases
			- empty and 1-character strings
		- Three compound expressions
			- union, concatenation, iteration 


## Formal Languages

- Formal language是计算机编程语言理论的基础
- 定义
	- 假设 $\sum$ 是一个字符集，Formal Languages是指建立在这个字符集之上的语言
		- Alphabet = English characters, Language = English sentences
		- Alphabet = ASCII, Language = C programs
	- 不同的语言是建立在不同的字符集上，因此讨论某种Formal Language的前提是先确定其所在的字符集合

### Meaning Function

- 每种Formal Language都有Meaning function
	- `L` maps syntax to semantics(将表达式转换为语义)

		$$
		L(e) \thinspace = \thinspace M 
		$$

		例如，对于正则语言来说，`e`为正则表达式，
		
	- 对于正则语言来说，`e`为正则表达式，`M`为其所匹配的字符串集合，
		- `L(e) = {""}`
		- `L('c') = {"c"}`
		- `L(A+B)` = `L(A) or L(B)`
		- `L(AB)` = `{ab| a<L(A) and b<L(B)}`
		- `L(A*)` = `{"",A,AA,...,A...A}`
		
- 为什么要定义meaning function？
	- Makes clear what is syntax, what is semantics
	- Allows us to consider notation as a separate issue ,分离语法(`e`)和语义(`L(e)`)
	- Because expressions and meanings are not 1-1, 描述同一字符集的正则表达式不唯一
		- 例如`0*`表示`{"",0,00,...}`该集合也可以用`0+0*`来描述
	- 对应到编程语言
		- 语法糖可以有很多，但是处理后得到的语义是相同的
		- 不同变成语言定义变量的方式不同，但其语义均为在内存中定义一个变量
			
## Lexical Specifications

- 用

- LA的具体表现:
	- keyword: "if" or "else" or "then" or ...
		- 正则： `'i''f' + 'e''l''s''e'`
		- 简化： `'if' + 'else' + 'then' + ...`
	
	- Integer: a non-empty string of digits
		- 单个integer的正则：`digit = '0'+'1'+'2'+'3'+...+'9'`
		- 至少有一个非空integer的正则：`AA* => digit digit*`
			- 简化上面，得到总的正则为:`A* => digit*`
	
	- Identifier: strings of letters or digits, starting with a letter
		- letter的正则 : `'a'+'b'+'c'+...+'z'+'A'+'B'+...+'Z'`
			- 简化上面的正则，使用`range:[]`符号:`[a-z A-Z]`
			- 总的正则`letter(letter + digit)* ` 

	- whitespace: a non-empty sequence of blanks, newlinke, and tabs
		- blanks : `' ' `
		- new line: `\n`
		- tabs: `\t`
		- 总的正则: `(‘ ’+‘\n’+'\t')+`

	- anyone@cs.standford.edu	
		- 正则为`letter+‘@’+letter+'.'+letter+'.'+letter`
		
	- PASCAL
		- digit = `'0'+'1'+...+'9'`
		- digits = `digit+`
		- opt_fraction = `('.'digit) + e`
		- opt_exponent = `('E'('+'+'-'+e) digits) + e`
		- num = `digits opt_fraction opt_exponent`


