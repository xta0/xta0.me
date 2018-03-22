---
layout: post
title: Compiler

---

<em>所有文章均为作者原创，转载请注明出处</em>

##1-1: Intro

###编译器和解释器的区别:

- Compiler: 

program -> compiler -> exec + data -> output

- Interpreters：

program+data -> Interpreter -> output


###历史:

- 1954 IBM develops the 704

这个时候发现software取代hardware，成了性能的瓶颈。

- Speedcoding:

	- 1953, by Jon Backus
	
	- 解释型语言
	
	- 优势是提高了开发效率
	
	- 劣势是代码执行效率很低
	
		- 比直接手写代码慢10-20倍
		
		- 解释器占了300字节，相当于全部内存的30%
		

Speedcoding并没有流行起来，但John Backus认为自己的思路是有正确的，并将它用到了另一个项目中。由于当时的重要的application主要是scientific application。学术界通常都需要计算机能直接执行公式（Formulas）,John Backus认为Speedcoding最大的问题是使用了解释器去实时解析公式，于是他这次改变了思路：将公式（Formulas）先变成可执行的机器码（Translate），这样能加快程序的执行速度，同时又能保持开发的高效，于是便有了FORTRAN 1。

- FORTRAN 1

	- 1954 - 1957
	
	- 1958 : 50% 的代码都是通过FORTRAN写的
	
	- 世界上第一个编译器诞生了
	
	- FORTRAN成了世界上第一个比较成功的高级编程语言，对计算机科学产生了深远的影响

	- 现代的编译器保留了FORTRAN 1的一些框架：
	
		 - Lexical Analysis

		 - Parsing

		 - Sementic Analysis

		 - Optimization

		 - Code Generation

## 1-2: Structure of Compiler

###Lexical Analysis

LA也叫<strong>词法分析</strong>，LA的输入是程序字符串，输出是一系列的词法单元（token），例如：

```
if x == y then 
	z = 1;
else
	 z=2;
end

```

上面代码可以分解为以下的一些tokens:

- keyword: if, then ,else, end
- variable : x, y ,z
- constant : 1, 2
- operator : = , ; , ==
- seperator : space

每个token的格式如下：

```
<token-name，attribute-value>
```

其中，token的name代表这个token的类型，value用来记录该token在符号表中的index。例如语句：`pos = initial + rate*60`，如何切分token呢？

1. `pos`是一个token，用`<id,1>`，`id`表示这个token是一个符号，`1`表示它在符号表的第一个位置
2. `=`是一个token，用`<=>`表示，因为它不是符号，因此不计入符号表
3. `initial`是token，用`<id,2>`表示
4. `+`是一个token，用`<+>`表示
5. `rate`是token，用`<id,3>`表示
6. `*`是一个token，用`<*>`表示
7. `60`是一个token，用`<number,4>`表示，严格意义来说它不计入符号表

`pos = initial + rate*60`经过LA之后变成:`[ <id,1>, <=>, <id,2>, <+>, <id,3>, <*>, <60> ]`

###Parsing

Parsing也叫**语法分析**(syntax analysis)，Parsing的目的是将LA产生的一系列token形成语法树。

假如我们要解析一个英文句子，可以把它的结构用树形结构来描述，例如：

`"This line is a long sentence"`

其树形结构为：

![](2013-04-0-1.png)

我们分析代码语句成分也是类似的，例如下面语句：

```
if x==y then z = 1;
else z = 2;

```
类似的树形结构为:

![](2013-04-0-3.png)

具体来说，对于上一节LA中的例子来说，语法树结构如下：

![](2013-04-0-2.png)


### Semantic Analysis:理解语义

一旦树形结构确定，接下来就是最难的语义分析，编译器在这方面很难保持它的理解和programmer的理解是一致的，以英语为例：

"Jack said Jerry left his assignment at home." 

编译器很难理解his指的是jack还是jerrry。再来看一种极端情况: 

"Jack said Jack left his assignment at home?"

编译器不理解，到底有几个人，两个Jack是不是同一个人，his只谁？

这种问题对于编译器来说，是**variable binding**，编译器在处理这类变量名称模糊的情况，是由严格语法规定的：

```
{
	int jack = 3;
	{
		int jack = 4;
		cout<<jack; 
	}
}

```
上面的例子中两个jack都是相同的类型，因此编译器需要通过scope来判断输出哪一个jack。除了通过variable binding之外，还可以使用类型判断语义。比如:

"Jack left her homework at home."

Jack的类型为male，her显然类型为female。由此，编译器便可以知道句子中Jack和her不是同一个人，对应到程序中，便是有两个不同类型的变量。


###Optimization: 优化

优化通常是用来减少代码体积，例如："But a little bit like editing" 可被优化成："But akin to editing"，节省了代码容量，代码能运行的更快，消耗内存较少，编译器的优化是有针对性的，比如： 

```
x = y*0 
```
在满足一定条件时才会会被优化成:

```
x = 0
```
仅仅当，x，y是整数的时候，编译器才会这么优化。当x或y为浮点型时，x,y为NAN类型，而

```
NAN * 0 = NAN
```


###Code Generation：生成代码

- 生成汇编代码

- 转换成平台相关语言


###小结

- 基本上所有的compiler都会遵从上面几个步骤

- 但是从FORTRAN开始，上面5部分的比重却在发生变化：

	- 对于比较老的编译器来说，L,P所占的比重会很高，S,O会很低
	
	- 对于现代编译器来说，由于有了工具，L,P所在比重明显下降，而O所占的比重却大幅度上升



##1-3：Economy of Programming Language


- 为什么有很多种编程语言？

	- 科学计算，需要大量计算，需要对float point numbers支持，需要对array支持的很好和并行计算。并不是每种语言都能很好的支持上面的需求，FORTRAN在这方面做的很好。

	- 商用：需要持久化存储，report generation，数据分析。SQL在这方面做的很好

 	- 系统编程：low level control of resource，real time constrains。C/C++
 	
 	- 需求很多，很难设计一种语言满足所有场景。

- 为什么还有新的编程语言不断涌现？

传统语言改变的很慢，学习新语言很容易。如果新语言能更快的解决问题，那么它就有存在的价值


##2-1：Cool OVerview


###overview:

- Cool是用来学习编译器的语言:(Classroom Object Oriented Language)。

- Compiler很容易被实现

- 特性：抽象，静态类型，继承，内存管理等。

- 目标 : 生成MIPS汇编代码

###brief demo:

- 使用emacs: 

	- `%emacs 1.cl`

- 编译cool：

	- `%coolc 1.cl`
	
	- 生成 `1.s`的汇编代码
	
- 运行cool:

	- `%spim 1.s`



##3-1: Lexical Analysis

- 词法分析：将代码分解成token.

- Token Class（Token的类型）: 

	- Identifier:以英文字母开头, A1,Foo,B17,....

	- Integer:一串非空的数字, 0,12,001,...

	- keyword:if,else,begin,...

	- whitespace: 空格，换行，tab等
	
- Token格式：<class:string>
	
- LA的输入是一串字符串，LA输出的是一系列token的数组。类似的例子可以参考[LLVM](http://akadealloc.github.io/blog/2013/03/20/LLVM-1.html)

	- 例如输入是: foo=42;

	- 输出是:`<identifier,"foo">,	<op,"=">, <int,"42">`
	
- 总结一下LA要做两件事：
	
	- 检查substring中包含的token
	
	- 每个token生成<class:string>的格式 



##3-2: Lexical Analysis example

- FORTRAN : 空格不记入token

	- 例1: `VA R1` = `VAR1`

	- 例2: 在FORTRAN中，循环：`DO 5 I = 1,25` 意思是在DO符号开始处和符号5(类似goto的label)之间的语句是循环体,变量从I的值从1到25
	
	- 例3: 代码`DO 5 I = 1.25`的意思是变量`DO51`的值为1.25
	
	- 问题: 对于例2，例3来说，LA是如何判断等号左边是变量赋值语句还是循环语句？

	- LA检测token的顺序是从左到右，这时它涉及到 look ahead,当LA发现DO时，它会look ahead去看等号后面是逗号还是点。类似的情况还有`if,else`等这种关键字，`=,==`这种操作符，当LA检测到首字母时并不能确定它是变量名称还是关键字或操作符，因此LA需要look ahead。
	
	- 因此在设计LA的时候要尽量避免这种look ahead，会影响性能。

	- FORTRAN有这种funny rule，是因为那个时候很容易不小心打出空格

- C++: 

	- C++的模版语法：`Foo<Bar>` 

	- C++的IO输入语法：`cin>>var`

	- 如果碰到:`Foo<Bar<Bazz>>`怎么办？, 最后的`>>`怎么判断呢？

	- C++的LA解决办法是，让developer手动加一个空格：``Foo<Bar<Bazz> >`

##3-3: Regular Languages

- Lexical structure = token classes

- We must say what set of strings is in a token class:当拿到了n多个token之后，我们首先要匹配出token class,然后才能拿到每个token class对应的string

	- 寻找token class的方法一般是使用<em>regular langauages</em>


- 什么是Regular Language:

	- 定义Regular Language一般需要使用正则表达式(Regular Expression):

		- 每一种正则表达式代表一种字符集，有两种基本的字符集：
		
			- Single character: 
			
				- `'c' = {"c"}`
				
				- 表示只含有这个字符的字符集
		
			- Epsilon: 
			
				- `ε = {""}`
				
				- 表示只包含一个空字符串的字符集
				
				- 注意`ε`不等于`o(empty)`

		- 除了这两种基本的字符集外，还有三种组合型的字符集:
		
			- Union
			
				- `A+B = {a|a<A} or {b|b<B}`
				
			- Concatenation
			
				- `AB = {ab|a<A and b<B}`
				
			- Iteration
			
				- `A*` = `AAA....A`(重复i次)
				
				- 如果`i=0`，那么上面式子变成`A0`，`A0 = e(Epsilon) = {""}` 
				
		- 正则表达式：在某个字符集(假如为Z)上，有一个最小的表达式集合，通过这个集合可以表征Z上任意字符的组合（可能的出现情况）。这个最小的表达式集合包括:
		
			- R = `e` = `{""}`(epsilon)
			
			- R = `c` c<Z
			
			- R = R+R
			
			- R = RR
			
			- R = R*
			
		- 正则表达式的例子:
		
			- `Z = {0,1}`
			
				- `1* ` = `""` + `1` + `11` + `111` + `111...1`(i次) = all strings of 1's，意思是`1*`这个正则表达式可以表示所有1的字符。
			
				- `(1+0)1`  = `{ab|a<1+0 and b<1} ` = `{11,01}`，意思是`(1+0)1`这个正则表达式可以表示`{11,01}`这两种情况的字符串
			
				- `(0+1)*` = `"" + (0+1) + (0+1)(0+1) + ... + (0+1)...(0+1) ` = all string of 1's and 0's。意思是这个正则表达式可以表示字符集Z的任意字符组合。

				- 表征同一字符集的正则表达式有不止一个，例如第1个例子`1*`也可以写作`1* +1 `第2个例子也可以写作`11+10`。


##3-4: Formal Languages

- 定义: 假设Z是一个字符集，有一种语言用来描述基于Z的，符合某些条件的，字符串集合。

- 例子:

	- 字符集 Z 为 English characters
	
	- 语言为English sentences(sentence是character的集合)

- 例子：
	
	- 字符集Z为 ASCII
	
	- 语言为:C语言
	
- 我们不能脱离字符集Z来讨论Formal Language

- 每种Formal Language都有Meaning function ：`L` maps syntax to semantics(将表达式转换为语义)

	- L(expression) = M
	
		- expression是某种syntax，比如正则表达式
		
		- M是一个字符串集合
		
	- 所有正则表达式都是expression,所有正则表达式描述的字符集都是M，因此需要有个Meaning Function来建立从express到字符集M的关系: 
		
		- 公式：`L:Exp -> Set of Strings`,例如：
	
		- `L(e) = {""}`
		
		- `L('c') = {"c"}`
		
		- `L(A+B)` = `L(A) or L(B)`
		
		- `L(AB)` = `{ab| a<L(A) and b<L(B)}`
		
		- `L(A*)` = `{AA....AAAA}`
		
	- 为什么要定义meaning function？
	
		- Makes clear what is syntax, what is semantics
		
		- Allows us to consider notation as a separate issue: 语法（syntax）和语义（semantics）不是一回事
		
		- Because expressions and meanings are not 1-1:描述同一字符集的正则表达式不唯一
		
		 
	- Meaning is many to one
	
		- syntax和semantics是多对1的关系，描述相同的semantics可以通过不同的syntax
		
		- never one to many
		
		
##Lexical Specifications


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



