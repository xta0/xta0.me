---
list_title: 写一个解释器 | Build a Parser By Hand | 语法分析 | Parsing
title: 语法分析
layout: post
mathjax: true
categories: [Parser,Compiler]
---

### Syntatical Analysis

上一篇文章中我们使用正则表达式可以将句子(代码)变成一个token list，但是我们如何能确定这个token list是否符合有效的语法规则呢？这就是Syntatical Analysis (Parsing) 需要解决的问题。

Noam Chomsky在1955年提出了一个Syntatic Structure的概念，即utterances have rules(formal grammars)。Formal Grmmar将token分成non-terminals和terminals。比如下面例子

```shell
Sentence -> Subject Verb
Subject -> 'Teachers'
Subject -> 'Students'
Verb -> 'Write'
Verb -> 'Think'
```
其中带引号的单词为terminals，则我们可以将任意句子按照上面规则进行替换，直到句子中全部都是terminals，我们看一个简单的例子

```shell
Sentence -> Subject Verb -> 'Students' Verb -> 'Students' 'Write'
Sentence -> Subject Verb -> Subject 'Think' -> 'Teachers' 'Think'
```

我们称上面替换的结果为Derivation。如果我们再加一条规则

```shell
Subject -> Subject 'and' Subject
```
则上面的例子中产生出一个新的Derivation

```shell
Sentence -> Subject Verb -> Subject 'and' Subject Verb -> 
'Students' 'and' Subject Verb -> 'Students' 'and' Subject 'Think' ->
'Students' 'and' 'Teachers' 'Think'
```
可见上述的替换过程是一个recursive的过程，我们指定的替换规则则叫做recursive grammar rule。可见通过有限的grammar rule以及递归关系的存在，我们可以产生无限的utterances(tokens)。

我们再来看一个实际的例子，假设我们有下面的grammar rule

```shell
stmt -> 'identifier' = exp
exp -> exp '+' exp
exp -> exp '-' exp
exp -> 'number'
```
则我们可以用这个rule来判断下面的表达式(statement)是否valid

```shell
x = 1
y = 2500 - 1
z = z+1
```
我们来看一下`y`和`z`的推导过程

```shell
y = exp -> exp '-' exp -> 2500 '-'
```
因此`y`是valid的表达式，而`z+1`不满足任意一条grammar rule，因此`z`不是一个valid的表达式。

另外，Grammar Rules比正则表达式更强大。比如解析数字，正则式为`[0-9]+`，如果用Grammar rule来表示，则为

```shell
number -> 'digit' more_digits
more_digits -> 'digit' more_digits
more_digits -> ϵ
```

如果用grammar rule来解析数字42，则推导过程为

```shell
number -> 'digit' more_digits -> 'digit' digit more_digits
-> 'digit' 'digit' ϵ -> 'digit' 2 -> 42
```

### Context-Free Grammar

实际上任何的正则表达式都可以用一种”语法“代替，我们看一个复杂一点的例子，假设有正则式r'p+i?'`，我们可用下面语法代替

```shell
regex -> pplus iopt
pplus -> 'p' pplus
pplus -> 'p'
iopt -> 'i'
iopt -> ϵ
```
我们称能用正则表达式解析的语言成为Regular Languages，而能用上述Grammar Rule解析的语言称为Context-Free Languages。而所谓的Context-Free指的是不论上下文的context如何，都不影响替换规则。

几种常见的正则表达式对应的Grammar如下

```shell
r'ab' = g -> 'ab'

r'a*' = g ->  ϵ
        g -> 'a' g

r'a|b' = g -> 'a'
         g -> 'b'
```

实际上，有些场景我们是不能使用regular expression做parser的，比如前文提到的解析HTML文本。这是由于正则式无法检测到括号mismatch的情况，比如

```shell
<p>abc<b>def</b>gh</p>
<p>abc<b>def</p>gh</b>
```
显然HTML的parser必须要能检测出括号的闭合，因此对于第二种情况是需要报错的，但是正则式无法做到这一点。

### Parser Tree

我们可以将任何一组Grammar Rules变成一个Parse Tree，例如还是前面的Grammar Rule

```shell
exp -> exp + exp
exp -> exp - exp
exp -> num
```
表达式为`1+2-3`，此时Parser Tree可以表示为

```shell
          exp
        /  |  \ 
      exp  -   exp
     / | \       \
  exp  +  exp    num
   /       |      |
 num      num     3
  |        |
  1        2
```
所有的叶子节点为terminal node，非叶子节点为non-terminal node。

实际上，上述的Parser Tree还有另外一种形式

```shell
          exp
        /  |  \ 
      exp  +   exp
     /        / | \
   num     exp  -  exp
    |       |       |
    1      num     num
            |       |
            2       3
```
此时，表达式计算的是`1+2-3`，和我们希望的不符，这说明我们的表达式具有**Ambiguity**，即一个表达式会产生不止一个Parse Tree。实际上我们的Parser并不知道四则运算的从左到右的运算规则，此时我们可以给Grammar加一个括号的rule

```shell
exp -> (exp)
```

### HTML Grammars

正如前面小节提到的，使用这则表达式不能帮助我们解析HTML文本，因此我们需要定义一个grammar rule

```shell
html -> element html
html -> ϵ
element -> 'word'
element -> tag_open html tag-close
tag-open -> '<word>'
tag-close -> '</word>'
```
假设有一段HTML文本为 `<p>welcome to <b>xta0</b> site</p>`，生成的Parse Tree为

```shell
                                 html
                               /       \ 
                             ele       html
                       /      |     \    |
                      to    html     tc  ϵ
                    /    /       \    \    
                '<p>'  ele       html '</p>'     
                        |      /      \   
                    'welcome' ele     html
                               |      /   \
                              'to'   ele   ϵ
                                /     |     \
                               to    html    tc 
                                |     / \     |
                              '<b>'  ele html '</b>'     
                                      |    |
                                    'xta0' ϵ
```
这种recursive的结构看起来复杂，实际上对于计算机是很简单的算法，我们后面会提到如何用代码生成Parse Tree

### Javascript Grammar



## Resources

- [Programming Language](https://classroom.udacity.com/courses/)