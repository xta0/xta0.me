---
list_title: 写一个解释器 | Build a Parser By Hand | 语法分析 | Parsing
title: 语法分析
layout: post
mathjax: true
categories: [Parser,Compiler]
---

### Syntatical Analysis

上一篇文章中我们使用正则表达式可以将句子(代码)变成一个token list，但是我们如何能确定这个token list是否符合有效的语法规则呢？这就是Syntatical Analysis (Parsing) 需要解决的问题。

Noam Chomsky在1955年提出了一个Syntatic Structure的概念，即utterances have rules(Formal Grammars)。Formal Grmmar将token分成non-terminals和terminals。比如下面例子

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
可见上述的替换过程是一个recursive的过程，我们指定的替换规则则叫做Recursive Grammar Rule。可见通过有限的Grammar rule以及递归关系的存在，我们可以产生无限的utterances(tokens)。

我们再来看一个实际的例子，假设我们有下面的Grammar Rule

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
因此`y`是valid的表达式，而`z+1`不满足任意一条Grammar rule，因此`z`不是一个valid的表达式。

另外，Grammar Rules比正则表达式更强大。比如解析数字，正则式为`[0-9]+`，如果用Grammar rule来表示，则为

```shell
number -> 'digit' more_digits
more_digits -> 'digit' more_digits
more_digits -> ϵ
```

如果用Grammar rule来解析数字42，则推导过程为

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

我们可以将任何一组Grammar Rules变成一个Parse Tree(或者AST)，例如还是前面的Grammar Rule

```shell
exp -> exp + exp
exp -> exp - exp
exp -> num
```
假设我们的表达式为`1+2-3`，对应的token list为`[1, + , 2, -, 3]`。此时，符合该token list的一个Parser Tree可以表示为

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
所有的叶子节点组成了我们的token list。但是同样的token顺序，我们根据上面的语法规则还可以生成另外一个合法的Parse Tree


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
此时，表达式计算的是`1+(2-3)`，显然，这和我们希望的结果不符。这说明我们的语法规则式具有**Ambiguity**，即一个表达式会产生不止一个Parse Tree。实际上我们的Parser并不知道四则运算的从左到右的运算规则。一种解决办法是可以给我们的语法规则加一个括号的rule

```shell
exp -> (exp)
```

### HTML Grammars

正如前面小节提到的，使用这则表达式不能帮助我们解析HTML文本，因此我们可以为HTML定义一个语法规则
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
                    'welcome' ele       html
                               |         /   \
                              'to'     ele     ϵ
                                   /    |    \
                                 to    html   tc 
                                 |     / \     |
                               '<b>'  ele html'</b>'     
                                      |    |
                                    'xta0' ϵ
```
这种Tree structure看起来复杂，但对于计算机来说确实很有效的数据结构，我们后面会提到如何用代码生成Parse Tree。

### Javascript Grammar

接下来我们来顶一个JavaScript的Grammar，我们先focus在一个简单函数

```javascript
function up_to_ten(x){
        if(x < 10){
           return x;
        } else {
           return 10;
        }
}
```
如果要支持parse上面的语法，我们先来定义如下的Grammar Rules

```shell
exp -> 'identifier'
exp -> 'number'
exp -> 'string'
exp -> 'TRUE'
exp -> 'FALSE'
exp -> exp '+' exp
exp -> exp '-' exp
exp -> exp '*' exp
exp -> exp '/' exp
exp -> exp '==' exp
exp -> exp '<' exp
exp -> exp '&&' exp

stmt -> 'identifier' = exp
stmt -> 'return' exp
stmt -> 'if' exp compound_stmt
stmt -> 'if' exp compound_stmt 'else' compound_stmt
compound_stmt -> '{' stmts '}'
stmts -> stmt ';' stmts
stmts -> ϵ
```
有了这些规则，我们可以顺利的解析一些expression，但是对于像JavaScript或者Python这类高级语言，除了expression之外，还有function，因此我们需要顶一个关于解析函数的Grammar Rule

```shell
js -> element js
js -> ϵ

# function definination
element -> 'function identifier(' opt_params ')' compound_stmt
element -> stmt;
opt_params -> params
opt_params -> ϵ
params -> 'identifier', params
params -> 'identifier'

# function call
exp -> ...
exp -> 'identifier(' opt_args) ')'
opt_args -> args
opt_args -> ϵ
args -> exp ',' args
args -> exp
```

### Grammar in Python

上面提到的这些Grammar Rule该如何用代码表示呢，我们以Python为例，还是上面提到的语法规则

```shell
exp -> exp + exp
exp -> exp - exp
exp -> (exp)
exp -> num
```
我们可以用下面规则来实现语法规则

```shell
A -> B C
#python tupe
("A", ["B", "C"])
```
则对应的Python代码为

```python
grammar = [
  ("exp", ["exp", "+", "exp"]),
  ("exp", ["exp", "-", "exp"]),
  ("exp", ["(", "exp", ")"]),
  ("exp", ["num"]),
]
```

现在假设我们有一个token list - `['print', exp, ;]`，我们需要将`exp`替换为其中一条Grammar Rule，假设我们用第一条 `exp -> exp + exp`，则替换后的结果为

```python
['print', 'exp', '+', 'exp', ';']
```
我们可以将`exp`不断的进行递归替换，直到数组中每个元素都是叶子节点为止

### Earley Parser (Shift-Reduce Parser)

回到最开始的问题，给定一组token和一系列Grammar Rules，我们如何知道这组token是否符合语法规则。比如，一组token如下

```shell
['(', ')',')']
```
而我们的语法是要求括号具有完整的匹配

```shell
exp -> (exp)
exp -> ϵ
```
为了知道上述token是否满足这个语法，一个简单的做法是Brute Force，即枚举所语法规则所产生的的所有可能性，看是否匹配输入和token。例如

```shell
()
(())
((()))
...
```
前文规则可知，由于递归的存在，context-free grammar所产生的出的规则是无限的，因此这种方式显然是不正确的。

接下来，我们来介绍一种基于`chart`的parsing technique - Earley Parser

我们需要定义一个`chart`字典，其中的元素是一个数组用来保存条parsing state，数组中不允许有重复的state，因此它是一个ordered set。

```python

# chart[index] returns a list that contains state exactly
# once. The chart is a Python dictionary and index is a number. addtochart
# should return True if something was actually added, False otherwise. You may
# assume that chart[index] is a list.

def addtochart(chart, index, state):
    if state not in chart[index]:
        chart[index] = [state] + chart[index]
        return True
    else:
        return False
```

我们python的tuple来代码来表示一条state

```python
# x -> ab . cd from j

state = ("x", ["a", "b"], ["c", "d"], j)
```

### Closure

我们可以对token list进行从左向右遍历，每遇到一个token，我们来检查它是否是non-terminal，如果是，则进行语法规则替换，这个过程称为computing the closure或者叫predicting。我们来看一个例子

假设当前的Grammar如下，它包含下面几条rewrite rules

```shell
E -> E - E
E -> E + E
E -> (E)
E -> 'num'
T -> 'I like t'
T -> ϵ
```
假设输入的token为 `['a','b','c','d']`，当我们paser到 `ab`时，下一个token为`c`，此时我们需要看`c`是否满足某条规则，如果是，则将其按照rewrite rule进行替换

```python

# We are currently looking at chart[i] and we see x => ab . cd from j

# Write the Python procedure, closure, that takes five parameters:

#   grammar: the grammar using the previously described structure
#   i: a number representing the chart state that we are currently looking at
#   x: a single nonterminal
#   ab and cd: lists of many things

# The closure function should return all the new parsing states that we want to
# add to chart position i
grammar = [ 
    ("exp", ["exp", "+", "exp"]),
    ("exp", ["exp", "-", "exp"]),
    ("exp", ["(", "exp", ")"]),
    ("exp", ["num"]),
    ("t",["I","like","t"]),
    ("t",[""])
    ]

def closure (grammar, i, x, ab, cd):
    next_states = [
        (rule[0], [], rule[1], i)
        for rule in grammar 
        if len(cd) > 0 and rule[0] == cd[0]
    ]

next_states = closure(grammar, i, x, ab, cd)
for next_state in next_states:
    any_changes = addtochart(chart, i, next_state)
```

### Shift 

所谓shift是指当前token如果是一个terminate token，则skip该token，继续向右parse

```python
# Writing Shift

# We are currently looking at chart[i] and we see x => ab . cd from j. The input is tokens.

# The procedure, shift, should either return None, at which point there is
# nothing to do or will return a single new parsing state that presumably
# involved shifting over the c if c matches the ith token.

def shift (tokens, i, x, ab, cd, j):
    if len(cd) == 0 and tokens[i] == cd[0]:
        return (x, ab + [cd[0]], cd[1:], j)
    else:
        return None

next_state = shift(tokens, i, x, ab, cd, j)
if len(next_state) > 0:
    any_changes = addtochart(chart, i+1, next_state)
```

### Reduction

Reduction是将已经存在的

```python

# Writing Reductions

# We are looking at chart[i] and we see x => ab . cd from j.
# you only want to do reductions if cd == []

def reductions(chart, i, x, ab, cd, j):

next_states = reductions(chart, i, x, ab, cd, j)
for next_state in next_states:
    any_changes = addtochart(chart, i, next_state)
```




## Resources

- [Earley parser](https://en.wikipedia.org/wiki/Earley_parser)
- [Programming Language](https://classroom.udacity.com/courses/)