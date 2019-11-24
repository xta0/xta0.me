---
list_title: 写一个解释器 | Build a Parser By Hand | 语法分析 | Parsing
title: 语法分析
layout: post
mathjax: true
categories: [Parser,Compiler]
---

前面我们使用正则表达式可以将一行代码切分成一个个token，但是我们如何验证这一行代码是否是正确的呢？例如，上面的token可组成下面两种形式的代码：

```html
<b>This is <i>my</i> webpage!</b>
<b>This is <i> my</b> webpage!</i>
```

显然第二种是非法的。这个问题和表达式括号匹配问题类似，数学上可以证明如果仅使用正则表达式是无法实现括号匹配的，因此我们需要一套可以帮我们校验语法的规则，即语义分析。

### Context-Free Grammar

上面的例子可以看出正则语言不足以帮助我们分析代码的语义。实际上任何的正则表达式都可以用一种”语法“代替，比如正则式`regex=r'p+i?'`可用下面语法代替

```
regex -> pplus iopt
pplus -> p pplus
pplus -> p
iopt -> i
iopt -> ϵ
```
上面的语法可称为Context-Free Grammar，即句子中的符号可进行等价替换，与符号前后的上下文无关。

> 关于Context-Free Grammar的替换规则可参考之前编译原理相关文章

我们可以用Context-Free Grammar来表示任何正则语言，例如

```
r'ab' = g -> ab

r'a*' = g ->  ϵ
        g -> a g

r'a|b' = g -> a
         g -> b
```