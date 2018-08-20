---
layout: post
title: Python和Javascript解释器的一点差异
list_title: Python和JavaScript解释器的一点差异
categories: [Python, JavaScript]
---

### 解释器的差别

看下面代码，你觉得那个是正确的呢？还是都不正确或者都正确呢？

<div class="highlight md-flex-h md-margin-bottom-20">
<div>
<pre class="highlight language-javascript md-no-padding-v md-height-full">
<code class="language-python">
#Javascript
function func1(){
	return func2();
}

func1();

function func2(){
	return "running func2";
}
</code>
</pre>
</div>
<div class="md-margin-left-12">
<pre class="highlight language-python md-no-padding-v md-height-full">
<code class="language-python">
#python
def func1():
    return func2() 

func1()

def func2():
    return "running func2"
</code>
</pre>
</div>
</div>

上面的两段代码，Javascript代码可以正常执行，Python代码则报错。错误的原因是:

```
NameError: name 'func2' is not defined
```

从这个例子可以看出，Python的解释器的设计和Javascript似乎有些区别。在分析具体原因之前，先来回顾一下[编程语言的原理](2014/04/24/Programming-Language-1-1.html)，对于任何一条表达式，编译器都需要确定三个问题

1. Syntax
2. Type-Checking Rules
3. Evaluation Rules

对于Function来说，在编译器确定完其类型后便将这个符号（函数名或者是按照某种规则mangle后的名字）放入了static enviroment中，留着运行时调用。而函数的Evaluation的规则是在运行时求值，对函数内部的符号是从static environment中寻找，找不到则报错。上面例子中，在执行`func1()`时，Python和JS均会在static environment中寻找`func2`，显然一个找到了，另一个没找到，因此，分歧可能出在`func2`这个符号注册的时机上。

接下来，我们可以大致分析一下JS和Python的解释器是怎么工作的。对于JS来说，在执行前代码前，对所代码从头至尾进行扫描，如果出现static enviroment中没有的符号，则向其内部注册该符号，并赋初值undefined（这个特性据说做Hoisting）。注意在static environment中并不会对符号求值，求值的过程在dynamic environment中。而python的解释器似乎不会提前在static enviroment中注册所有符号，而是在运行时不断更新static enviroment中的符号, 并在dynamic environment中对其求值，当然如果发现没该有符号，则会在求值的过程中报错。

哪种设计合理呢？感觉Python解释器的设计更合理一些，JS在执行前要扫描并注册所有符号，其效率显然不如逐句解释来的快，并且一般有良好变成素养的程序员也不会写出上面的代码。


<p class="md-h-center">(全文完)</p>


