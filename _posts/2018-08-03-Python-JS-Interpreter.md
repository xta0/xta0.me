---
layout: post
title: A Fun small fun fact
list_title: Python和Javascript解释器的一点差异
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

func_1();

function func2(){
	return "running func_2";
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

func1();  

def func2():
    return "running func2"
</code>
</pre>
</div>
</div>

这个例子可以看出，Python的解释器设计和Javascript有很大区别，JS是在执行前对所代码从头至尾进行扫描，如果出现static enviroment中没有的符号，则向其内部注册该符号，并赋初值undefined，注意在static environment中并不会对符号求值，求值的过程在dynamic enviroment中。而python的解释器似乎不会提前在static enviroment中注册所有符号，而是在运行时不断更新Dynamic enviroment中的符号, 如果发现没有符号，则报错。
