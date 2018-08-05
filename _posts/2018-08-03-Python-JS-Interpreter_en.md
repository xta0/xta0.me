---
layout: post
title: A Fun small fun fact
list_title: A little difference between two interpreters
---

### Interpreter difference

Look at the code below, do you think that is correct? Still not correct or correct?

<div class="highlight md-flex-h">
<div>
<pre class="highlight language-javascript md-no-padding-v">
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
<pre class="highlight language-python md-no-padding-v">
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

As you can see from this example, Python's interpreter design is very different from Javascript. JS scans the code from start to finish before execution. If there is a symbol that is not in static enviroment, the symbol is registered internally. Assign the initial value undefined, note that the symbol is not evaluated in the static environment, the evaluation process is in the dynamic enviroment. The Python interpreter does not seem to register all symbols in the static enviroment in advance, but instead constantly updates the symbols in the Dynamic enviroment at runtime.