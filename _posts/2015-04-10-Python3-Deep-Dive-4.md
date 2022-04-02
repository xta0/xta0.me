---
title: Iteration and Generators
list_title: Python Deep Dive | Iteration and Generators
layout: post
categories: [Python]
---

### Lists vs Tuples

我们可以先对比一下List和Tuple的bytecode

<div class="highlight md-flex-h md-margin-bottom-24">
<div>
<pre class="highlight language-javascript md-no-padding-v md-height-full">
<code class="language-python">
dis(compile('(1, 2, 3, "a")', 'string', 'eval'))

0 LOAD_CONST      0 ((1, 2, 3, 'a'))
2 RETURN_VALUE
</code>
</pre>
</div>
<div class="md-margin-left-12">
<pre class="highlight language-python md-no-padding-v md-height-full">
<code class="language-python">
dis(compile('[1, 2, 3, "a"]', 'string', 'eval'))

0 LOAD_CONST               0 (1)
2 LOAD_CONST               1 (2)
4 LOAD_CONST               2 (3)
6 LOAD_CONST               3 ('a')
8 BUILD_LIST               4
10 RETURN_VALUE
</code>
</pre>
</div>
</div>

上面例子中，无论是tuple还是list，他们包含的是immutbale的data，我们可以发现，bytecode中tuple是一个constant，可以直接load，
而list需要通过更多的op code去动态创建。这是因为tuple本身是Immutable的数据结构。

如果tuple中包含mutable的数据结构，则tuple的创建将变的和list一样

```python
dis(compile('([1, 2], 3, "a")', 'string', 'eval'))

0 LOAD_CONST               0 (1)
2 LOAD_CONST               1 (2)
4 BUILD_LIST               2
6 LOAD_CONST               2 (3)
8 LOAD_CONST               3 ('a')
10 BUILD_TUPLE             3
12 RETURN_VALUE
```
此外，由于list的动态性，它需要preallocate space，因此它比tuple在内存上有更大的overhead

```python
import sys
s1 = sys.getsizeof((1,)) #48
s2 = sys.getsizeof([1,]) #64
```

### More on Tuples (NamedTuple)

### Copying Sequences

对于mutable sequence，一个问题是当它做参数时，如何确保它不会被mutate，比如下面例子

```python
x = [1,2,3]
def reverse(x):
    x.reverse()
    return x
y = reverse(x)
```
当然，上面这个例子有些牵强，一般来说不会有人这么写代码

### Slicing

### List Comprehensions

## Generator

## Context Managers