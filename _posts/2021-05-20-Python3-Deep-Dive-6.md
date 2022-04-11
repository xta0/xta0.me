---
title: Module
list_title: Python Deep Dive | Module
layout: post
categories: [Python]
---

## What is a Module

Python中的module只是一个`Module`类型的instance。比如我们import一个module

```python
import math
type(math) #<class 'module'>

import types
isinstance(math, types.ModuleType) # True
```
此时`math`会作为一个object出现在global namespace中

```python
> globals()

#...

'math': <module 'math' from ...>}
```
Python中的namespace是一个dictionary，因此我们可以直接访问`math`对象，并调用它里面的函数

```python
math = globals()['math']
math.sqrt(2) #1.4142...
```
实际上`math`不仅被注册到global的namespace中，它还被注册到了system的cache里

```python
id(math) #140636556863584

import sys
id(sys.modules['math']) # 140636556863584
```

重复`import math`并不会创建新的math object，因此它已经被load到system的module里了。

我们可以用`math.__dict__`来查看module中包含那些attributes，通常是一些函数，例如我们可以用下面的方式来调用`sqrt`

```python
math.__dict__
f = math.__dict__['sqrt']
```
类似的使用`dir(math)`也可以达到同样效果

## How does Python `import` Modules

当import一个module的时候，Python会从`sys.path`中获取module的path，因此我们确保被import的module在这个`path`中

```python
import sys
print(sys.path)
# ['', '/Users/taox/anaconda/lib/python38.zip', 
# '/Users/taox/anaconda/lib/python3.8', 
# '/Users/taox/anaconda/lib/python3.8/lib-dynload', 
# '/Users/taox/anaconda/lib/python3.8/site-packages']
```
在`import`的时候，module会被创建出来，，同时，module中的代码会被执行。当module被import之后，它会被注册到`sys.path`中，并缓存到cache里，此时，如果我们再次`import`这个module，系统会从cache中直接load，而并不会再执行module中的代码

```python
# main.py
import module1.py

print(globals())
# 'module1': <module 'module1' from ...>} 
```
上述代码中，我们假设在`main.py`中import了一个module，此时`module1`将出现在main.py的namespace中。

