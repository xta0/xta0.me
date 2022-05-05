---
title: Modules and Packages
list_title: Python Deep Dive | Modules and Packages
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
我们可以扩展这个`sys.path`从而让Python可从我们指定的地方import module

```python
sys.path.append('path_to_module')
```

在`import`的时候，module会被创建出来，同时，module中的代码会被执行。当module被import之后，它会被注册到`sys.path`中，并缓存到cache里(`sys.modules`)，此时，如果我们再次`import`这个module，系统会从cache中直接load，而并不会再执行module中的代码

```python
# main.py
import module1.py

print(globals())
# 'module1': <module 'module1' from ...>}
```
上述代码中，我们假设在`main.py`中import了一个module，此时`module1`将出现在main.py的namespace中。当我们运行一个module的时候，module的名字会被改成`__main__`。

小结一下，当`import math`时，Python会做下面两件事情

1. Check `sys.modules()`. If not there, load it and insert it.
2. Add `math` to the global namespace (`globals()`) of the module which imports it.

默认情况下，一个python文件是一个module。但是我们也可以在运行时动态创建module，比如下面代码中，我们可以自定义一个`importer`，它从`module1_src.py`中读入代码，并动态创建一个module

```python
# importer.py

import os.path
import types
import sys


def import_(module_name, module_file, module_path);
    if module_name in sys.modules:
            return sys.modules[module_name]

    module_rel_file_path = os.path.join(module_path, module_file)
    module_abs_file_path = os.path.abspath(module_rel_file_path)

    # read code from file
    with open(module_rel_file_path, 'r') as code:
        source_code = code.read()

    # create a module object
    mod = types.ModuleType(module_name)
    mod.__file__ = module_abs_file_path

    # set a ref in sys modules
    sys.modules[module_name] = mod

    # compile source code
    code = compile(source_code, filename=module_abs_file_path, mode='exec')

    # execute compiled source code
    exec(code, mod.__dict__)

    return sys.modules[module_name]

# main.py

import sys
import importer

impoter.import_('module1', 'module1_src.py', '.')
# module1 has been registered to sys.modules

import module1 # now module1 is avaiable for `import`
```

Python中的每个module都有`__spec__`方法，它包含module的位置和它的loader信息

```shell
>>> import fractions
>>> fractions.__spec__
ModuleSpec(name='fractions', loader=<_frozen_importlib_external.SourceFileLoader object at 0x7fa3c817b880>, origin='/Users/taox/anaconda/lib/python3.7/fractions.py')
```
## Packages

Package在python中也是一种module，但是不是所有的module都是package。当我们`import`一个module时，可以查看其`__path__`的值，如果存在则表示它是一个package，否则是一个module。Package的import规则为

```python
import pack1.pack1_1.module1
```
此时，Python会先执行`import pack1`，然后`import pack1.pack1_1`，最后`import pack1.pack1_1.module1`。因此，他们均会出现在`sys.module()`里面。

多数情况下Python中的package是基于文件结构，directory名即是package的名字，同时，我们需要创建一个`__init__.py`在该direcotry下面。此时，Python会知道当前directory是一个package。如果我们不创建`__init__.py`，Python会创建一个implicit的namespace package