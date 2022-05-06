---
layout: post
list_title: Python Tips | Functions
title: Python Functions
categories: [Python]
---

## Function Arguments

如果一个函数的positional arg定义了default value，那么它后面的参数都需要定义default value

```python
def my_func(a, b=6, c=10):
    # code..
```
如果我们只想指定`a`和`c`，则需要用到keyword arguments

```python
my_func(a=1, c=2)
```
需要注意，如果某个参数使用了keyword argument，它后的参数必须也指定名字

```python
def my_func(a, b, c)
my_func(c=1, 2, 3) # wrong
my_func(1, b=2, 3) # wrong
my_func(1, b=2, c=3) # corret
my_func(1, c=2, b=3) # corret
```

使用默认参数有一些需要注意的地方，他们在函数创建时就是求值，比如

```python
def log(s, dt=datetime.utcnow()):
    #code

log('hello')
```
此时，如果我们不传入`date`，那么`date`的时间将永远是函数创建的时间，而不是函数调用的时间。解决这个问题，可以将默认参数生命为`None`

```python
def log(s, dt=None):
    dt = dt or datetime.utcnow()
    # code
```

另一个需要注意的地方是尽量不要用mutable的数据结构，比如

```python
args = [1,2,3]
def my_func(list = args)
    print(args)

args.append(4)
my_func()
```
此时函数的默认值`list`会被mutate，导致结果不符合预期。和上面例子类似的另一个例子是空数组做默认值

```python
def add(name, name_list=[]):
    name_list.add(name)
    return name_list

l1 = add('jon')
l2 = add('kate')
```
上述代码中，`l1`和`l2`指向的实际上是同一个数组，因此`l1`和`l2`中的内容是一样的，当第二次调用`add`的时候，`name_list`已经不是空的了。解决办法和上面一样


```python
def add(name, name_list=None):
    if not name_list:
        name_list = []
    name_list.add(name)
    return name_list
```


### Pack与Unpack

所谓Packed Values是指一些值以某种方式被pack到一起，最常见的有tuple, list, string, set, 和map这些集合类。对于这些集合类，Python提供了一种展开的方式，即将集合类中的元素以tuple的形式展开

```python
a,b,c = [1,2,3] #a->1, b->2, c->3
a,b,c = 10,20,'hello' #a->10, b->20, c->hello
a,b,c = 10, {1,2}, ['a','b'] #a->10, b->{1,2}, c->['a','b']
a,b,c = 'xyz' #a->x, b->y, c->z
```
上述代码中，等号左边定义了一个tuple，右边是一个集合对象，unpacking的方式是按照位置一一对应。看起来所谓unpacking，实际上就是对集合类对象进行`for`循环为变量依次赋值。

但是对于哈希表，`for`循环只得到`key`，因此unpacking的结果也是key，且由于哈希表是无序的，unpacking出来的结果也是不确定的，对于set同理。

```python
# unpacking a map object
d = {'key1':1, 'key2':2, 'key3':3}
a,b,c = d #a->'key2' b->'key3', c='key1'

# unpacking a set object
s = {'x','y','z'}
a,b,c = s #a->'z' b->'x', c='y'
```

在Python中，unpacking对于实现swap功能很方便，只需要unpack一次即可

```python
#swap a,b
a,b = b,a
```
上面代码的执行顺序是先进行RHS求值，然后将得到的值再进行LHS赋值给`a,b`。

### 使用`*`和`**`进行unpack

在Python 3.5后，可使用`*`做局部的unpack，比如一个集合，我只想unpack第一个元素，然后将剩下部分unpack给另一个变量

```python
l = [1,2,3,4,5,6]

#using slicing
a = l[0]
b = l[1:]

#using simple unpacking
a,b = l[0]:l[1:]

#using * operator
a, *b = l
```
slicing只适用于数组，而`*`可适用于任何iterable的集合变量，对于有序集合，`*` unpack的结果为为数组

```python
a, *b = (1,2,3) #a = 1, b = [2,3]
a, *b = "abc" #a = 'a', b = ['b','c']  
a, *b, c = 1,2,3,4 #a = 1, b = [2,3], c = 4
```

对于无序集合，比如dict和set，unpack的结果是无序的

```python
d1 = {'p':1, 'y':2}
d2 = {'t':3, 'h':4}
d3 = {'o':5, 'n':6}

d = [*d1, *d2, *d3] # unpack key
```
对于dict来说，`*` 只能unpack key，如果想要unpack pair，需要用`**`，但是只能用在等号右边

```python
d = {**d1, **d2, **d3}
x = {'a':5, 'b':6, **d1}
```

### `*args` and `*kwargs`

`*args`可以unpack 任意多个positional arguments，如果作为函数参数，则它后面不能再有其它的positional args

```python
def func1(a,b,c):
    #code

l = [10, 20 ,30]
func(l) # wrong, func1 expects three positional args
func(*l) # works

def func2(a, b, *args, d):
    # code

func2(10, 20, 'a', 'b', 100) #wrong
```
前面提到，我们也可以用keyword argument来给函数传参

```python
func1(a=1, c=2, b=3)
```
如果想要强制用keyword argument传参，那么keyword argument需要定义在`*args`的后面，并且调用时必须传入

```python
def func1(a,b,*args, d):
    # code
func1(1, 2, 'x', 'y',d = 100) # correct
func1(1, 2, d = 100) # correct
func1(1, 2) # wrong d must be assigned
```
如果在keyword argument之前不需要任何positional arguments，则可以用`*`代替

```python
def func(*, d): 
    # code 
func(1, 2, d=100) # wrong
func(d = 100) #correct
```
`*`说明`d`前面没有任何的positional argument，注意这个和`*args`不同，后者表示有任意多个positional arg

和`*args`类似，`**kwargs`可以指定任意数量的keyword arguments，但是`**kwargs`后面不能再定义参数

```python
def func(*, d, **kwrags):
    # code
func(d=1, a=2, b=3)
func(d=1)
```
上面例子中，`*`表示函数不接受任何positional arguments，`d`是一个keyword-only argument


## Resources

- [Effective Python]()
- [Real Python]()