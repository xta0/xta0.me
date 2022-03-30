---
layout: post
list_title: Python 语法速查(二) | Python3 Cheatsheet
title: Python 语法速查(二)
categories: [Python]
---

## Functions

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

对于默认参数有一个需要注意的地方，他们在函数创建时就是求值，比如

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

- 使用`*`和`**`进行unpack

在Python 3.5后，可使用`*`做局部的unpack，比如一个集合，我只想unpack第一个元素，然后将剩下部分unpack给另一个变量

```python
l = [1,2,3,4,5,6]

#using slicking
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


### Generator

Generator是Python中协程的实现

```python
def gen_cube(n):
    for x in range(n):
        yield x**3
#返回generator object        
gen_cube(4) #<generator object gen_cube at 0x10567b150>
#pull value from gen_cube
list(gen_cube(4)) #[0, 1, 8, 27]

def gen_fib(n):
    a = 1
    b = 1
    for i in n:
        yield a
        a,b = b,a+b
for number in gen_fib(10):
    print(number)

def simple_gen():
    for i in range(3):
        yield i
g = simple_gen()
next(g) #0
next(g) #1
next(g) #2

s = "hello"
next(iter(s))
```


### Async/Wait

- `async`修饰的函数为一个coroutine对象，

### Regular Expression

- 正则表达式字符串以`r"regex"`表示

```py
# List of patterns to search for
patterns = [ 'term1', 'term2' ]
# Text to parse
text = 'This is a string with term1, but it does not have the other term.'
result1 = re.search(patterns[0],text)
result2 = re.search(patterns[1],text)
print(result1)#<_sre.SRE_Match object; span=(22, 27), match='term1'> None
print(result2) #none
result1.start() #22
result1.end() #27

##split
split_term = "@"
phrase = "jayson.xu@foxmail.com"
list1 = re.split(split_term, phrase)
print(list1) #['jayson.xu', 'foxmail.com
```

- 匹配字符的几种方式
    1.  以`*`结尾，表示被匹配的字符出现0次或者多次
    2.  以`+`结尾，表示被匹配的字符至少出现1次
    3.  以`?`结尾，表示被匹配的字符出现0次或者1次
    4.  以`{m}`结尾，表示被匹配的字符出现m次
    5.  以`{m,n}`结尾，表示被匹配的字符出现[m,n]次
    6.  以`[mn..]`结尾，表示被匹配的字符是m或者n或者...

```python
import re

test_phrase = 'sdsd..sssddd...sdddsddd...dsds...dsssss...sdddd'
test_patterns = [ 'sd*',     # s followed by zero or more d's
                'sd+',          # s followed by one or more d's
                'sd?',          # s followed by zero or one d's
                'sd{3}',        # s followed by three d's
                'sd{2,3}',      # s followed by two to three d's
                '[sd]',          #either s or d
                's[sd]+'        #s followed by one or more s or d
                ]
# ['sd', 'sd', 's', 's', 'sddd', 'sddd', 'sddd', 'sd', 's', 's', 's', 's', 's', 's', 'sdddd']                            
print(re.findall(test_patterns[0],test_phrase))
# ['sd', 'sd', 'sddd', 'sddd', 'sddd', 'sd', 'sdddd']
print(re.findall(test_patterns[1],test_phrase))
#['sd', 'sd', 's', 's', 'sd', 'sd', 'sd', 'sd', 's', 's', 's', 's', 's', 's', 'sd']
print(re.findall(test_patterns[2],test_phrase))
#['sddd', 'sddd', 'sddd', 'sddd']
print(re.findall(test_patterns[3],test_phrase))
#['sddd', 'sddd', 'sddd', 'sddd']
print(re.findall(test_patterns[4],test_phrase))
```    

- <mark>过滤</mark>某些字符
    - `[^...]`会匹配文本中不在`[]`中的字符

```python
test_phrase = 'This is a string! But it has punctuation. How can we remove it?'
re.findall('[^!.? ]+',test_phrase)
```

- 匹配<mark>英文字符</mark>

```python
test_phrase = 'This is an example sentence. Lets see if we can find some letters.'

test_patterns=[ '[a-z]+',      # sequences of lower case letters
                '[A-Z]+',      # sequences of upper case letters
                '[a-zA-Z]+',   # sequences of lower or upper case letters
                '[A-Z][a-z]+'] # one upper case letter followed by lower case letters
```
- <mark>匹配特殊字符</mark>


Code |Meaning
---|---
`\d` | a digit
`\D` | a non-digit
`\s` | whitespace (tab, space, newline, etc.)
`\S` | non-whitespace
`\w` | alphanumeric
`\W` | non-alphanumeric

```python
test_phrase = 'This is a string with some numbers 1233 and a symbol #hashtag'

test_patterns=[ r'\d+', # sequence of digits
                r'\D+', # sequence of non-digits
                r'\s+', # sequence of whitespace
                r'\S+', # sequence of non-whitespace
                r'\w+', # alphanumeric characters
                r'\W+', # non-alphanumeric
                ]
#['1233']
print(re.findall(test_patterns[0],test_phrase))
#['This is a string with some numbers ', ' and a symbol #hashtag']
print(re.findall(test_patterns[1],test_phrase))
#[' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ']
print(re.findall(test_patterns[2],test_phrase))
#['This', 'is', 'a', 'string', 'with', 'some', 'numbers', '1233', 'and', 'a', 'symbol', '#hashtag']
print(re.findall(test_patterns[3],test_phrase))
#['This', 'is', 'a', 'string', 'with', 'some', 'numbers', '1233', 'and', 'a', 'symbol', 'hashtag']
print(re.findall(test_patterns[4],test_phrase))
#[' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' #']
print(re.findall(test_patterns[5],test_phrase))
```
### StringIO

StringIO提供了一个在内存中读写字符串的方式，通过string构建StringIO对象来进行IO操作，可以像操作文件一样操作string

```python
from io import StringIO
# Arbitrary String
message = 'This is just a normal string.'
# Use StringIO method to set as file object
f = StringIO(message)
str = f.read() #'This is just a normal string.'
f.write(' Second line written to file like object')
# Reset cursor just like you would a file
f.seek(0)
# Read again
str = f.read() #Second line written to file like object
```

## STL Module

类似C++的STL

### Counter

返回集合中元素的出现次数

```python
from collections import Counter

#统计元素出现次数
l = [1,1,1,3,3,3,4,2,2]
Counter(1) #Counter({1: 3, 3: 3, 2: 2, 4: 1})
s = 'asssvavaasvsbsa'
Counter(s) #Counter({'s': 6, 'a': 5, 'v': 3, 'b': 1})
ss = 'How How test is is gonna gonna work work work out'
words = ss.split(' ')
c = Counter(words) #Counter({'work': 3, 'How': 2, 'is': 2, 'gonna': 2, 'test': 1, 'out': 1})
c.most_common(2) 
```
其它成员函数

```python
sum(c.values())                 # total of all counts
c.clear()                       # reset all counts
list(c)                         # list unique elements
set(c)                          # convert to a set
dict(c)                         # convert to a regular dictionary
c.items()                       # convert to a list of (elem, cnt) pairs
Counter(dict(list_of_pairs))    # convert from a list of (elem, cnt) pairs
c.most_common()[:-n-1:-1]       # n least common elements
c += Counter()                  # remove zero and negative counts
```

### Default Dict

更安全的dictionary，对于访问不存在的key，不会报错

```python
from collections import defaultdict
d = defaultdict(object)
d['one'] #访问一个不存在的key，返回一个<object object at 0x105424110>

#自定义默认value
d = defaultdict(lambda: 0) #对于不存在的key，返回value = 0
d['one'] #0
```

### Ordered Dict

有序字典

```python
from collections import OrderedDict
d = OrderedDict()
d['a'] = 1
d['b'] = 2
d['c'] = 3
d['d'] = 4
for k,v in d.items():
    print(k,v) #顺序输出

##比较
d1 = {"a":1,"b":2}
d2 = {"b":2,"a":1}
d1 == d2 #True

d1 = OrderedDict()
d1['a'] = 1
d1['b'] = 2
d2 = OrderedDict()
d2['b'] = 2
d2['a'] = 1
d1 == d2 #False
```

### Named Tuple

可以用名字去索引的tuple

```python
from collections import namedtuple
Dog = namedtuple('Dog','age breed name')
sam = Dog(age=2, breed='Lab', name='Sammy')
sam.age
sam.breed
sam.name
```
### Datetime 

```python
import datetime
t = datetime.time(5,25,1) #时，分，秒
print(t) #05:25:01
print(datetime.time.min) #00:00:00
print(datetime.time.max) #23:59:59.999999
print(datetime.time.resolution) #0:00:00.000001

today = datetime.date.today()
today.timetuple()

d1 = datetime.date(2015,3,11)
print(d1)
d2 = d1.replace(year=1990)
```

## Python Debugger

- 使用`pdb`打断点

```python
import pdb

x = 1
y = [12,2,3]
z = 10
r1 = x+z
pdb.set_trace() #断点调试, q退出
r2 = y+z
```

- 使用`timeit`计算代码执行时间

```py
import timeit
#将待测试代码执行1000次
timeit.timeit('"-".join(str(n) for n in range(100))', number=1000)#0.034958536038175225
```