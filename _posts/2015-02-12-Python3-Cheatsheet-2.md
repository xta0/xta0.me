---
layout: post
list_title: Python 语法速查(二) | Python3 Cheatsheet
title: Python 语法速查(二)
categories: [Python]
---

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