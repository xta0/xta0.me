---
list_title: Python3 基本语法 | Basic Syntax
layout: post
title: Python3 Basic Syntax
---

## Language Overview

- **Numbers**
    - Integral
        - Integer
        - Boolean
    - Non-Integral
        - Floats(c doubles)
        - Complex
        - Decimals
        - Fractions
- **Collections**
    - Sequences
        - Mutable
            - Lists
        - Immutable
            - Tuples
            - String
    - Sets
        - Mutable
            - Sets
        - Immutable
            - Frozen Setes
    - Mappings
        - Dictionaries

- **Callables**
    - User-Defined Functions
    - Generators
    - Classes
    - Instance Methods
    - Class Instances(`__call__()`)
    - Built-in Functions (e.g. `len()`,`open()`)
    - Built-in Methods (e.g. `my_list.append(x)`)

- **Singletons**
    - None
    - NotImplemented
    - Ellipsis(...)

## Strings

## Naming Conventions

## Conditionals

## Functions

## Loops

### Printing

- 链接字符串

```py
a="abc"
b="123"
print(a+b)
```
- 使用`,`输出空格

```py
print("Hens", 25+30/6); #Hens 30
print("Hens", "Kay", "Thomas"); #Hens Kay Thomas
print("Is it greater or equal?", 5 >= -2) #Is it greater or equal? True
```
- 格式化字符串
    - 使用format

    ```python
    age_eval = "my age is {}"
    print(age_eval.format(age)) #my age is 33
    print("This is a string {}".format('INSERTED'))
    print("The {} {} {} ".format(1,2,3))
    print("The {0} {0} {0}".format(20.3,100)) #The 20.3 20.3 20.3
    print("The {q} {b} {f}".format(f='fox', b='brown', q='quick')) #The quick brown fox
    #格式化浮点数:{value:width.precision f}
    result = 100/777 #0.1287001287001287
    print("result is {r:1.3f}".format(r=result)) #result is 0.129

    ```
    - 使用`f-string`

    ```py
    name="tao"
    age = 33;
    #使用fstring
    print(f"{name} is {age} years old ") #my name is tao
    ```
- 转义字符

Escape | What it does.
------------- | -------------
`\\` |  Backslash (`\`)
`\'` | Single-quote (`’`)
`\"` | Double-quote (`”`)
`\a` | ASCII bell (BEL)
`\b` | ASCII backspace (BS)
`\f` | ASCII formfeed (FF)
`\n` | ASCII linefeed (LF)
`\N` | {name} Character named name in the Unicode database (Unicode only)
`\r` | Carriage Return (CR)
`\t` | Horizontal Tab (TAB)
`\uxxxx` | Character with 16-bit hex value `xxxx`
`\Uxxxxxxxx` | Character with 32-bit hex value `xxxxxxxx`
`\v` | ASCII vertical tab (VT)
`\ooo` | Character with octal value `ooo`
`\xhh` | Character with hex value `hh`


### Primary Types

- `locals()/globals()`

```py
#查看全局变量
print(globals())
#{'__name__': '__main__', '__doc__': None, '__package__': None, '__loader__': <class '_frozen_importlib.BuiltinImporter'>, '__spec__': None, '__annotations__': {}, '__builtins__': <module 'builtins' (built-in)>}

def func:
    a = 'some text'
    b = 100
    #查看函数内的局部变量
    print(locals()) #{'b': 1, 'a': 's'}
```


### Datastructures

- **Numbers**   
    - 进制转换
        - 十六进制：`hex(12)` 
        - 二进制：`bin(1234)`
    - 内置数值运算
        - 次方:`pow(2,4)`等价`2**4`
        - 绝对值: `abs(-2)`
        - 四舍五入：`round(3.9) #4.0`

- **String**
    - 支持`[]`索引

    ```py
    a = "hello";
    a[0] #h
    a[-1] #o
    ```
    - 字符串常量是Immutable,不能用`[]`的方式改变字符串内容

    ```python
    name="SAM"
    name[0]='P' #TypeError: 'str' object does not support item assignment
    ```
    - `fstring`格式化字符串

    ```python
    binary = "binary"
    do_not = "don't"
    y = f"Those who know {binary} and those who {do_not}."
    ```

    - 获取字串
        - 使用`[]`索引,格式为`[起始index:结束index:步长]`

        ```python
        a="hello"
        a[1:] #ello, 包括第一个字符
        a[:3] #hel, 不包括第三个字符
        a[1:3] #el
        a[1:-1] #ell
        a[:] #hello
        a[::] #hello
        a[::2] #hlo 步长是2，抽取字串
        a[1:-1:2] #el,起始1，终点-1，步长2
        a[::-1]#olleh, 反转字符串
        ```
    - 正则表达
        
        ```python
        s = 'hello'
        s.split('e') #['h','llo']
        s.partition('l') #('he', 'l', 'lo')
        ```
    - 其它API
        - 首字母大写: `s.capitalize` 
        - 大小写转换: `s.lower()`,`s.upper()` 
        - 字符出现次数:`s.count('o')`
        - 字符出现位置: `s.find('o')`
        - 检查字符是否是数字或字母: `s.isalum()`
        - 检查字符是否是字母:`s.isalpha()`
        - 开头结尾：`s.startswith(str)`/`s.endswith(str)`

- **list**
    - 创建list

    ```python
    mylist=[1,2,'three']
    mylist[2] #three
    len(mylist) #长度
    mylist[1] = 10
    mylist.index(2) #1
    ```
    - 追加元素

    ```python
    list1=['one','two']
    list1.append('three') #['one', 'two', 'three']
    list1.append([1,2]) #['one', 'two', 'three', [1, 2]]
    list1.insert(2,'str') #['one', 'two', 'str', 'three',[1, 2]]
    ```
    - 删除元素

    ```python
    val = list.pop() #默认删除尾部
    val = list.pop(-1) #删除尾部
    val = list.pop(2) # 删除index=2的元素
    list1 = [1,2,2,3,4]
    list1.remove(2) #删除数组中第一个2
    ```
    - 拼接list

    ```python
    list1=[1,2,3]
    list2=[4,5]
    list3 = list1 + list2 #[1, 2, 3, 4, 5]
    list1.extend(list2) #等价于list1 = list1+list2
    ```
    - 其它API

    ```python
    list1=[1,2,3]
    max_num = max(list)
    min_num = min(list)
    list1.revers()
    list1.sort()
    ```
    - functional API

    ```python
    mylist1 = [x for x in 'abc'] # ['a',b,'c']
    mylist2 = [num**2 for num in range(0,11)] #[0, 1, 4, 9, 16, 25, 36, 49, 64, 81, 100]
    mylist3 = [x for x in range(0,11) if x%2==1]#[1, 3, 5, 7, 9]
    mylist4 = [x if x%2 == 0 else 'ODD' for x in range(0,11)] #[0, 'ODD', 2, 'ODD', 4, 'ODD', 6, 'ODD', 8, 'ODD', 10]
    mylist5 = [x*y for x in [2,4,6] for y in [1,10,100]] #[2, 20, 200, 4, 40, 400, 6, 60, 600]

    ##返回一个参数为偶数的数组
    def myfunc(*args):
        return [x for x in args if x%2 == 0]
    ```

- **Dictionary**
    - 无序map

    ```python
    d={'k1':123, 'k2':[0,1,2], 'k3':{'insidekey':100}}
    d['k2'] #[0,1,2]
    d['k4']="abc"
    keys = d.keys() 
    ```

    - functional API

    ```python
    #{key:value | 规则}
    d = {x:x**2 for x in range(10)}#{0: 0, 1: 1, 2: 4, 3: 9, 4: 16, 5: 25, 6: 36, 7: 49, 8: 64, 9: 81}
    #{key:value | zip(k,v)}
    d = {k:v**2 for k,v in zip(['a','b'],range(2))} #{'a': 0, 'b': 1}
    ```

-  **Tuples**
    - Immutable，不能修改tuple中的元素
    - Tuple uses parenthesis: `(1,2,3)`
    - Only two methods
        - `index`
        - `count`

    ```python
    t=(1,2,3)
    type(t) #tuple
    t=('one',2)
    t[0] #one
    t[1] 2 #2
    t=('a','a',2)
    t.count('a') #2
    t.index('a') #0
    t[0]='NEW' #TypeError
    ```

- **Sets**
    - Unordered collections of unique elements

    ```python
    myset = set()
    #添加
    myset.add(1)
    myset.add(2)
    #删除
    myset.discard(2)
    #清空
    myset.clear() 
    # create set using array
    nset = set([1,2,1,2]) #{1,2}
    #copy
    cset = nset.copy()
    #求差集
    s1 = {1,2,3}
    s2 = {2,3,4,5}
    s2.difference(s1) #{2, 3, 4, 5}
    s2.difference_update(s1) #s2更新为二者差集{2, 3, 4, 5}

    #求交集
    s1 = {1,2,3}
    s2 = {1,2,4}
    s1.intersection(s2) #{1,2}
    s3 = {5,6}
    s1.disjoint(s3) #无交集返回True，有交集返回False

    #父集子集
    s1.issubset(s2)
    s1.issupperset(s1)

    #求并集
    s1.union(s2)
    s1.update(s2) #将s1更新为s1,s2的并集
    ```

- **File**
    - mode
        - `r`: read
        - `w`: write
        - `r+`
        - `w+`
        - `a` : append 

    ```python
    myfile=open("./test.txt",mode='r') #_io.TextIOWrapper
    content = myfile.read() #string
    myfile.seek(0) #move file cursor to front
    myfile.readlines() #list
    myfile.close()
    file = open('TEXT.txt',mode='w')
    file.write('THis is a test file!')
    file.close()
    ```
    - 使用`with`操作

    ```py
    with open("welcome.txt") as file: # Use file to refer to the file object
        data = file.read()
        #do something with data
   ```


### Statements

- **And,Or,Not**

```python
1<2 and 2<3
2<3>10 #false, 等价于2<3 and 3>10
100==1 or 2==2
not 1==1
```
- **if-elif-else**

```
if some_condition: #注意冒号
    #execute some code 
elif some_other_condition:
    #do something different
else:
    # do something else

#三元运算
condition_is_true if condition else condition_is_false
```

- **for-loops**

```pyton
list = [1,2,3]
for item in list: #注意冒号
    print(item)

#只关注循环次数
for _ in list:
    print('cool')

#遍历tuple list
mylist = [(1,2),(3,4)]
for t in mylist:
    print(t) #(1,2) (3,4)
for (a,b) in mylist: #使用pattern matching
    print(a) #1 3
    print(b) #2 4

#遍历map list
d = {'k1':1, 'k2':2, 'k3':3}
for item in d:
    print(item) #k1,k2,k3  #只返回key

for item in d.items(): #返回key-value
    print(item) #('k1',1),('k2',2),('k3',3)

for key,value in d.items(): #使用pattern matching
    print(value)
```

- **while-loops**

```
while some_boolean_condition:
    #do something
else:
    #do something different
```
```python
x = 0 
while x<5:
    print(f'value of x is {x}')
    x += 1
else:
    print('loop end')
```

- **stop-loop**
    - `break`: Breaks out of the current closest enclosing loop.
    - `continue`: Goes to the top of the closest loop.
    - `pass`: Does nothing at all.


- **Useful Operators**
     - `range`
    
    ```python
    for num in range(3,10,2):
        print(num) #打印3到9（不包括10），步长为2的数
    list(range(0,11,2)) #产生0-10的偶数
    ```
    - `enumerate`

    ```python
    word = 'adc'
    for item in enumerate(word): #返回一组tuple
        print(item) #(0,'a')(1,'b')(2,'c')

    for index,letter in enumerate(word):
        print(index)
        print(letter)
    ```

    - `zip`

    ```python
    list1 = [1,2,3]
    list2 = ['a','b','c']

    for item in zip(list1, list2):
        print(item) #(1,'a'),(2,'b'),(3,'c')

    list3 = list(zip(list1,list2))
    ```

    - `in`

    ```python
    2 in [1,2,3] #True
    'a' in 'world' #True
    'mykey' in {'mykey':345} #True
    d = {'mykey':345}
    345 in d.values() #True
    ```

    - `import`

    ```python
    from random import shuffle #从random库中引用shuffle函数
    list1 = [1,2,3]
    shuffle(list1)
    ```

    - `input`

    ```python
    result input('what is your name?') #从键盘接受输入到result，类型是string
    type(result) #str
    int(result)
    float(result)
    ```

### Functions

- 定义

```python
def name_of_function(args): #注意冒号
    #some_code 
    print("Hello")  #注意tab
```
- 可变参数
    - `*args`: 将参数打包成tuple

    ```python
    def myfunc(*args): #可变参数，args是tuple
        return sum(args)*0.05
    myfunc(12,2,3)
    ```
    - `**kwargs`: 将参数打包成dictionary

    ```python
    def myfunc(**kwargs):
        print(kwargs) #可变参数，kwargs是map
        if 'fruit' in kwargs: #kargs中是否有key为fruit的参数
            print(kwargs['fruit'])
        else
            print('I did not find any fruit')

    myfunc(fruit='apple', veggie='lettuce')
    ```
    - 两者可以一起使用
    
    ```python
    def myfunc(*args, **kwargs):
        print(args)
        print(kwargs)
    myfunc(1,2,3,fruit='apple',veggie='lettuce')
    ```
    - unpack args

    ```python
    def print_two(*args)
        arg1, arg2 = args
        print(f"arg1: {arg1}, arg2: {arg2}")
    ```

- 函数性
    - first-class object
    
    ```python
    def hello():
        return 1
    greet = hello
    greet() #1

    del hello 
    greet() #1, 删除hello，greet仍然有效
    ```
    - 函数对象做函数参数

    ```python
    def func1:
        return 1
    def func2(f):
        f()
    func2(func1)
    ```
    - 函数对象做范数返回值

    ```python
    def func():
        def inner_func():
            return 'inner_func() called!'
        return inner_func

    func()() #'inner_func() called!'
    ```

- Nested Function

```python
def hello(name="Jason"):
    print('The hello() function has been executed!')

    #nested function
    def greet(): #greet()的作用域仅在hello内部
        return '\t this is the greet() func inside hello!'
    def welcome():
        return '\t this is the welcome() func inside hello!'
    
    if name == 'Jason':
        return greet
    else:
        return welcome

new_func = hello('Jason')
print(new_func) #this is the greet() func inside hello!
```

### Decorators

- 在不修改原函数的前提下，对已有函数进行扩展后，返回一个新的函数给原函数
- 在需要扩展的函数上面，使用`@`符号标记

```python
def some_decorator(some_func)    
    def wrap_func()
        #some code
        some_func()
        #some code 
    return wrap_func            
    
#通过@修饰，将simple_func传递给some_decorator函数，some_decorator将wrap_func返回给simple_func，这样在后面调用simple_func时就相当于调用了wrap_func()
@some_decorator 
def simple_func():
    #DO something
    return something
```


- Decorator的实现

```python
def new_decorator(orig_func):
    def wrap_func():
        #some code before execute  orig_func
        print('before orig_func')
        orig_func()
        #some code after execute orig_func
        print('after orig_func')
    return wrap_func

def fun_needs_decorator():
    print("fun needs decorator")

fun_needs_decorator = new_decorator(fun_needs_decorator)
fun_needs_decorator()

#使用 @符号
@new_decorator
def fun_needs_decorator():
    print("fun needs decorator")

fun_needs_decorator()#得到相同结果
```


### Lambda Expressions

- `map/filter`: 第一个参数是函数对象，第二个参数是数组

```python
def square(num):
    return num*num
def check_even(num):
    return num%2 == 0

list1 = [12,3,3]
list2 = map(square,list1)
list3 = filter(check_even,list1)
```
- `lamda expression`

```python
square = lambda num: num**2;
square(3)

mynums=[1,2,3,4]
list(map(lambda num:num**2,mynums))

```


### OOP

- `class`，成员变量，成员函数

```python
class NameOfClass(): #注意括号和冒号
    instance_variable = some_value #共有成员
    def __init__(self,param1, param2):
        self.param1 = param1 #定义私有成员
        self.param2 = param2
    def some_method(self): #成员函数
        print(self.param1)

class Dog():
    species="mammal" #定义共有成员变量
    def __init__(self,breed):
        self.name = breed #定义私有成员变量
    def bark(self): #定义成员函数
        print("WOOF") 

my_dog = Dog(breed='Lab')
type(my_dog)  #<class '__main__.Dog'>
print(my_dog.name)
print(my_dog.species)
```

- 继承

```python
class Animal():
    def __init__(self):
        print("Animal created")
    def who_am_i(self):
        print("I am an animal")
    def eat(self):
        print("I am eating")

class Dog(Animal): #继承
    def __init__(self):
        Animal.__init__(self) #调用父类构造
        print("Dog Created")
    def who_am_i(self): #override
        print("I am a dog")
```

- 抽象类

```python
class Animal():
    def __init__(self,name):
        self.name = name
    def speak(self):
        raise NotImplementedError("Subclass must implement this abstract method")

class Dog(Animal):
    def speak(self):
        return self.name + "say WOOF!"
```

- 特殊API

```python
class Book():
    def __init__(self,title,author):
        self.title = title
        self.author = author
        self.pages = 100
    def __str__(self): #自定义类描述
        return f"{self.title} by {self.author}"
    def __len__(self): #len()使用
        return self.pages
    def __del__(self):
        print("A book object has been deleted")

b = Book('Python3','Jose')
print(b) #Python3 by Jose
len(b) #100
del b #A book object has been deleted
```

### Errors and Exception

- Three keywords
    - `try`: block of code might lead to an error
    - `except`:block of code will be executed in case there is an error in `try` block
    - `finally`: A final block of code to be executed, regardless of an error

```python
def ask_for_int():
    while true:
        try:
            result = int(input("Please provide number: "))
        except:
            print("Whoops! this is not a number!)
            continue
        else:
            print(result)
            break
        finally:
            print("END")
```

- `except`可以捕获具体的错误类型

```python
try:
    f = open('testfile','w')
    f.write("Write a test line")
except TypeError: #捕获具体错误类型
    print("There was a type error!")
except OSError: #捕获具体错误类型
    print("You have an OS Error")
finally:
    print("End)

```

### Unit Test

- `pylint`: 静态语法检查
    - `pip install pylint`
    - `> pylint xx.py`

- `unittest`: built-in library，自带的单元测试库

```python

#cap.py - file to be tested
def cap_text(str):
    return str.capitalize()
def title_text(str):
    return str.title()

#test.py - Unit test file
import unitest
import cap #file name

class TestCap(unittest.TestCase):
    def test_one_word(self):
            text = 'python'
            result = cap.cap_text(text)
            self.assertEqual(result,'Python')
    def test_multiple_words(self):
            text = 'monty python'
            result = title_text(text)
            self.assertEqual(result,'Monty Python')
    
if __name == '__main__':
        unittest.main()
```

### Modules and Packages

- `PyPI` python的包管理系统，类型`RubyGem`,`NPM`
    - `pip3 install request`
- 文件引用
    
```python
### mymodule.py
def my_func():
    print("from my_module")

### other files
from mymodule import my_func
my_func()
```
- 文件夹(package)引用

```
└── package
├── __init__.py
├── main_script.py
└── subpackage
    ├── __init__.py
    └── sub_script.py
```
假设包结构如上，注意包内要包含`__init__.py`

```python
from package import main_script #引用包内文件
from package.subpackage import sub_script #引用包内文件

main_script.func_from_mainsript() #调用main_script的方法
sub_script.func_from_subsript() #调用sub_script的方法
```

- `__main__`

python没有`main`函数，当执行`python xx.py`时，在`xx.py`内有一个全局变量`__name__`被赋值为`"__main__"`表示这个文件是被直接运行的文件，也就是相当于`main`函数所在的文件。在程序里可以做如下判断:

```python
if __name == '__main__':
    #当被直接运行时，需要执行的代码
    some_func()
```

### Generator

惰性求值

```python
def gen_cube(n):
    for x in range(n):
        yield x**3
#返回generator object        
gen_cube(10) #<generator object gen_cube at 0x10567b150>
#pull value from gen_cube
list(gen_cube(10)) #

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

### Collections Module

类似C++的STL

- **Counter**

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

- **defaultdict**

更安全的dictionary，对于访问不存在的key，不会报错

```python
from collections import defaultdict
d = defaultdict(object)
d['one'] #访问一个不存在的key，返回一个<object object at 0x105424110>

#自定义默认value
d = defaultdict(lambda: 0) #对于不存在的key，返回value = 0
d['one'] #0
```
- **OrderedDict**

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

- **namedtuple**

可以用名字去索引的tuple

```python
from collections import namedtuple
Dog = namedtuple('Dog','age breed name')
sam = Dog(age=2, breed='Lab', name='Sammy')
sam.age
sam.breed
sam.name
```
### Datetime Module

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

### Python Debugger

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

### Regular Expression

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



## Resources for basic Practice
- [Intermediate Python](http://book.pythontips.com/en/latest/)
- [Basic Practice](http://codingbat.com/python)
- [More Mathematical (and Harder) Practice](https://projecteuler.net/archives)
- [List of Practice Problems](http://www.codeabbey.com/index/task_list)
- [Reddit](https://www.reddit.com/r/dailyprogrammer)
- [PythonChallenge](http://www.pythonchallenge.com/)