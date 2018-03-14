---
title: python3
layout: post
---

## Python3 Cheatsheet

## Resources for basic Practice

- [Basic Practice](http://codingbat.com/python)
- [More Mathematical (and Harder) Practice](https://projecteuler.net/archives)
- [List of Practice Problems](http://www.codeabbey.com/index/task_list)
- [Reddit](https://www.reddit.com/r/dailyprogrammer)
- [PythonChallenge](http://www.pythonchallenge.com/)

### py2 vs py3


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


### Datastructures

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
- **list**
    - 创建list

    ```python
    mylist=[1,2,'three']
    mylist[2] #three
    len(mylist) #长度
    mylist[1] = 10
    ```
    - 追加元素

    ```python
    list=['one','two','three']
    list.append('four') #['one', 'two', 'three', 'four']
    ```

    - 删除元素

    ```python
    val = list.pop() #four 默认删除尾部
    val = list.pop(-1) #删除尾部
    val = list.pop(2) # three
    ```

    - 拼接list

    ```python
    list1=[1,2,3]
    list2=[4,5]
    list = list1 + list2 #[1, 2, 3, 4, 5]
    ```
    - `max` , `min`

    ```python
    list=[1,2,3]
    max_num = max(list)
    min_num = min(list)
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
    myset.add(1)
    myset.add(2)
    mylist = [1,2,1,2]
    newset = set(mylist) #{1,2}
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

### Statements

- **And,Or,Not**

```python
1<2 and 2<3
2<3>10 #false, 等价于2<3 and 3>10
100==1 or 2==2
not 1==1
```
- **if elif else **

```
if some_condition: #注意冒号
    #execute some code 
elif some_other_condition:
    #do something different
else:
    # do something else
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

- Decorators
    - 在不修改原函数的前提下，对已有函数进行扩展
    - 在需要扩展的函数上面，使用`@`符号标记

    ```python
    def some_decorator(some_func)
        #some code
        some_func()
        #some code 
        return some_func            
        
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
    
    new_func = new_decorator(fun_needs_decorator)
    new_func()

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

- 使用函数对象

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

