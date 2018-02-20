---
title: python3
layout: post
---

## Python3 Cheatsheet

### py2 vs py3

### Variable and Printing

- **print**
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

### String

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

### list

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

### dictionary

- 无序map

```python
d={'k1':123, 'k2':[0,1,2], 'k3':{'insidekey':100}}
d['k2'] #[0,1,2]
d['k4']="abc"
keys = d.keys() 
```