---

title: Effective C++ Reading Note
layout: post
tag: C
categories: 随笔

---

<em></em>

###构造函数

```cpp

struct Widget{
	Widget(); //default constructor
	Widget(const Widget& t); //copy构造
	Widget&  operator=(const Widget& t); //copy assignment操作符

};

Widget w1; //default构造
Widget w2(w1); //copy构造
w1 = w2; //调用copy assignment操作符
Widget w1 = w2; //调用copy构造函数

```

###以const,enum,inline替换#define

- const和#define有什么不同？

在定义常量上，两者相同，但const有其优点：

  - cosnt有数据类型，宏常量没有数据类型，只做文本方式的替换，没有安全检查。
  - 不能对宏常量进行调试，const可以完全取代宏

对这一点，《Effective C++》的条款2，3做了细致的讲解，摘录如下：

   * 对于单纯常量，最好以const或enums替换#defines
   * 对于函数宏，最好改用inline替换#defines

如下例所示：


```c
 #define MIN(A,B)((A)<(B)? (A):(B))
```


```c

//将这个宏用模板类inline函数替代

template<typename T>
inline T callWithMin(const T& a, const T& b)
{
     return a>b ? b:a;
}

```


###尽可能的使用const


- 用来修饰变量：

```c
const char* p = "greeting";               //p指向内容不可修改
char const* p = "greeting";               //意义同上

char* const p = "greeting";               //p值不可修改

```

根据const和*的位置不同，const含义不同

- 用来修饰函数参数和返回值

const成员函数：提出的目的：条款20说提高效率的一个办法是用传引用代替传值，因为减少了变量复制的过程，因此需要成员函数提供带有const参数的函数。

例如:

```c
void print(const TestBook& ctb)
{
     std::cout<<ctb[0];
     ...
}
```

##04:确定对象被使用前已被初始化

要区别赋值和初始化：对类地成员变量进行赋值和初始化是不同的操作，老生常谈问题，构造时候使用初始化参数列表效率更高。

例如：

```c
class a
{
    a(): _ival(0),_ival2(10),_ival3(_ival){}                                  // 一个良好的习惯是为每个类提供一个默认的构造函数

     a(int _ival1):_ival(0),_ival2(10),_ival3(_ival1){}                    // 这种构构造初始化列表的形式高效,
     ~a(){}

private :
     const int _ival2;
     int _ival;

     int &_ival3;     

}
```

