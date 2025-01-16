---
list_title:  C++ Runtime Polymorphism
title: C++ Runtime Polymorphism
layout: post
categories: ["C++"]
---

长时间以来我对C++多态的理解紧紧停留在虚函数上，而虚函数只能通过继承来实现。实际上除了使用虚函数外，还有很多种方式可以实现多态，比如使用template meta-programing，使用interface等。

> 关于什么是虚表和虚函数的基本概念可以参考以前的文章[C++面向对象设计](https://xta0.me/2009/09/10/CPP-Basics-6.html)

### Template meta-programming

考虑下面这个例子

```cpp
class BaseFilter {
public:
    virtual void activate(char* pixel) const {
        cout<<"BaseFilter"<<endl;
    }
    virtual ~BaseFilter() {}
    char val;
};

class FilterBright: public BaseFilter {
public:
     void activate(char* pixel) const override {
         *pixel += val;
     }
     ~FilterBright() {}
};
```
上面例子中我们创建了一个基类和一个子类，基类提供了虚函数让子类来override

### Interface/Protocol

虽然Objective-C已经要渐渐淡出历史舞台了，但我