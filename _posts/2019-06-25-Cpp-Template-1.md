---
layout: post
list_title: C++ Template | Compile Time Programming
title: Compile Time Programming
categories: [C++]
---

## 模板的编译

当编译器遇到一个模板定义时，并不会立刻进行模板展开，只有当模板被使用时（被实例化出一个特定版本），编译器才会生成相应的代码。<mark>如果一个成员函数没有被调用，那么它将不会被实例化出来</mark>，也就不会进行语法检查，因此对于没有被使用的模板函数，即是它出现模板类型不匹配的情况，也不会被编译器发现，而拥有该成员函数的类模板仍然可以被正常的实例化。如下面例子

```cpp
//1. 实例化Blob<string>
Blob<string> articles = {"a","an","the"}; //2. 实例化构造函数
for(size_t i=0; i!= articles.size(); ++i){
	cout<<articles[i]<<endl; //3.实例化 Blob<int>::operator[](size_t)
}
```

### 模板编译期计算

关于模板编译另一个值得注意的是，模板在编译时是turing complete的，也就是说我们可以在编译器利用模板来做运算，比如下面的代码

```cpp
template <int n>
struct factorial {
    static const int value = n*factorial<n-1>::value;
};
template <>
struct factorial<0> {
    static const int value = 1;
};
int main(){
    cout << factorial<10>::value<< endl;
}
```
上述代码中，main函数调用了`factorial`函数模板，此时我们利用了模板参数`n`来实例化多个模板

```cpp
template<>
struct factorial<10> {};
template<>
struct factorial<9> {};
...
template<>
struct factorial<2> {};
```

### 模板的声明

由于模板在使用时才会进行实例化，相同的实例可能出现在多个对象文件中，当两个或多个文件使用相同的模板并提供了相同的参数时，每个文件就都会需要一个该模板的实例。因此在编译时，同一个模板在不同的compilation unit中会被实例化多次

> One annoying thing with using C++ templates is that they get instantiated multiple times if used with the same template arguments in different compilation units. The compiler happily instantiates the template in every compilation unit, only for the the linker to throw away all but one instances later.

```cpp
// t.h
#include <iostream>

template<class T>
T func(T x, T y){
    return x+y;
}

// file1.cpp
#include "t.h"
int func1(int x, int y){
    return func<int>(10,10);
}

// file2.cpp
#include "t.h"
int func2(int x, int y){
    return func<int>(10,10);
}
```
上面例子中，`file1`和`file2`都会实例化`int func<int>(int, int)`，并最终由linker来做去重。在C++11的新标准中，引入了`extern`关键字来避免重复实例化模板

```cpp
// file2.cpp

#include "t.h"
extern template int func(int, int);
int func2(int x, int y){
    return func<int>(x,y);
}
```
