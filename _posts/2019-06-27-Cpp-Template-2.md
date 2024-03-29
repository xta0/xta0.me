---
layout: post
list_title: C++ Template | MISC
title: MISC
categories: [C++]
---

### 使用别名

我们可以使用`typedef`来给一个模板类定义别名

```cpp
typedef Blob<string> StrBlob
```
<mark>在C++11中</mark>, 我们也可以为类模板定义别名

```cpp
template<typename T> using twin = pair<T,T>;
twin<string> authors; //authors的类型是pair<string, string>
```

### `typename`的作用

`typename`用来告诉模板如何解析`T::foo`此类的代码。上一节可知类模板可以有静态成员，那么`T`可以指代一个模板类，即存在`Foo<int>::foo`的代码，其中`foo`为一个静态成员。但是同样的代码也可以指代某个类的类型成员，比如我们前面Blob类中的`Blob<T>::size_type`。因此编译器无法知道`T::foo`表示的到底是哪一种情况，此时需要`typename`来显式的告诉编译器`foo`是一个类型而非变量。

```cpp
template<typename T>
struct Obj {
    using type = T;
};
template <typename T>
void f() {
    typename Obj<T>::type var;
}
```
 上面的例子可以很直观的看出这一点，当编译器在编译 `typename Obj<T>::type var;`时，会将`Obj<T>::type`认为是某种类型。如果不加`typaname`，编译器则会将`Obj<T>::type var`理解为访问`Obj<T>`的静态成员，从而报错。

 ### Dot Template



 

 

## Useful C++ Resources

- [Which book/course is recommended to learn c++ in depth and clear concepts?](https://www.quora.com/Which-book-course-is-recommended-to-learn-c++-in-depth-and-clear-concepts)
