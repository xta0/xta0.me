---
list_title: C++ Template Programming
title: C++ Template Programming
layout: post
mathjax: true
categories: ["AI", "Machine Learning","Deep Learning"]
---

> Notes of <C++ Templates - the complete guide>

## Template Argument Deduction

The process of replacing template parameters by concrete types is called **instantiation**. It results in an **instance** of a template. An attempt to instantiate a template for a type that doesn't support all the operators used within it will result in compile-time error. For example

```
std::complex<float> c1, c2; //doesn't provide < operator

::max(c1, c2) //Error at compile time
```

Thus, templates are compiled in two phases:

1. Without instantiation at definition time, the template code itself is checked for correctness ignoring the template parameters. This includes
    - syntax error are discovered, such as missing semicolons
    - Using unknown names that don't depend on template parameters are discovered
    - Static assertions that don't depend on template parameters are discovered
2. At instantiation time, the template code is checked (again) to ensure that all code is valid. That is, now especially, all parts that depend on template parameters are double-checked

For example:

```
template<typename T>
void foo(T t){
    undeclared(); //first-phase compile-time error
    undeclared(T); //second-phase compile-time error
}

### Compiling and Linking

Two-phase translation leads to an important problem in the handling of templates in practice: When a function template is used in a way that triggers its instantiation, a compiler will need to see the template's definition. This breaks the usual compile and link distinction for ordinary functions, when the declarations of a function is sufficient to compile its use. Methods of handling this problem will be discussed later. For the moment, let's take the simplest approach: ***Implement each template inside a header file*.








```

