---
layout: post
list_title: C++中的右值引用与std::move
title: C++中的右值引用与std::move
categories: [C++]
---

在C++11以前，一个右值可以被绑定到一个常量的左值引用，即`const T&`，当不能绑定到非常量引用`T&`。在C++11之后，提供了
