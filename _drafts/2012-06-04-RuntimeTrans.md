---
layout: post
title: Objective-C Runtime文档翻译
categories: 翻译
tags: iOS

---

##Type Encodings

为了支持runtime系统，编译器会将方法的入参和返回值encode成一个string。encode后的string可以通过`@encode()`查看。`@encode()`的类型可以为`int`，指针，结构体或共同体以及`class`，只要｀sizeof`合法的类型，都能被encode。



