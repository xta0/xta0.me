---
update: "2019-11-30"
list_title: 链接与加载 | Linkers and Loaders | LLVM与IR (2) | IR with LLVM Part 2
title: LLVM中与IR
layout: post
mathjax: true
categories: ["LLVM", "Linker","Compiler"]
---

上一篇文章中，我们了解了IR的语法和LLVM生成IR的一些命令。这篇文章中我们将探究一下IR是如何被生成的，我们将使用LLVM的C++ API来生成一段IR

### LLVM中IR的对象模型

LLVM是如何生成IR的呢？显然不可能靠字符串拼接，因此在内存中，LLVM必然需要对IR进行建模。我们可以在LLVM的源码中找到IR相关的API - [include/llvm/IR](https://github.com/llvm/llvm-project/tree/master/llvm/include/llvm/IR)。这些类中，我们需要重点关注下面几个

- Module



