---
list_title: The secrets of Swift Runtime | Swift Runtime Part 1
title:  Swift Runtime Part 1
layout: post
categories: ["C++", "Objective-C", "C", "Assembly"]
---

> Notes of the book Secrets of Swift Runtime


### LLVM & the IR revelation

这个podcast提到了Chris最早写LLVM以及Swift最开始的motivation。LLVM的一个比较大的创新是发明了LLVM IR以及bitcode(A serialised for or LLVM IR)。这使得LLVM可以将编译器前端和后端进行解耦，任何编程语言都可以通过LLVM生成IR，从而不用关心具体machine code的生成。这个创新，也让Chris拿到了很软件界大奖。

### Compiler Phases

对于Swift来说，它的编译过程有下面五个步骤

1. Swift源码解析，将imported modules变成decalrations
2. 语义分析 (Semantic Anaysis)
3. 生成Swift的中间代码[SIL](https://apple-swift.readthedocs.io/en/latest/SIL.html)
4. SIL优化
5. 生成LLVM IR

## Resources

1. [Swift SIL](https://apple-swift.readthedocs.io/en/latest/SIL.html)
2. [Embrace Swift Type Inference](https://developer.apple.com/videos/play/wwdc2020/10165/)
