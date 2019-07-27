---
layout: post
title: 如何在MacOS上分析静态库的大小
list_title:  链接与加载 | Linkers and Loaders | Code size analysis on MacOS
categories: [C,C++]
---

最近在做cross-platform的编译工作，遇到了一个在macOS上是使用lipo和ar可以用来分析静态库的信息，

- 使用lipo切分不同架构的binaryNotic

```
lipo xxx.a -thin arm64 -output xxx_arm64.a
```


