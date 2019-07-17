---
layout: post
title: Architechtural Issues
list_title:  链接与加载 | Linkers and Loaders | Part2 Architechtural Issues
categories: [C,C++]
---

在介绍Linker具体的工作方式之前，我们有必要先复习下计算机体系结构方面的知识，当然我们只会讨论和Linker相关的内容，具体来说是程序的寻址方式以及指令集格式，这两部分对于Linker来说非常重要。因为只有明确了寻址方式Linker才能正确的计算偏移量来定位符号，同时Linker也需要了解目标体系结构下的指令集，这样才能正确的修改跳转指令等。

