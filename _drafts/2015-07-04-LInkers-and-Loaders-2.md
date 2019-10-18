---
layout: post
list_title: 链接与加载 | Linkers and Loaders | Static and Dynamic Libraries
title: 静态库与动态库
categories: [C,C++]
---

## 静态库

无论是动态库还是静态库都是为了解决代码重用的问题。静态库可以理解为一系列object files的集合，link的时候是静态链接，linker会将库中的object files拆出来和host app的object files一起链接，如下图所示

<img src="{{site.baseurl}}/assets/images/2015/07/static-linking.png">

### 制作一个简单的静态库

我们可以使用CMake简单制作一个静态库