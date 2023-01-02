---
list_title: Swift Module Part 1
title:  Swift Module
layout: post
categories: ["iOS", "Swift"]
---

### LLVM Module

由于Swift中没有头文件，它中的API调用都是通过Module来完成。一个Module可以包含若干个Swift文件，它可以是静态库也可以是动态库。实际上Module的概念早在Swift之前就已经被应用在Objective-C中了
，在2013年[WWDC session 404](https://devstreaming-cdn.apple.com/videos/wwdc/2013/404xbx2xvp1eaaqonr8zokm/404/404.pdf)中，Module被首次引入到Objective-C中，并用来解决头文件的编译性能问题。在LLVM官方文档中有对Module的详细描述，我们做过多的赘述