---
title: The Vizzle Framework
list_title: The Vizzle Framework
layout: post
categories: [Architect]
---

大概2，3年前，大家都在用Facebook的`Three20`，那个时候我们也在用，后来我发现它实在太繁琐，于是我将`TTTModelViewController`和`TTTTableViewController`进行了改写，抽象了很多API，比如列表的下拉刷新，自动翻页，数据绑定等，与此同时，又抽象了model的请求和回调，使ViewController变的很轻量。改写完成后基本变成了一种模板+填空的开发模式，使用者只需要override一些简单API就可快速完成业务的开发，这在当时显著的提高了开发效率。

在后来的一段时间里，我不断增加一些新的feature，并放到项目中实践：

- 增加网络适配层，可适配第三方网络请求库(比如`AFNetworking`)
- 增加了Model的单元测试模板
- 增加了Logic对象用来分离ViewController中UI无关的代码，并能将其用做单元测试

经过了一年多的项目打磨，它现在已经成为一套稳定的，敏捷的，适用于WebService的App框架。于是我将它整理并发布到Github上，取名[Vizzle](https://github.com/Vizzle/Vizzle)。

在整理Vizzle同时，我也在大量的使用Ruby on Rails，同样以敏捷著称的web框架，它的Scaffold深深的启发了我，于是我为Vizzle写了一个类似于Scaffold的代码生成工具，取名:[VZScaffold](https://github.com/xta0/Scaffold)。

VZScaffold + Vizzle = 敏捷业务开发
