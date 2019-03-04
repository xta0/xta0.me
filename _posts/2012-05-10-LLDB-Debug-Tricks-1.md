---
layout: post
list_title: LLDB的一些调试技巧(一) | LLDB Debug Tricks Part 1
title: LLDB的一些调试技巧
categories: [iOS, LLDB]
updated: "2018-11-02"
---

在上一篇文章中我们曾简单的提到了LLDB的一些基本命令，在实际的iOS开发中，熟练掌握LLDB的各种命令对于提高调试效率有着非常重要的帮助，同时通过学习这些命令也可以帮我们加深对iOS系统的理解，从这篇文章开始，我将总结一些自己使用LLDB的一些心得体会，这些技巧从简单到复杂，涉及到iOS系统的方方面面，理解这些技巧背后的原理对提升iOS开发技术将会非常有帮助。

> 在展阅读下面内容之前，要先确保系统Rootless功能是Disable的，可在命令行中输入 `csrutil status` 查看结果，如果是`enabled.` 则需要对系统进行Rootless禁用，禁用方法可在Google中自行查阅

### Attaching LLDB to Xcode





