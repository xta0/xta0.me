---
layout: post
list_title: LLDB的一些调试技巧(一) | LLDB Debug Tricks Part 1
title: LLDB的一些调试技巧
categories: [iOS, LLDB]
updated: "2018-11-02"
---

在N年前刚从事iOS开发不久的时候整理过一份LLDB的常用命令，随着这几年使用的不断加深，加上LLDB自身的不断发展，使我对LLDB又有了些新的体会，也深感LLVM体系的强大，因此决定用接下来的两篇文章总结一下使用LLDB的一些心得和调试技巧






> 在展阅读下面内容之前，要先确保系统Rootless功能是Disable的，可在命令行中输入 `csrutil status` 查看结果，如果是`enabled.` 则需要对系统进行Rootless禁用，禁用方法可在Google中自行查阅

### Attaching LLDB to Xcode





