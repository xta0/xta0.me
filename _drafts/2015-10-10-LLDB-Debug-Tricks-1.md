---
layout: post
list_title: LLDB的一些调试技巧(一) | LLDB Debug Tricks Part 1
title: LLDB的一些调试技巧
categories: [iOS, LLDB]
updated: "2018-11-02"
---

在N年前刚从事iOS开发不久的时候整理过一份LLDB的常用命令，随着这几年使用的不断加深，加上LLDB自身的不断发展，使我对LLDB又有了些新的体会，也深感LLVM体系的强大，因此决定用接下来的两篇文章总结一下使用LLDB的一些心得和调试技巧

> 在展阅读下面内容之前，要先确保系统Rootless功能是Disable的，可在命令行中输入 `csrutil status` 查看结果，如果是`enabled.` 则需要对系统进行Rootless禁用，禁用方法可在Google中自行查阅

### Attaching LLDB to Any Process

我们可以使用LLDB attach到某个进程上，可以用进程号，也可以用进程的名称

```
lldb -n Xcode

pgrep -x Xcode #24834
lldb -p 24834
```
如果要等待挂起一个还未启动的进行，可以用下面命令

```
lldb -n Finder -w
#attach the Finder
pkill Finder
```

### 在LLDB中使用`ls`

我们经常希望在LLDB中可以执行`ls`命令，我们需要用 `-f` 参数告诉lldb来执行`/bin/ls`

```shell
> lldb -f /bin/ls
(lldb) process launch
```
`process launch`命令会列出当前目录下的文件，如果想要切换路径需要用`-w`或者`--`

```shell
(lldb) process launch -w /Applications
(lldb) process launch -- /Applications
```
如果是相对路径则要用`-X true`告诉LLDB将路径展开

```shell
(lldb) process launch -X true -- ~/Desktop
```

实际上上面命令都可以用`run`简化

```shell
(lldb) help run
#'run' is an abbreviation for 'process launch -X true --'
```

### `image lookup`

image lookup 用来检查symbol的位置

```shell
(lldb) image lookup -n "-[UIViewController viewDidLoad]"
# Address: UIKitCore[0x0000000000438886] (UIKitCore.__TEXT.__text + 4414950)
# Summary: UIKitCore`-[UIViewController viewDidLoad]
```
上述命令可以打印出`viewDidLoad`在哪个framework中，其中`-n`表示**完全匹配**搜索的函数或者symbol，我们也可以用`-rn` 后面接正则表达式来做模糊匹配

### Breakpoint

XCode中自带的符号断点非常强大和好用，我们也可以使用LLDB完成更高级的任务，实践中比较有效的debug方式是加基于Regex的符号断点

```shell
(lldb) rb '\-\[UIViewController\ ' 
(lldb) rb . -f DetailViewController.swift #给所有DetailViewController.swift中的方法打断点
(lldb) rb . #给每行代码都打断点
(lldb) rb . -s UIKitCore #给UIKitCore这个库打断点
(lldb) rb . -s UIKitCore -o 1 #one-shot 断点，只hit UIKitCore的第一个方法，执行完后断点自动delete
```
还有更复杂的case

```shell
(lldb) breakpoint set -n "-[UIViewController viewDidLoad]" -C "po $arg1" -G1
```
上述命令给所有的`-[UIViewController viewDidLoad]`都打上断点，当hit后执行(`-C`) `po $arg1`，`-G1`的意思是告诉LLDB命令执行完后继续向下执行

将断点信息保存在文件中

```shell
(lldb) breakpoint write -f /tmp/br.json
(lldb) platform shell cat /tmp/br.json
(lldb) breakpoint read -f /tmp/br.json
```
上述命令会将断点信息保存到`/tmp/br.json`文件中

### Expression

我们常用的`po`命令是`expression -o --`的简写，

### MISC

- env

`env`会列出当前lldb环境中所有可见的环境变量，我们可以

```shell
> lldb -f /bin/ls
(lldb)env
(lldb)process launch -v LSCOLORS=Ab -v CLICOLOR=1 -- /Applications/
```








