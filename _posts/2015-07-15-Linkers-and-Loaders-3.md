---
layout: post
list_title: 链接与加载 | Linkers and Loaders | 动态链接与动态库 | Dynamic Library 
title: 动态链接与动态库
categories: [C,C++]
---

### 动态库

动态库概念的出现要远远晚于静态库，它是在现代多任务操作系统成熟只后才出现的。多任务的操作系统带来了资源的共享，比如PC上的键盘鼠标驱动程序，你的应用只需要调用这些内置驱动程序的API接口即可，而不需要将驱动程序也一并打入到你的二进制中。再比如很多移动操作系统中UIKit，提供了一套UI编程的框架供第三方App使用。这些被共享的代码都是以动态库的形式存在，并且随着操作系统的升级可以进行动态新。对比前面的静态库，这种方式显然更加的灵活。

我们可以将动态库简单的理解为一个没有main函数的binary。这说明动态库并不是像静态库那种目标文件的集合，而是一个实实在在的，具备某平台结构的binary。构建一个动态库也同样需要完整的编译和链接过程

```shell

> clang++ -shared -fPIC a.cpp -o a.so
> clang++ main.cpp -b.cpp -Wl,a.so
```

上面的第一条命令用来将`a.cpp`编译为一个动态库，在linux系统中，一般以`.so`结尾，第二条指令则用来链接该动态库到一个可执行文件中。

### fPIC的意义

动态链接要解决的问题是如何让代码在进程间共享。为了达到这个目的，一种容易想到的做法是将动态库中的内容直接加载到某个进程中去，这种方式也叫做Loader-Time Relocation (LTR)。显然，这需要loader根据目标进程来修改动态库中`.text`段指令的地址。这种方式的问题在于，由于动态库中指令地址被修改了，因此只能被用于第一个加载它的进程，如果有其它进程想要使用，则需要再拷贝一份，并让loader再修改一遍指令地址以适应当前的目标进程。显然，这种方式效率很低，而且没有达到被各进程共享的目的。

为了解决这个问题，fPIC被设计出来

### Naming Convention

在基于UNIX的系统中，动态库**文件**的表示方法为

```shell
lib + <library name> + .so + <library version information>
```
注意以**lib**开头的表示动态库文件，而不是动态库的名字，上面提到的版本信息可以表示为

```shell
<M>.<m>.<p>
```
其中`M`表示大版本，`m`表示小版本，`p`表示小改动。如果是**soname**表示，则格式为

```shell
lib + <libraray name> + .so + <Major Version>
```
例如libz.so.1.2.3.4的**soname**为**libz.so.1**。

说完了命名规则，在编译链接的时候，如果使用gcc或者clang，也需要准寻某些特定的convention，比如用`-L`表示库所在的路径，`-l`表示该路径下的动态库名称。如果将编译和链接的命令写在一起，则需要使用`-Wl,`告诉编译器后面的flag是linker的flag。一个好的习惯是同时使用`-L`和`-l`，尽量避免直接使用`-l`加绝对路径的方式

## Resources

- [Linkers and Loaders](https://www.amazon.com/Linkers-Kaufmann-Software-Engineering-Programming/dp/1558604960)
- [Advanced C and C++ Compiling](https://www.amazon.com/Advanced-C-Compiling-Milan-Stevanovic/dp/1430266678)