---
layout: post
list_title: 链接与加载 | Linkers and Loaders | 动态链接与动态库 | Dynamic Library 
title: 动态链接与动态库
categories: [C,C++]
---

### 动态库

动态库概念的出现要远远晚于静态库，它是在现代多任务操作系统成熟只后才出现的。多任务的操作系统带来了资源的共享，比如PC上的键盘鼠标驱动程序，你的应用只需要调用这些内置驱动程序的API接口即可，而不需要将驱动程序也一并打入到你的二进制中。此外，使用动态库可以减少link的时间，相比于静态库每次修改都需要都需重新编译来建立地址绑定，动态库可以做到只编译自己，从而减少链接时间。

我们可以将动态库简单的理解为一个没有main函数的binary。这说明动态库并不是像静态库那种目标文件的集合，构建一个动态库也同样需要完整的编译和链接过程

```shell
> clang++ -shared -fPIC a.cpp -o a.so
> clang++ main.cpp -b.cpp -Wl,a.so
```

上面的第一条命令用来将`a.cpp`编译为一个动态库，在linux系统中，一般以`.so`结尾，第二条指令则用来链接该动态库到一个可执行文件中。这里面最重要的参数是`-fPIC`，它可以让编译器生成地址无关的代码，这是动态库可以被多个应用程序共享的前提。我们后面会详细介绍其工作原理。

### Load-Time Relocation

动态链接要解决的问题是如何让代码在进程间共享。为了达到这个目的，我们需要将动态库中的内容加载到目标进程中去。这就涉及到对动态中代码地址进行重定位的问题。实际上在`fPIC`被发明之前，早期的动态库加载用的是所谓的Loader-Time Relocation (LTR)。这种方法需要loader根据目标进程来修改动态库中`.text`段指令的地址。显然，由于动态库中指令地址被修改了，因此，它只能被用于第一个加载它的进程，如果有其它进程想要使用，则需要再拷贝一份，并让loader再修改一遍指令地址以适应当前的目标进程。显然，这种方式效率很低，而且没有达到被各进程共享的目的。

> 如果要验证LTR，需要使用32bit的linux，64bit机器上生成的动态库默认是fpic格式

### `-fPIC`

在具体详细介绍`-fPIC`是如何工作的之前，我们先来想想如何让动态库的代码被多个process共用，显然，虚拟内存能帮我们做到这一点。我们可以将动态库加载到物理内存中，然后通过虚拟内存将其映射到不同的process中。注意，我们映射的只是text段，而对于data段，由于它是read-write的，我们需要为每个进程copy一份。如下图所示

<img src="{{site.baseurl}}/assets/images/2015/07/dynamic-linking-1.png">

由于text段可以被映射到不同进程的不动位置，因此，它里面的代码必须是地址无关的。接下来我们需要思考的问题是动态库中的符号是如何被引用的。这又包含两部分

1. 动态库中的外部symbol的调用
2. 动态库中的内部symbol的调用

第二个问题其实很好回答，内部的函数调用可以直接使用offset来定位，调用者不需要知道其在虚拟内存中绝对的地址。而回答第一个问题则比较复杂。




接下来我们看`fPIC`到底做了什么，我们用下面的例子

<div class="md-flex-h md-margin-bottom-24">
<div>
<pre class="highlight language-python md-no-padding-v md-height-full">
<code class="language-cpp">
// mylib.c

unsigned long mylib_int;
unsigned long dummy_var;

void set_mylib_int(unsigned long x){
    mylib_int = x;
}

unsigned long get_mylib_int() {
    return mylib_int;
}
</code>
</pre>
</div>
<div class="md-margin-left-12">
<pre class="highlight md-no-padding-v md-height-full">
<code class="language-cpp">
// main.c

extern void set_mylib_int(unsigned long x);
extern long get_mylib_int();

unsigned long glob = 5555;

int main() {
    set_mylib_int(100);
    printf("value set in mylib is %ld", get_mylib_int());
    printf("value set in glob is %ld", glob);

}
</code>
</pre>
</div>
</div>

我们先把`mylib.c`编译成动态库，并用`objdump -d`来看它反汇编的代码

```shell
...
0000000000000609 set_mylib_int:
 609:	55                   	push   %rbp
 60a:	48 89 e5             	mov    %rsp,%rbp
 60d:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
 611:	48 8b 05 c8 09 20 00 	mov    0x2009c8(%rip),%rax        # 200fe0 <mylib_int@@Base-0x48>
 618:	48 8b 55 f8          	mov    -0x8(%rbp),%rdx
 61c:	48 89 10             	mov    %rdx,(%rax)
 61f:	90                   	nop
 620:	5d                   	pop    %rbp
 621:	c3                   	retq

0000000000000622 get_mylib_int:
 622:	55                   	push   %rbp
 623:	48 89 e5             	mov    %rsp,%rbp
 626:	48 8b 05 b3 09 20 00 	mov    0x2009b3(%rip),%rax        # 200fe0 <mylib_int@@Base-0x48>
 62d:	48 8b 00             	mov    (%rax),%rax
 630:	5d                   	pop    %rbp
 631:	c3                   	retq
 ...
 ```

### 地址解析问题

当一个动态库在编译完成后，linker实际上已经对其内部的符号建立好了地址绑定，我们可以将每条指令的地址理解为相对于基地址的偏移。当某个进程加载动态库时，loader需要将动态库中的代码嵌入到该进程中，这样该进程才能访问库中的symbol，例如目标进程要改变动态库中某个变量的值

```shell
mov eax, ds:0xBFD10000 ;load the variable from address 0xBFD10000 to register eax
add eax, 1             ;increment the load value
mov ds:0xBFD10000, eax ;store the result back to the address 0xBFD10000
```
此时我们需要知道该变量在内存中的绝对地址，而不是它在库中的相对地址。再比如跳转指令

```shell
call 0x0A120034 ;calling function whose entry point is 0x0A120034
```
如果该地址位于动态库中，则我们需要在动态库加载时确定好这个地址在内存中的位置。

实际上我们并不需要对动态库中所有的指令或者symbol进行地址修改，对于库中内部的函数或者`static`定义的变量我们可以使用相对offset进行寻址。我们只需要对于export出来的接口或者变量进行地址解析即可。

但此时问题来了，我们该如何计算修改地址？正如前面所说，无论是动态库还是使用它的binary，在编译时，linker已经完成了对各自符号地址的计算，因此依靠linker是不可能了。此时我们需要借助loader来帮我们完成任务，这种特殊的loader也叫做**dynamic loader**。

### `.rel.dyn`

实际上在我们编译bianry时，如果linker发现某些symbol需要在load的时候确定，linker会给这些symbol的地址填充一个临时值(通常是0）





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