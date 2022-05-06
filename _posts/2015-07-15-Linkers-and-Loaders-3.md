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
例如libz.so.1.2.3.4的**soname**为**libz.so.1**。 在编译链接的时候，如果使用gcc或者clang，也需要准寻某些特定的convention，比如用`-L`表示库所在的路径，`-l`表示该路径下的动态库名称。如果将编译和链接的命令写在一起，则需要使用`-Wl,`告诉编译器后面的flag是linker的flag。一个好的习惯是同时使用`-L`和`-l`，尽量避免直接使用`-l`加绝对路径的方式

### Load-Time Relocation

动态链接要解决的问题是如何让代码在进程间共享。为了达到这个目的，我们需要将动态库中的内容加载到目标进程中去。这就涉及到对动态中代码地址进行重定位的问题。实际上在`fPIC`被发明之前，早期的动态库加载用的是所谓的Loader-Time Relocation (LTR)。这种方法需要loader根据目标进程来修改动态库中`.text`段指令的地址。显然，由于动态库中指令地址被修改了，因此，它只能被用于第一个加载它的进程，如果有其它进程想要使用，则需要再拷贝一份，并让loader再修改一遍指令地址以适应当前的目标进程。显然，这种方式效率很低，而且没有达到被各进程共享的目的。

> 如果要验证LTR，需要使用32bit的linux，64bit机器上生成的动态库默认是fpic格式

### Position Independent Code

在具体详细介绍`-fPIC`是如何工作的之前，我们先来想想如何让动态库的代码被多个process共用，显然，虚拟内存能帮我们做到这一点。我们可以将动态库加载到物理内存中，然后通过虚拟内存将其映射到不同的process中。注意，我们映射的只是text段，而对于data段，由于它是read-write的，我们需要为每个进程copy一份。如下图所示

<img src="{{site.baseurl}}/assets/images/2015/07/dynamic-linking-1.png">

由于text段可以被映射到不同进程的不动位置，因此，它里面的代码必须是地址无关的。接下来我们需要思考的问题是动态库中的符号是如何被引用的。这又包含两部分

1. 动态库中的外部symbol的调用
2. 动态库中的内部symbol的调用

第二个问题其实很好回答，内部的函数调用可以直接使用offset来定位，调用者不需要知道其在虚拟内存中绝对的地址。而回答第一个问题则比较复杂, 我们还是用一个例子说明，假设我们有两动态库`libx.so`和`liby.so`，其中`libx.so`的`foo`函数调用了`liby.so`中的`bar`函数。如下图所示

<img src="{{site.baseurl}}/assets/images/2015/07/dynamic-linking-2.png">

此时，对于进程#1来说，`libx.so`的`foo`函数被映射到`0x5000`的位置，`liby.so`的`bar`函数被映射到`0x3000`的位置，此时，如果让编译器生成`foo`函数的汇编指令，则是

```shell
call -0x2000(%rip) #bar
```
而对于进程#2来说，`libx.so`的`foo`函数被映射到`0x8000`的位置，`liby.so`的`bar`函数被映射到`0x1000`的位置，此时，如果让编译生成`foo`的汇编指令，则是

```shell
call -0x7000(%rip) #bar
```
显然，编译器是无法对`foo`生成不同汇编代码的，换一个角度说，这相当于让编译器生成地址相关的代码，这违背了fPIC的设计初衷。

如果想让编译器生成地址无关的代码，那么对于`bar`的调用应该是和具体地址无关的，也就是说基于`(%rip)`的偏移必须是一个固定值，这样编译器产生的`foo`的汇编指定才是确定的。推而广之，对于动态库中外部符号的引用必须是地址无关的。那么如何做到这一点呢？我们可以引入一个中间层，让动态库内对外部符号的调用都指向这个中间层，再由这个中间层来查找真实的符号地址，而动态库和这个中间层的offset是固定的，这就确保了编译器可以为符号调用生成确定的代码。这个中间层叫做**Global Offset Table**，简称**GOT**。我们接着看上面的例子

<img src="{{site.baseurl}}/assets/images/2015/07/dynamic-linking-3.png">

上图中，两个进程的地址空间都有各自的GOT表，此时，对于`libx.so`中对于`bar`的调用转成了对GOT中`bar`的寻址。而由于在linking期间，`libx.so`和`liby.so`相对于GOT的offset是可以被确定的（具体来说，对于进程#1和进程#2，相对于GOT中`bar`符号的offset为`-0x1000`），因此，编译器可以为`foo`生成确定的汇编代码。

```shell
call -0x1000(%rip) #bar
```
然后，进程#1 通过访问自己的GOT表，查到`bar`函数的地址是`0x3000`，它就能真正地调用到`bar`函数了。进程#2访问自己的GOT表，查到`bar`函数的地址是`0x1000`，它也能顺利地调用`bar`函数。这样我们就通过引入了 GOT 这个间接层，解决了 call 指令和 `bar` 函数定义之间的偏移不固定的问题。

了解GOT之后，我们回顾一下动态库的编译链接过程，为了生成地址无关代码，linker会为外部的符号建立基于GOT的间接跳转。到这一步linker的工作就完成了，当动态库被加载到进程空间时，loade需要根据offset的值来创建GOT符号表，然后对client binary中符号进行修正，使他们全部指向GOT，从client的角度看，进程的内存空间如下所示

<img src="{{site.baseurl}}/assets/images/2015/07/dynamic-linking-4.png">


### Demonstration of PIC Code

如果上面的描述太过抽象的话，接下来我们看通过实际的程序来演示`-fPIC`是如何工作的，我们用下面的例子

<div class="md-flex-h md-margin-bottom-24">
<div>
<pre class="highlight language-python md-no-padding-v md-height-full">
<code class="language-cpp">
// mylib.c

unsigned long mylib_int;
unsigned long dummy_var;
static int y = 3;
void set_mylib_int(unsigned long x){
    mylib_int = x+y;
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

extern 
void set_mylib_int(unsigned long x);
extern long get_mylib_int();
unsigned long glob = 5555;

int main() {
    set_mylib_int(100);
}
</code>
</pre>
</div>
</div>

我们先把`mylib.c`编译成动态库，并用`objdump -S`来看它反汇编的代码


```shell

> gcc mylib.c -fPIC -shared -fno-plt -o mylib.so
> objdump -S mylib.so

...
0000000000000609 \<set_mylib_int\>:
 609:	55                   	push   %rbp
 60a:	48 89 e5             	mov    %rsp,%rbp
 60d:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
 611:	8b 05 09 0a 20 00    	mov    0x200a09(%rip),%eax        # 201020 <y>
 617:	48 63 d0             	movslq %eax,%rdx
 61a:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
 61e:	48 01 c2             	add    %rax,%rdx
 621:	48 8b 05 b8 09 20 00 	mov    0x2009b8(%rip),%rax        # 200fe0 <mylib_int@@Base-0x50>
 628:	48 89 10             	mov    %rdx,(%rax)
 62b:	90                   	nop
 62c:	5d                   	pop    %rbp
 62d:	c3                   	retq

000000000000062e \<get_mylib_int\>:
 62e:	55                   	push   %rbp
 62f:	48 89 e5             	mov    %rsp,%rbp
 632:	48 8b 05 a7 09 20 00 	mov    0x2009a7(%rip),%rax        # 200fe0 <mylib_int@@Base-0x50>
 639:	48 8b 00             	mov    (%rax),%rax
 63c:	5d                   	pop    %rbp
 63d:	c3                   	retq
 ...
 ```

 分析汇编代码之前，我们先来看下`mylib.c`都有哪些符号






## Resources

- [Linkers and Loaders](https://www.amazon.com/Linkers-Kaufmann-Software-Engineering-Programming/dp/1558604960)
- [Advanced C and C++ Compiling](https://www.amazon.com/Advanced-C-Compiling-Milan-Stevanovic/dp/1430266678)
- [THE INSIDE STORY ON SHARED LIBRARIES AND DYNAMIC LOADING](https://cseweb.ucsd.edu/~gbournou/CSE131/the_inside_story_on_shared_libraries_and_dynamic_loading.pdf)