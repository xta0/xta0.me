---
update: "2019-06-29"
layout: post
title: Overview
list_title:  链接与加载 | Linkers and Loaders | Part1 Overview
categories: [C,C++]
---

今天开始我们来讨论下C/C++程序的链接，加载和执行，我们将把重点放在目标文件的链接和加载上。

一般来说，源文件编译完成后会生成`.o`文件，多个`.o`文件和一些lib，一起link得到可执行文件，下面是GCC的一些常用编译选项，我们稍后会用到：

- `-c`: 用来生成`.o`文件
- `-o` : 用来创建目标文件
- `-g`: 编译器在输出文件中包含debug信息，产生dSYM符号表
- `-Wall`:编译器编译时打出warning信息,强烈推荐使用这个选项。
- `-I+dir`: 除了在main.c当前目录和系统默认目录中寻找.h外，还在dir目录寻找，注意，dir是一个绝对路径。
- `-01,-02,-03`: 编译器优化程度

> 如果使用macOS，gcc实际上是Clang的alias，binary的结构为MachO，而不是UNIX的ELF格式，不过这并不影响理解本文的内容

## Linkers

现在假设我们有三个文件:`function.h`,`function.m`和`main.c`，代码如下

```c
//function.h
#define FIRST_OPTION
#ifdef FIRST_OPTION
#define MULTIPLIER (3.0)
#else
#define MULTIPLIER (2.0)
#endif

float add_and_multiply(float x, float y);

//function.c
#include "function.h"

int nCompletionStatus = 0;
float add(float x, float y) {
    float z = x+y;
    return z;
}
float add_and_multiply(float x, float y){
    float z = add(x,y);
    z *= MULTIPLIER;
    return z;
}

//main.c
#include "function.h"

extern int nCompletionStatus;
int main() {
    float x = 1.0;
    float y = 1.5;
    float z;

    z = add_and_multiply(x,y);
    nCompletionStatus = 1;
    return 0;
}
```
为了生成目标文件`.o`，我们可以使用gcc来进行编译，我们先来要编译`main.c`，得到`main.o`: `gcc -c main.c -o main.o`。由于我们后面还会对目标文件做详细分析，现在可以将其简单理解为汇编指令和数据的集合，格式如下

<img src="{{site.baseurl}}/assets/images/2015/01/elf.png">

对于上面图中的每个section，可以认为它们都是一个有起始实地址和固定长度构成的一段连续空间。其中，有四个section需要特别关注

1. `.text`段，也叫代码段，用来存放汇编指令
2. `.data`段，也叫数据段，用来保存程序里设置好的初始化数据信息
3. `.rela.text`段，叫做重定位表(Relocation Table)。在表里，保存了所有未知跳转的指令
4. `.symtab`段，也叫做符号表，用来存放当前文件里定义的函数和对应的地址

接下来linker要做的事情就是将所有已经生成的`.o`文件link在一起，具体来说过程如下，linker会扫描所有的目标文件，将所有符号表中的信息收集起来，构成一个全局的符号表，然后再根据重定位表，把所有不确定的跳转指令根据符号表里地址，进行一次修正。最后，把所有的目标文件的section分别进行合并，行程最终的可执行代码。整个过程如下图所示

<img class="md-img-center" src="{{site.baseurl}}/assets/images/2015/01/linker.png">

如果将上述步骤分解开来，则linker的执行步骤分为两部分，分别是的Relocation和symbol的Resolution

### Relocation

Section和symbol的Relocation很简单，就是将所有目标文件的各个section按照某种规则进行合并。由于虚拟内存的存在，使目标文件不需要考虑自己在真实的内存中的绝对地址（memory map中的地址），它们的地址都是相对的，比如`main.o`中，`main`符号的地址为`0x0000000000000000`，`function.o`中`add`地址也为`0x0000000000000000`。显然，在实际的内存中，不管是`add`还是`main`的地址是不可能为`0x0000000000000000`的。这时候就需要Relocation发挥作用，将所有section的地址重新分配到合理的位置。但需要注意的是，在分配的过程中要保证每个Section的连续性不被破坏，因此linker是需要知道每个section的大小和范围。

<img class="md-img-center" src="{{site.baseurl}}/assets/images/2009/05/c-compile-1.png">

 Relocation完成后，每个section和symbol的虚拟内存地址也就确定了，接下来要做的事情就是修正每个section中指令地址，以及为undefined的symbol绑定地址。

### Symbol Resolution

我们可以先查看一下`main.o`中都有哪些symbol

```shell
                    U add_and_multiply
0000000000000000    T main
                    U nCompletionStatus
```
可以看到`add_and_multiply`的symbol类型为`U`，意味着编译器并不知道这个符号在哪里，`nCompletionStatus`同理。因此linker要做的事情就是将所有`Undefined`符号进行地址绑定（动态库中的符号暂不考虑），方法也很简单，就是扫描所有undefined的symbol，在symbol表中查找并将其绑定到真实的地址上。除了绑定Undefined Symbol外，linker还要负责修正每一条机器码中在内存中的真实地址。

下面我们通过观察汇编代码来加深对Relocation和Symbol Resolution的理解，我们可以使用`objdump`命令来反汇编目标文件

```shell
objdump -d -M intel -S main.o

main.o:     file format elf64-x86-64
Disassembly of section .text:
0000000000000000 main:
0:	55                   	push   rbp
1:	48 89 e5             	mov    rbp,rsp
4:	48 83 ec 20          	sub    rsp,0x20
8:	f3 0f 10 05 00 00 00 	movss  xmm0,DWORD PTR [rip+0x0]        # 10 <main+0x10>
f:	00
10:	f3 0f 11 45 f4       	movss  DWORD PTR [rbp-0xc],xmm0
15:	f3 0f 10 05 00 00 00 	movss  xmm0,DWORD PTR [rip+0x0]        # 1d <main+0x1d>
1c:	00
1d:	f3 0f 11 45 f8       	movss  DWORD PTR [rbp-0x8],xmm0
22:	f3 0f 10 45 f8       	movss  xmm0,DWORD PTR [rbp-0x8]
27:	8b 45 f4             	mov    eax,DWORD PTR [rbp-0xc]
2a:	0f 28 c8             	movaps xmm1,xmm0
2d:	89 45 ec             	mov    DWORD PTR [rbp-0x14],eax
30:	f3 0f 10 45 ec       	movss  xmm0,DWORD PTR [rbp-0x14]
35:	e8 00 00 00 00       	call   3a <main+0x3a>
3a:	66 0f 7e c0          	movd   eax,xmm0
3e:	89 45 fc             	mov    DWORD PTR [rbp-0x4],eax
41:	c7 05 00 00 00 00 01 	mov    DWORD PTR [rip+0x0],0x1        # 4b <main+0x4b>
48:	00 00 00
4b:	b8 00 00 00 00       	mov    eax,0x0
50:	c9                   	leave
51:	c3                   	ret
```
上述是`main`函数的汇编指令，我们发现在第#35行，有一个`call`的指令，参考`main`函数的源码可知，这个指令是用来调用`add_and_multiply`函数的，但是由于这个`.o`中`_add_and_multiply`符号未知，`call`变成调用自己，显然是不正确的。因此，我们可以猜想在最终的binary中，这行指令应该会发生变化，linker会对符号进行解析和地址绑定。

```shell
$gcc function.c mian.c -o demoApp
$objdump -D demoApp

0000000000400540 main:
  400540:	55                   	push   rbp
  400541:	48 89 e5             	mov    rbp,rsp
  400544:	48 83 ec 20          	sub    rsp,0x20
  400548:	f3 0f 10 05 d4 00 00 	movss  xmm0,DWORD PTR [rip+0xd4]        # 400624 <_IO_stdin_used+0x4>
  40054f:	00
  400550:	f3 0f 11 45 f4       	movss  DWORD PTR [rbp-0xc],xmm0
  400555:	f3 0f 10 05 cb 00 00 	movss  xmm0,DWORD PTR [rip+0xcb]        # 400628 <_IO_stdin_used+0x8>
  40055c:	00
  40055d:	f3 0f 11 45 f8       	movss  DWORD PTR [rbp-0x8],xmm0
  400562:	f3 0f 10 45 f8       	movss  xmm0,DWORD PTR [rbp-0x8]
  400567:	8b 45 f4             	mov    eax,DWORD PTR [rbp-0xc]
  40056a:	0f 28 c8             	movaps xmm1,xmm0
  40056d:	89 45 ec             	mov    DWORD PTR [rbp-0x14],eax
  400570:	f3 0f 10 45 ec       	movss  xmm0,DWORD PTR [rbp-0x14]
  400575:	e8 80 ff ff ff       	call   4004fa add_and_multiply
  40057a:	66 0f 7e c0          	movd   eax,xmm0
  40057e:	89 45 fc             	mov    DWORD PTR [rbp-0x4],eax
  400581:	c7 05 a9 0a 20 00 01 	mov    DWORD PTR [rip+0x200aa9],0x1        # 601034 <nCompletionStatus>
  400588:	00 00 00
  40058b:	b8 00 00 00 00       	mov    eax,0x0
  400590:	c9                   	leave
  400591:	c3                   	ret
  400592:	66 2e 0f 1f 84 00 00 	nop    WORD PTR cs:[rax+rax*1+0x0]
  400599:	00 00 00
  40059c:	0f 1f 40 00          	nop    DWORD PTR [rax+0x0]
```
对比两段汇编代码，不难发现，前者的BaseAddress是`0000000000000000`，每条汇编指令前面的数字是该指令的offset。而后者的BaseAddreess变成了`0000000000400540`，说明Linker对每个目标文件的汇编指令做了地址修正，另外，`call`指令后的符号变成了我们期望的函数符号`add_and_multiply`，地址为`4004fa`。说明Linker完成的符号和地址的绑定。

```shell
00000000004004fa <add_and_multiply>:
4004fa:	55                   	push   rbp
4004fb:	48 89 e5             	mov    rbp,rsp
...
40051a:	f3 0f 10 45 e4       	movss  xmm0,DWORD PTR [rbp-0x1c]
40051f:	e8 b2 ff ff ff       	call   4004d6 <add>
400524:	66 0f 7e c0          	movd   eax,xmm0
...
40053e:	c9                   	leave
40053f:	c3                   	ret
```

## Loaders

现在我们已经有了一个link好的binary文件了，当我们执行它的时候，Loader会将binary中的section按照一定规则加载到内存中。实际上Loader的工作就这么简单，但是所谓的“一定规则”确又很复杂，具体来说Loader面临的挑战主要是如何为每个可执行文件找到一片连续的内存空间。解决这个问题有两个办法，一个是使用内存分段，一个是使用内存分页

### Segmentation

内存分段的思路很简单，就是找出一段连续的物理内存，然后和虚拟内存进行映射得到虚拟内存地址，再将这个地址分给被load的binary。这种思路虽然简单，但有一个很大的问题就是内存碎片。如下图所示，当C程序退出后，释放的128MB内存成了不连续的空间，当有loader要加载D程序时，会发现没有足够的连续内存空间可以使用。

<img class="md-img-center" src="{{site.baseurl}}/assets/images/2015/01/linker-1.png">

解决这个问题，可以使用内存的置换技术，即将程序B先写入到硬盘，释放内存空间来装在D，当D装载完成后，再将B读入到内存中。但是由于内存置换的成本极高，硬盘访问的速度要比内存慢太多，因此这种方案有很大的性能问题。

### Paging

为了解决内存碎片和交换问题，现代操作系统采用了内存分页的技术。它将物理内存和虚拟内存均切分成4KB大小的一页。由于数据量小，这种切分带来的好处在于内存置换的成本变低了。

```shell
getconf PAGE_SIZE #4096
```

进一步讲，当程序加载时，loader不需要将程序一次性加载到物理内存中，而是可以按需加载，当需要用到虚拟内存里的指令和数据时，操作系统触发一个CPU的确页错误，然后触发加载虚拟内存中的页到物理内存。

这并不是说一段连续完整程序可以被零散的映射物理内存中，程序的寻址空间在内存中依旧是一段连续的区域，只不过当出现内存碎片或者物理内存不足时，系统进行内存置换的成本变小了(因为置换现在是以4KB为单位，速度变快了)。如下图中，当程序A试图加载第三片虚拟内存到物理内存中时，会触发系统的内存置换

<img class="md-img-center" src="{{site.baseurl}}/assets/images/2015/01/linker-2.png">

总的来说loader可以将程序和物理内存隔离开，然程序不需要考虑实际的物理内存地址，大小和分配方案，大大降低了程序编写复杂度。

### Excutable

当loader将程序完全加载到内存中后，程序就可以运行了。如果从C/C++程序的角度看，那么程序的入口应该就是main函数，而如果从loaders的角度看，则在main函数执行之前程序就已经start了。我们还是通过反汇编上面的`demoApp`来观察

> 注意，这里的`demoApp`依旧是ELF格式

```shell
Disassembly of section .text:

00000000004003e0 <_start>:
  4003e0:	31 ed                	xor    %ebp,%ebp
  4003e2:	49 89 d1             	mov    %rdx,%r9
  4003e5:	5e                   	pop    %rsi
  4003e6:	48 89 e2             	mov    %rsp,%rdx
  4003e9:	48 83 e4 f0          	and    $0xfffffffffffffff0,%rsp
  4003ed:	50                   	push   %rax
  4003ee:	54                   	push   %rsp
  4003ef:	49 c7 c0 10 06 40 00 	mov    $0x400610,%r8
  4003f6:	48 c7 c1 a0 05 40 00 	mov    $0x4005a0,%rcx
  4003fd:	48 c7 c7 40 05 40 00 	mov    $0x400540,%rdi
  400404:	e8 b7 ff ff ff       	callq  4003c0 <__libc_start_main@plt>
  400409:	f4                   	hlt
  40040a:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)
```
我们看到代码段(text section)的第一个函数是`_start`，这个函数最后又调用了`__libc_start_main`函数。我们先看看这个函数的原型

```c
int __libc_start_main(
    int *(main) (int, char * *, char * *),  /* address of main function */
    int argc, 
    char * * ubp_av, 
    void (*init) (void), /* address of init function */
    void (*fini) (void), 
    void (*rtld_fini) (void), 
    void (* stack_end));
```
这个函数的定义在libc中，由于篇幅有限，源码就不展开分析了，但是我们可以大致猜到它的作用，即初始化程序进程中的环境变量，为main函数的执行做准备，比如某个类的构造函数使用了`__attribute__((constructor))`这种keyword，那么在main函数执行前，自定义的初始化逻辑将被执行。

### 小结

到目前为止，我们已经对linkers和loaders有了一些直观的感觉，当然这些感觉还很表面，很多细节问题还不是很清楚，比如Symbol Resolution的具体过程是怎样的，Relocation时偏移地址是如何计算的，动态库是如何加载的，等等。我们将在后面逐步讨论这些问题

## Resources

- [Linkers and Loaders]()
- [C and C++ compiling]()

### `nm`命令

> 注意，`nm`命令不会列出DLL的entry point，除非有和它关联的符号表。

```
- A :absolute symbol, global
- a :absolute symbol, local
- B :uninitialized data (bss), global
- b :uninitialized data (bss), local
- D :initialized data, global
- d :initialized data, local
- F :file name
- l :line number entry (see -a option)
- N :no defined type, global; this is an unspecified type, compared to the undefined type U
- n :no defined type, local; this is an unspecified type, compared to the undefined type U
- S :section symbol, global
- s :section symbol, local
- T :text symbol, global
- t :text symbol, local (static)
- U :undefined symbol
- ? :unknown symbol
```

