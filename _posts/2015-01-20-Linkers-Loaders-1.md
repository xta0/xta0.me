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
为了生成目标文件`.o`，我们可以使用gcc来进行编译，我们先来要编译`main.c`，得到`main.o`: `gcc -c main.c -o main.o`。该文件包含了很多重要的信息

1. 可以将`.o`理解为所有symbol的集合，可以用`nm`命令查看其包含的symbol

    ```shell
                     U _add_and_multiply
    0000000000000000 T _main
                     U _nCompletionStatus
    ```
    这些symbol的类型可以在文末的附录中查询。值得注意的是，`_add_and_multiply`的symbol类型为`U`，意味着这个符号的定义并不在`main.o`中，`_nCompletionStatus`同理。因此在单独编译`main.c`时，编译器并不知道这两个符号具体在哪里

2. 除了包含基本的符号信息外，`.o`还包含了section的信息，关于section将在后面跟linking的阶段做详细讨论。需要注意的是，`.o`并不包含符号在内存中的真实地址，地址绑定将在linking的阶段完成。但是每个section的长度和其address range却是`.o`中很重要的一条信息，linking需要依靠这个信息做符号的定位

尽管我们可以使用gcc为每个文件生成单独的目标文件，但是如何把这些目标文件组织到一起还有许多问题要解决，比如

1. 当`main.o`调用`add_and_multiply`函数时，程序该去哪里寻找这个函数呢？
2. `extern`声明的全局变量`_nCompletionStatus`在被使用时，该去哪里寻找呢?

因此链接器要做的事情就是将这些目标文件打包成一个可执行文件，这个打包的过程分几个步骤，我们先从Relocation说起。

### Relocation

Relocation的任务很简单，就是将所有目标文件的各个section按照规则进行合并，进而产生一个全新的memory map。这个过程虽然好理解，但有些细节需要注意，比如符号地址的重新分配。

由于虚拟内存的存在，对于每个`.o`文件中的符号，它们不需要考虑自己在真实的内存中的绝对地址（memory map中的地址），它们的地址都是相对的，比如`main.o`中，`_main`符号的地址为`0x0000000000000000`，`function.o`中`_add`地址也为`0x0000000000000000`。显然，在实际的内存中，不管是`_add`还是`_main`的地址是不可能为`0x0000000000000000`的。这时候就需要Relocation发挥作用，将所有symbol的地址重新分配到合理的位置。但需要注意的是，所有的这些重新分配都是基于section的，因此linker是需要知道目标文件中每个section的大小和范围的。

<img class="md-img-center" src="{{site.baseurl}}/assets/images/2009/05/c-compile-1.png">

### Resolving Reference

当我们把section都拼装好之后，接下来需要解决的问题就是symbol之间的调用问题，比如在main.o中`_add_and_multiply`符号类型`Undefined`，这说明main并不知道这个符号在哪，因此linker要做的事情就是将所有`Undefined`符号进行地址绑定（动态库中的符号暂不考虑），具体来说有下面几个步骤

1. 扫描memory map中已有的section
2. 找到调用外部符号的symbol (undefined symbol)
3. 找到被调用的symbol的地址，并将其绑定到目标symbol上

我们可以通过观察汇编代码来加深对上述过程的理解，我们可以使用`objdump`命令来反汇编目标文件

```shell
$objdump -D main.o

Disassembly of section __TEXT,__text:
_main:
       0:	55 	pushq	%rbp
       1:	48 89 e5 	movq	%rsp, %rbp
       4:	48 83 ec 10 	subq	$16, %rsp
       8:	f3 0f 10 05 44 00 00 00 	movss	68(%rip), %xmm0
      10:	f3 0f 10 0d 40 00 00 00 	movss	64(%rip), %xmm1
      18:	c7 45 fc 00 00 00 00 	movl	$0, -4(%rbp)
      1f:	f3 0f 11 4d f8 	movss	%xmm1, -8(%rbp)
      24:	f3 0f 11 45 f4 	movss	%xmm0, -12(%rbp)
      29:	f3 0f 10 45 f8 	movss	-8(%rbp), %xmm0
      2e:	f3 0f 10 4d f4 	movss	-12(%rbp), %xmm1
      33:	e8 00 00 00 00 	callq	0 <_main+0x38>
      38:	31 c0 	xorl	%eax, %eax
      3a:	48 8b 0d 00 00 00 00 	movq	(%rip), %rcx
      41:	f3 0f 11 45 f0 	movss	%xmm0, -16(%rbp)
      46:	c7 01 01 00 00 00 	movl	$1, (%rcx)
      4c:	48 83 c4 10 	addq	$16, %rsp
      50:	5d 	popq	%rbp
      51:	c3 	retq
```
上述是`main`函数的汇编指令，我们发现在第#33行，有一个`callq`的指令，参考`main`函数的源码可知，这个指令是用来调用`add_and_multiply`函数的，但是由于这个`.o`中`_add_and_multiply`符号未知，`callq`变成调用自己，显然是不正确的。因此，我们可以猜想在最终的binary中，这行指令应该会发生变化，linker会对符号进行解析和地址绑定。

```shell
$gcc function.c mian.c -o demoApp
$objdump -D demoApp

_main:
100000f50:	55 	pushq	%rbp
100000f51:	48 89 e5 	movq	%rsp, %rbp
100000f54:	48 83 ec 10 	subq	$16, %rsp
100000f58:	f3 0f 10 05 50 00 00 00 	movss	80(%rip), %xmm0
100000f60:	f3 0f 10 0d 4c 00 00 00 	movss	76(%rip), %xmm1
100000f68:	c7 45 fc 00 00 00 00 	movl	$0, -4(%rbp)
100000f6f:	f3 0f 11 4d f8 	movss	%xmm1, -8(%rbp)
100000f74:	f3 0f 11 45 f4 	movss	%xmm0, -12(%rbp)
100000f79:	f3 0f 10 45 f8 	movss	-8(%rbp), %xmm0
100000f7e:	f3 0f 10 4d f4 	movss	-12(%rbp), %xmm1
100000f83:	e8 78 ff ff ff 	callq	-136 <_add_and_multiply>
100000f88:	31 c0 	xorl	%eax, %eax
100000f8a:	48 8d 0d 6f 00 00 00 	leaq	111(%rip), %rcx
100000f91:	f3 0f 11 45 f0 	movss	%xmm0, -16(%rbp)
100000f96:	c7 01 01 00 00 00 	movl	$1, (%rcx)
100000f9c:	48 83 c4 10 	addq	$16, %rsp
100000fa0:	5d 	popq	%rbp
100000fa1:	c3 	retq
```
对比两段汇编代码，不难发现，`callq`指令后的符号变成了我们期望的函数符号`_add_and_multiply`。

### Loaders

现在我们已经有了一个link好的binary文件了，当我们执行它的时候，Loader会将binary中的section按照一定策略加载到内存中。实际上Loader的工作就这么简单，但是这里还是有一些点值得讨论，比如程序的Entry Point在哪里。

如果从C/C++程序的角度看，那么程序的入口应该就是main函数，而如果从loaders的角度看，则在main函数执行之前程序就已经start了。我们还是通过反汇编上面的`demoApp`来观察（注意，这里的`demoApp`是ELF格式）

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

到目前为止，我们已经对linkers和loaders有了一些直观的感觉，当然这些感觉还很粗浅，接下来我们会深入linkers和loaders的各个部分，来分析它们具体是怎么工作的。

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

