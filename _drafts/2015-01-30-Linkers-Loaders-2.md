---
layout: post
title: Architechtural Issues
list_title:  链接与加载 | Linkers and Loaders | Part2 Architechtural Issues
categories: [C,C++]
---

在继续讨论Linker和Loader之前，我们有必要先复习一些体系结构和操作系统方面的知识，具体来说包括虚拟内存，ABI以及CPU的指令集，这些知识对于理解Linker和Loader的工作方式非常重要。

### 虚拟内存

### 指令集与ABI

指令集很好理解，就是目标体系结构所支持的汇编代码（机器码）的指令格式。我们还是先来写一段简单的代码

```cpp
int main()
{
  int a = 1;
  int b = 2;
  a = a + b;
}
```
然后我们在Linux上用`gcc`和`arm-linux-gnueabi-gcc`来编译上述代码并使用`objdump`观察各自的汇编指令和机器码:

```cpp
main.o:     file format elf64-x86-64
Disassembly of section .text:
0000000000000000 main:
int main()
{
   0:	55                   	push   rbp
   1:	48 89 e5             	mov    rbp,rsp
   4:	c7 45 f8 01 00 00 00 	mov    DWORD PTR [rbp-0x8],0x1
   b:	c7 45 fc 02 00 00 00 	mov    DWORD PTR [rbp-0x4],0x2
  12:	8b 45 fc             	mov    eax,DWORD PTR [rbp-0x4]
  15:	01 45 f8             	add    DWORD PTR [rbp-0x8],eax
  18:	b8 00 00 00 00       	mov    eax,0x0
}
  1d:	5d                   	pop    rbp
  1e:	c3                   	ret

main.o:	file format Mach-O 64-bit x86-64
Disassembly of section __TEXT,__text:
_main:
       0:	55 	pushq	%rbp
       1:	48 89 e5 	movq	%rsp, %rbp
       4:	31 c0 	xorl	%eax, %eax
       6:	c7 45 fc 01 00 00 00 	movl	$1, -4(%rbp)
       d:	c7 45 f8 02 00 00 00 	movl	$2, -8(%rbp)
      14:	8b 4d fc 	movl	-4(%rbp), %ecx
      17:	03 4d f8 	addl	-8(%rbp), %ecx
      1a:	89 4d fc 	movl	%ecx, -4(%rbp)
      1d:	5d 	popq	%rbp
      1e:	c3 	retq
```
可以看到在Linux上，目标文件的格式为`ELF`，在Mac上为`Mach-O`。左侧的一堆数字表示一条条机器码，右侧一系列的`push,mov,add,pop`等是对应的汇编代码。汇编代码和机器码是一一对应的。
