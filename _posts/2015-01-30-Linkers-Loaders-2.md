---
layout: post
title: Architechtural Issues
list_title:  链接与加载 | Linkers and Loaders | Part2 Architechtural Issues
categories: [C,C++]
---

在介绍Linker具体的工作方式之前，我们有必要先复习下计算机体系结构方面的知识，当然我们只会讨论和Linker相关的内容，具体来说是程序的寻址方式以及指令集格式，这两部分对于Linker来说非常重要。因为只有明确了寻址方式Linker才能正确的计算偏移量来定位符号，同时Linker也需要了解目标体系结构下的指令集，这样才能正确的做指令修改等等。

### 指令集与ABI

指令集很好理解，就是目标体系结构所支持的opcode，我们还是先来写一段简单的代码

```cpp
int main()
{
  int a = 1;
  int b = 2;
  a = a + b;
}
```
然后我们分别使用`gcc`在Linux上编译和`clang`在MacOS上来编译上述代码，观察对应的汇编指令和机器码:

<div class="md-flex-h md-margin-bottom-24">
<div>
<pre class="highlight md-no-padding-v md-height-full">
<code class="language-cpp">

main.o:     file format elf64-x86-64
Disassembly of section .text:
0000000000000000 <main>:
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
</code>
</pre>
</div>
<div class="md-margin-left-12">
<pre class="highlight md-no-padding-v md-height-full">
<code class="language-cpp">
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
</code>
</pre>
</div>
</div>

可以看到在Linux上，目标文件的格式为`ELF`，在Mac上为`Mach-O`。左侧的一堆数字表示一条条机器码，右侧一系列的`push,mov,add,pop`等是对应的汇编代码。汇编代码和机器码是一一对应的
