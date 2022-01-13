---
list_title: Code size matters | Reading the assembly
title: Reading the assembly
layout: post
categories: ["C++", "C", "Assembly"]
---

几年前写过[一篇关于ARM的汇编的文章](https://xta0.me/2013/06/15/ARM-Assembly.html)，那时候的iPhone还是32bit的，一转眼10年过去，现在的ARM基本早已经是64bit了，所对应的汇编指令也发生了一些变化。这篇文章并不会重复之前关于汇编的基本内容，而是会通过一些汇编知识来观察不同代码对code size的影响。

### Basics

所有ARM64的指令都是基于对寄存器的操作，其中每个寄存器大小为8 bytes，在ARM64的设备上，我们可以操作大约32个寄存器。显然，数据是无法都存在寄存器的，为了从内存中读写数据，需要用到所谓的[load和store指令](https://en.wikipedia.org/wiki/Load%E2%80%93store_architecture)，其中`ldr`指令将内存中的数据读入寄存器，`stp`


### Example #1







## Resources

- [ARM64指令集手册](https://developer.arm.com/documentation/100076/0100/a64-instruction-set-reference/a64-data-transfer-instructions)
- [Load-store Architecture](https://en.wikipedia.org/wiki/Load%E2%80%93store_architecture)


