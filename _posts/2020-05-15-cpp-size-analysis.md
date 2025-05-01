---
list_title:  C++ Performance | Code Size Analysis
title: Code Size Analysis
layout: post
categories: ["C++", "C", "Assembly"]
---

几年前写过[一篇关于ARM的汇编的文章](https://xta0.me/2013/06/15/ARM-Assembly.html)，那时候的iPhone还是32bit的，一转眼10年过去，现在的ARM基本早已经是64bit了，所对应的汇编指令也发生了一些变化。这篇文章并不会重复之前关于汇编的基本内容，而是会通过一些汇编知识来观察不同代码对code size的影响。

### Register Basics

1. 所有ARM64的指令都是基于对寄存器的操作，其中每个寄存器大小为8 bytes，在ARM64的设备上，我们可以操作大约32个寄存器。
2. 显然，数据是无法都存在寄存器的，为了从内存中读写数据，需要用到所谓的[load和store指令](https://en.wikipedia.org/wiki/Load%E2%80%93store_architecture)，其中`ldr`指令将内存中的数据读入寄存器，`str`将寄存器中的数据写入内存，详细[参考这里](https://developer.arm.com/documentation/dui0552/a/the-cortex-m3-instruction-set/memory-access-instructions/ldr-and-str--register-offset#:~:text=LDR%20instructions%20load%20a%20register,to%203%20bits%20using%20LSL%20.)。

### A Simple Exercise

下面这两个函数很简单，做的事情也相同，但第二个函数会产生较多的代码

```cpp
#import <vector>

int useIndex(const std::vector<int>& v) {
    return v[1];
}

int useAt(const std::vector<int>& v) {
    return v.at(1);
}
```
用`xcrun --sdk iphoneos clang -arch arm64 -c -Oz`进行编译，并用`symbols`来查看这两个函数的大小

```shell
test.o [arm64, 0.006064 seconds]:
    null-uuid                            /Users/taox/Projects/CodeBase/cpp/perf/test.o [OBJECT, FaultedFromDisk]  
        0x0000000000000000 (    0xb8)  SEGMENT
            0x0000000000000000 (    0x50) __TEXT __text
                0x0000000000000000 (     0xc) useIndex(std::__1::vector<int, std::__1::allocator<int> > const&) [FUNC, EXT, NameNList, ...
                0x000000000000000c (    0x1c) useAt(std::__1::vector<int, std::__1::allocator<int> > const&) [FUNC, EXT, NameNList, Man...
                0x0000000000000028 (    0x28) std::__1::vector<int, std::__1::allocator<int> >::at(unsigned long) const [FUNC, EXT, Nam...
            0x0000000000000050 (     0x8) __DATA __objc_imageinfo
            0x0000000000000058 (    0x60) __LD __compact_unwind
```
可以看到 

- `useIndex`的大小为`0xc`(12字节)
- `useAt`的大小为`0x1c`(28字节)
- `vector::at`的大小为`0x28` (40字节)

相加正好是80字节(0x50)，和`__TEXT`段大小一致，再来对比一下汇编代码

<div class="md-flex-h md-margin-bottom-24">
<div>
<pre class="highlight language-python md-no-padding-v md-height-full">
<code class="language-shell">
useIndex:
; %bb.0:
	ldr	x8, [x0]
	ldr	w0, [x8, #4]
	ret
</code>
</pre>
</div>
<div class="md-margin-left-12">
<pre class="highlight md-no-padding-v md-height-full">
<code class="language-shell">
useAt:
; %bb.0:
	mov	w1, #1
	bl	std::vector::at
	ldr	w0, [x0]
	ret
</code>
</pre>
</div>
</div>

`useIndex`的汇编代码比较简洁，这里编译器应该是inline了某些代码，因为`v[i]`是一个C++函数，这里应该直接inline了。注意`w0`相当于`x0`的lower 4 bytes，`#4`是offse表示skip 4 bytes。`ret`返回的结果通常保存在寄存器`x[0]`中

- `ldr x8, [x0]` 相当于`*x0 -> x8`或者 `v.begin -> x8`
- `ldr w0, [x8, #4]` 相当于`*(x8+4) -> w0`

我们接着分析`useAt`的代码。`v.at(1)`会被编译为`at(v, 1)`，因此`w1`中保存index=1。接着`bl`到`vector::at`。

```shell
std::__1::vector<int, std::__1::allocator<int> >::at(unsigned long) const
; %bb.0:
	ldp	x8, x9, [x0]
	sub	x9, x9, x8
	cmp	x1, x9, asr #2
	b.hs	LBB2_2
; %bb.1:
	add	x0, x8, x1, lsl #2
	ret
LBB2_2:
	bl	std::__1::__vector_base_common<true>::__throw_out_of_range() const
	.cfi_endproc
```
`vector::at`的代码比较多，我们逐条分析

- `ldp	x8, x9, [x0]` 这里用到`ldp`，表示load pair，它会一次load两个连续的值到寄存器中。`[x0]`中保存的是`v`的地址，因此`x8`是`v.begin()`，`x9`是`v.end()`
- `sub	x9, x9, x8` 是计算size,单位是bytes，相当于`x9 <- v.size()*sizeof(int)`
- `cmp	x1, x9, asr #2`，这里`x1`保存参数index，`x9 asr #2`是右移操作，相当于除法，除数为`2^2 = 4`，也就是说`x9 <- v.size()*sizeof(int) / sizeof(int)`。接下来的`cmp`操作用来检测index是否越界
- `b.hs`是conditional jump，如果越界，则jump到`LBB2_2`，进而throw exception
- `add	x0, x8, x1, lsl #2` 上面提到`x8`保存的是`v.begin`，`lsl`是左移，因此这里计算的是 `x0 <- x8 + x1<<2`，左移相当于乘法，继续展开相当于 `x0 <- v.begin() + index*sizeof(int)`实际上就是`this->beign[index]`

由此，我们可以推测`at`的实现大概为

```cpp
template<typename T>
const T& std::vector<T>::at(unsigned long index) const {
    if(index >= size()) {
        std::__1::__vector_base_common<true>::__throw_out_of_range();
    }
    return this->beign[index];
}
```

## Resources

- [ARM64指令集手册](https://developer.arm.com/documentation/100076/0100/a64-instruction-set-reference/a64-data-transfer-instructions)
- [Load-store Architecture](https://en.wikipedia.org/wiki/Load%E2%80%93store_architecture)


