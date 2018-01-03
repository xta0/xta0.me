---
layout: post
title: Transition To 64 Bit 
tag: iOS
categories: 随笔
---

<em>所有文章均为作者原创，转载请注明出处</em>

两个月前Apple放出了这样[一条消息](https://developer.apple.com/news/?id=10202014a)。今天Apple[又来了](https://developer.apple.com/news/)：

>As we announced in October, beginning on February 1, 2015 new iOS apps submitted to the App Store must include 64-bit support and be built with the iOS 8 SDK. Beginning June 1, 2015 app updates will also need to follow the same requirements. To enable 64-bit in your project, we recommend using the default Xcode build setting of “Standard architectures” to build a single binary with both 32-bit and 64-bit code.

意思是大概这么三条:

- 从2015年2月1日起，新提交的App，必须用iOS 8 SDK打包，且必须要包含64bit的版本。

- 从2015年6月1日起，更新的App也必须符合上面你的要求。

- 建议开发者用Standard architecture打包，会build出32bit和64bit两个版本。

这样的要求带来的结果是:

- 如果你的App还支持iOS 6以下的版本，那么对于iOS 6及以下的用户，在App Store上下载的app是 32bit的，对于iOS 7以上的用户，下载的是64 bit的版本。当然，还要将设备考虑进去，比如你用的是iPhone4，装了iOS7，那么也只能下载32bit的版本，因为iPhone4不支持64bit。

看来Apple这次是铁了心的要推广64bit的app，进而推动它新操作系统的更新（淘汰iOS 7以下的系统）和设备的更新换代（淘汰iPhone 5s以下的设备 ）。

但Apple还没有明确说禁止提交32bit的app，估计是要慢慢过渡。

今天仔细读了一遍[64-Bit Transition Guide for Cocoa Touch](https://developer.apple.com/library/ios/documentation/General/Conceptual/CocoaTouch64BitGuide/Introduction/Introduction.html#//apple_ref/doc/uid/TP40013501-CH1-SW1)。摘录下其中关键的部分:

##At a Glance

###Apple A7支持两种不同的指令集:
	
- 32bit ARM : 所有CPU都支持
	
- 64bit ARM : 64bit ARM
	
###使用64bit ARM architecture的优势:

- CPU寻址空间变大了
	
- 整型,浮点型的寄存器数量增加了一倍，这意味着有更多register可以被利用，从而性能将会带来极大的提升（访问寄存器的速度要远快于访问内存）。
	
- LLVM针对64bit进行了优化，提升了App的性能
	
###注意的点:

- 64bit的pointer意味着消耗更多的内存。
	
- 64bit的app只能运行在iOS 7.0.3以后的系统上，并且设备是支持64bit的（iPhone 5s以后）。
	
- 从32bit到64bit需要对基本的数据类型做一些处理,如int,NSInteger,float,CGFloat等。
	

##Major 64 bit changes

### 64bit 和 32bit的运行时环境，主要由两方面区别：

- 64bit环境中，基本数据类型均要求严格的内存对齐，这会带来自身size的增加。
	
- 64bit环境中，对函数原型([function protype](http://en.wikipedia.org/wiki/Function_prototype))的声明，有这个更严格的要求。
	

C或Objective-C是不会去限制原生数据类型的size的，因为这与具体的平台相关，不同平台会针对硬件环境和操作系统来重新定义这些数据类型。从32bit到64bit，这些基本数据类型的大小要重新定义。

###ILP32 vs LP64

32bit的runtime环境使用ILP32的数据模型，integer，long，pointer都是32bit长。64bit的runtime环境使用的是LP64的数据模型，integer是32bit长，long，pointer类型是64bit长。更多数据类型的变化如下图：

![alt text](/assets/images/2014/12/64bit-data-types.png)

关于浮点型的变化如下：

![alt text](/assets/images/2014/12/64bit-data-type-float.png)

- 对于使用可变参数的函数和普通函数的相互cast要格外小心。

- accessing isa：

>If you are writing low-level code that targets the Objective-C runtime directly, you can no longer access an object’s isa pointer directly. Instead, you need to use the runtime functions to access that information.

直接给出在64bit平台下，访问isa指针的方法：

```C

 #ifdef __arm64__
        // See http://www.sealiesoftware.com/blog/archive/2013/09/24/objc_explain_Non-pointer_isa.html        
        extern uint64_t objc_debug_isa_class_mask WEAK_IMPORT_ATTRIBUTE;
        clz = (__bridge Class)((void *)((uint64_t)obj->isa & objc_debug_isa_class_mask));
 #else
        clz = obj->isa;
 #endif

```

- 64bit的汇编指令集发生了变化，关于32bit的ARM指令集，可以参考[我之前的文章](http://akadealloc.github.io/blog/2013/06/15/assembly-on-arm.html)。


##Converting Your App to a 64-Bit Binary

### 不要将pointer强制转换为int类型。

考虑下面代码:

```c
int *c = something passed in as an argument....
int *d = (int *)((int)c + 4); // Incorrect.
int *d = c + 1;               // Correct!

```
在32bit的系统中，pointer和int大小相同，相互cast没有问题，但是在64bit的系统中，pointer的size大于int，因此这样cast会破坏pointer。如果一定要做转换，要使用`uintptr_t`。

### 保持数据类型一致性

例子:
	
```c

long PerformCalculation(void);
 
int  x = PerformCalculation(); // incorrect
long y = PerformCalculation(); // correct

```
函数返回值类型要保持一致。

### CocoaTouch的一些基本数据类型发生了变化

例子:

```c

// Incorrect.
CGFloat value = 200.0;
CFNumberCreate(kCFAllocatorDefault, kCFNumberFloatType, &value);
 
// Correct!
CGFloat value = 200.0;
CFNumberCreate(kCFAllocatorDefault, kCFNumberCGFloatType, &value);

```
### 有符号和无符号数据的运算要注意:
	
- 无符号数通过补零来转换为larger type。
	
- 有符号数通过扩展符号位来转换为larger type（如: `int a=-2`，表示为`0xfffffe`，`long b = a; ` 则b表示为 `0xffffff ffffe`）。
	
- 常量（除非明确声明其类型，如: 0x8L）都将使用size最小的数据类型来表示。16进制表示的数字会被编译器解析为`int`,`long`,`long long`类型。
	
- 当相同位数的有符号数和无符号数相加，结果为无符号数。

例子：
	
```c

int a=-2;
unsigned int b=1;
long c = a + b;
long long d=c; // to get a consistent size for printing.
 
printf("%lld\n", d);

```
上面代码在32bit运行时环境的结果为:` 4294967295 `。

- 原因是:

a的16进制值为 `0xfffffe`(-2的反码为`0xfffffd`,补码=反码+1), b的16进制值为`0x000001`,求和后c为`0xffffffff`，由于c为long型，根据上面第1条和第4条，高32位补0，结果为:`(0x00000000ffffffff)`。

- 解决办法:

一种巧妙的解决办法是将 b 声明为long型。这样b就变为 `0x0000000000000001`。由于a的位数小于b，因此a要补齐，根据上面第2条，a变为`0xfffffffffffffffe`。这样相加后，结果变为`0xffffffffffffffff`为`-1`。得到了正确的结果:

```c
int a=-2;
unsigned long b=1;
long c = a + b;
long long d=c; // to get a consistent size for printing.
 
printf("%lld\n", d);

```

###使用一些不会根据平台环境变化的数据结构：


![alt text](/assets/images/2014/12/64bit-data-type-c99.png)


###要兼顾字节对齐：

定义struct要注意字节对齐:

```c
struct bar {
    int32_t foo0;
    int32_t foo1;
    int32_t foo2;
    int64_t bar;
};
```
在32bit的环境中，`bar`的padding offset是12字节。到了64bit中，`bar`的padding offset变成了16字节。因为在64bit的环境中，最小alignment的字节数变为8字节，那么在`foo2`后面会额外补充4字节的0。

如果你要定义一个新的结构体，将size最大的数据类型定义在最前面，size最小的数据类型定义在最后面。这种定义方式会尽量减少padding的字节数。如果你要使用原先32bit环境下的struct，可以用`#pragma pack(4)`来强行指定字节对齐方式按照32bit来对齐，这种方式当然会损耗一些性能。

如下:

```c

 #pragma pack(4)
struct bar {
    int32_t foo0;
    int32_t foo1;
    int32_t foo2;
    int64_t bar;
};
 #pragma options align=reset

```


###函数与函数指针


在64bit的环境中，编译器生成的用来处理可变参数的函数的指令顺序和32bit环境有着较大的区别,因此这两者不可以强制相互cast：

```c

int MyFunction(int a, int b, ...);
 
int (*action)(int, int, int) = (int (*)(int, int, int)) MyFunction;
action(1,2,3); // Error!

```
在64bit的环境中，尽量使用函数原型，这种通过函数指针调用函数的方式有风险，因为可能会碰到可变参数的函数。


###Message Dispatch

`objc_msgSend`的函数原型为:`id objc_msgSend(id self, SEL op, ...)`，显然它是一个可变参数的函数。在32bit的环境中，我们可以通过这个方法在运行时调用某个类的某个method，而不用考虑可变参数的问题:

```c
  int ret1 = objc_msgSend(obj,@selector(method1:),@"hello");
  int ret2 = objc_msgSend(obj,@selector(method1:),@"hello",nil);

```
但是在64bit环境中，如果使用`objc_msgSend`来调用某个method，需要明确的提供这个method的函数原型:

```c

- (int) doSomething:(int) x { ... }
- 
- (void) doSomethingElse {
    int (*action)(id, SEL, int) = (int (*)(id, SEL, int)) objc_msgSend;
    action(self, @selector(doSomething:), 0);
}

```

###其它

- 不要自定义原生数据类型，因为它涉及到32bit和64bit两套环境，以及他们之间的相互转换。

- 不要直接访问`isa`指针，因为它指向的数据结构发生了变化。使用`object_getclass`或`object_setclass`。 

- 不要Hard Code虚拟内存的page size。关于iOS下的虚拟内存，参考之前的[这篇文章](http://akadealloc.github.io/blog/2012/07/10/Debuging-Memory-Issues-1.html)。

- 关于使用内存的优化，这部分不详细列出。


---

##附：iPhone Hardware/OS Model


| Hardware |iPhone 3gs| iPhone 4 |iPhone 4s| iPhone 5 | iPhone 5s | iPhone 6 | iPhone 6+ |
| -------- | ---------| -------- | ------- | -------- | --------- | -------- | --------- |
| CPU Model| ARM Cortex-A8 | ARM Cortex-A8（Apple A4） |  dual-core ARM Cortex-A9(Apple A5) | dual-core ARMv7s (Apple A6) | dual-core ARMv8-A (Apple A7) | dual-core ARMv8-A-64bit(Apple A8) | dual-core ARMv8-A-64bit(Apple A8 |
| CPU Frequence| 600 MHZ | 1G HZ| 1G HZ | 1.3G HZ |  1.3G HZ | 1.4G HZ | 1.4G HZ | 
| Bus Width | 32 bit | 32 bit | 32 bit | 32 bit | 64 bit | 64 bit | 64 bit | 
| GPU | PowerVR SGX535 GPU |PowerVR SGX535 GPU|dual-core PowerVR SGX535 GPU| triple-core PowerVR SGX543MP3 | quad-core PowerVR G6430 | quad-core PowerVR GX6450 | quad-core PowerVR GX6450 |
| GPU Frequence | 160 MHZ | 200 MHZ| 200 MHZ| 266MHZ | 450MHZ | 450MHZ | 
| RAM | 256 MB | 512 MB | 512 MB | 1G | 1G | 1G | 

---


##Further Reading


- [iPhone Hardware Models](http://en.wikipedia.org/wiki/List_of_iOS_devices)
- [ARM Cortext-A8](http://en.wikipedia.org/wiki/ARM_Cortex-A8)
- [Apple A4](http://en.wikipedia.org/wiki/Apple_A4)
- [Apple A5](http://en.wikipedia.org/wiki/Apple_A5)
- [Apple A6](http://en.wikipedia.org/wiki/Apple_A6)
- [Apple A7](http://en.wikipedia.org/wiki/Apple_A7)
- [Apple A8](http://en.wikipedia.org/wiki/Apple_A8)
- [Mike Ash:ARM64](https://www.mikeash.com/pyblog/friday-qa-2013-09-27-arm64-and-you.html)
- [iOS ABI](https://developer.apple.com/library/ios/documentation/Xcode/Conceptual/iPhoneOSABIReference/Introduction/Introduction.html#//apple_ref/doc/uid/TP40009023)



