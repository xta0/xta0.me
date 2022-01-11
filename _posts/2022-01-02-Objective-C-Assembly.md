---
list_title: ARM | 从汇编层面看Objective-C的实现
title: 从汇编层面看Objective-C的实现
layout: post
categories: ["C++", "Objective-C", "C", "Assembly"]
---

### Motivation

最近需要给组内分享一些iOS的知识，其中大部分听众是C/C++的工程师。由于FB还是在大量使用Objective-C，工作中一个误区是很多人认为Objective-C只是语法层面和C/C++不同，而实际上，这不完全正确。为了把Objective-C将清楚，本文尝试从汇编的角度来分析Objective-C的一些实现细节。一部分资料来自[Apple's Objective-C runtime open source release](https://opensource.apple.com/source/objc4/)，以及[Github上这个mirror](https://github.com/opensource-apple/objc4)。所有例子使用下面命令编译，需要本地安装XCode。

```shell
#!/usr/bin/env bash
xcrun --sdk iphoneos clang -arch arm64 -S -Os $@
```

## Class Metadata

Objective-C中的class包含两部分`@interface`和`@implementation`

### `@interface`

Objective-C的类通常包含下面两部分`@interface`和`@implementation`。编译器对`@interface`本身并不产生有意义的汇编代码，如下面例子

```objc
#import <Foundation/Foundation.h>

@interface Noop{
@private
  NSString *aIvar;
}
- (int)aMethod;
@property (nonatomic, strong) NSString *aProperty;
@end
```
编译器产生的汇编代码为

```shell
	.section	__TEXT,__text,regular,pure_instructions
	.build_version ios, 15, 0	sdk_version 15, 0
	.section	__DATA,__objc_imageinfo,regular,no_dead_strip
L_OBJC_IMAGE_INFO:
	.long	0
	.long	64

.subsections_via_symbols
```

### `@implementation`

`@implementation`会产生具体的汇编代码，我们先看一个Empty class

```objc
#import <Foundation/Foundation.h>

@interface SomeClass : NSObject
@end

@implementation SomeClass
@end
```
编译器会对上面的类产生下面代码

- `L_OBJC_CLASS_NAME_`
- `__OBJC_CLASS_RO_$_SomeClass` 和 `__OBJC_METACLASS_RO_$_SomeClass` 对应[objc-runtime-new.h](https://opensource.apple.com/source/objc4/objc4-680/runtime/objc-runtime-new.h)中的`struct class_ro_t`
- `_OBJC_CLASS_$_SomeClass` 和 `_OBJC_METACLASS_$_SomeClass` 对应[objc-runtime-new.h](https://opensource.apple.com/source/objc4/objc4-680/runtime/objc-runtime-new.h)中的`struct objc_class`
- 一个指向` __DATA.__objc_classlist`的指针

具体的汇编代码如下

```shell
	.section	__TEXT,__objc_classname,cstring_literals
l_OBJC_CLASS_NAME_:                     ; @OBJC_CLASS_NAME_
	.asciz	"SomeClass"

	.section	__DATA,__objc_const
	.p2align	3                               ; @"_OBJC_METACLASS_RO_$_SomeClass"
__OBJC_METACLASS_RO_$_SomeClass:
	.long	1                               ; 0x1
	.long	40                              ; 0x28
	.long	40                              ; 0x28
	.space	4
	.quad	0
	.quad	l_OBJC_CLASS_NAME_
	.quad	0
	.quad	0
	.quad	0
	.quad	0
	.quad	0

	.section	__DATA,__objc_data
	.globl	_OBJC_METACLASS_$_SomeClass     ; @"OBJC_METACLASS_$_SomeClass"
	.p2align	3
_OBJC_METACLASS_$_SomeClass:
	.quad	_OBJC_METACLASS_$_NSObject
	.quad	_OBJC_METACLASS_$_NSObject
	.quad	__objc_empty_cache
	.quad	0
	.quad	__OBJC_METACLASS_RO_$_SomeClass

	.section	__DATA,__objc_const
	.p2align	3                               ; @"_OBJC_CLASS_RO_$_SomeClass"
__OBJC_CLASS_RO_$_SomeClass:
	.long	0                               ; 0x0
	.long	8                               ; 0x8
	.long	8                               ; 0x8
	.space	4
	.quad	0
	.quad	l_OBJC_CLASS_NAME_
	.quad	0
	.quad	0
	.quad	0
	.quad	0
	.quad	0

	.section	__DATA,__objc_data
	.globl	_OBJC_CLASS_$_SomeClass         ; @"OBJC_CLASS_$_SomeClass"
	.p2align	3
_OBJC_CLASS_$_SomeClass:
	.quad	_OBJC_METACLASS_$_SomeClass
	.quad	_OBJC_CLASS_$_NSObject
	.quad	__objc_empty_cache
	.quad	0
	.quad	__OBJC_CLASS_RO_$_SomeClass

	.section	__DATA,__objc_classlist,regular,no_dead_strip
	.p2align	3                               ; @"OBJC_LABEL_CLASS_$"
l_OBJC_LABEL_CLASS_$:
	.quad	_OBJC_CLASS_$_SomeClass
```
在64bit的ARM系统中，根据[ARM手册](https://developer.arm.com/documentation/100067/0612/armclang-Integrated-Assembler/Data-definition-directives)，一个`.quad`占`8`字节，一个`.long`占`4`字节。每一个`@implemention`至少有25个`.quad`和6个`long`，因此共占`200+24=224`字节。

### ivars

如果在类中增加一个ivar

```objc
@implementation SomeClass {
    NSString* _anIVar;
}
@end
```
观察汇编代码的变化

```shell
_OBJC_METACLASS_$_SomeClass:
	.quad	_OBJC_METACLASS_$_NSObject
	.quad	_OBJC_METACLASS_$_NSObject
	.quad	__objc_empty_cache
	.quad	0
	.quad	__OBJC_METACLASS_RO_$_SomeClass

	.private_extern	_OBJC_IVAR_$_SomeClass._anIVar1 ; @"OBJC_IVAR_$_SomeClass._anIVar1"
	.section	__DATA,__objc_ivar
	.globl	_OBJC_IVAR_$_SomeClass._anIVar1
	.p2align	2
_OBJC_IVAR_$_SomeClass._anIVar1:
	.long	8                               ; 0x8

	.section	__TEXT,__objc_methname,cstring_literals
l_OBJC_METH_VAR_NAME_:                  ; @OBJC_METH_VAR_NAME_
	.asciz	"_anIVar1"

	.section	__TEXT,__objc_methtype,cstring_literals
l_OBJC_METH_VAR_TYPE_:                  ; @OBJC_METH_VAR_TYPE_
	.asciz	"@\"NSString\""

	.section	__DATA,__objc_const
	.p2align	3                               ; @"_OBJC_$_INSTANCE_VARIABLES_SomeClass"
__OBJC_$_INSTANCE_VARIABLES_SomeClass:
	.long	32                              ; 0x20
	.long	1                               ; 0x1
	.quad	_OBJC_IVAR_$_SomeClass._anIVar1
	.quad	l_OBJC_METH_VAR_NAME_
	.quad	l_OBJC_METH_VAR_TYPE_
	.long	3                               ; 0x3
	.long	8                               ; 0x8

__OBJC_CLASS_RO_$_SomeClass:
	.long	0                               ; 0x0
	.long	8                               ; 0x8
	.long	16                              ; 0x10
	.space	4
	.quad	0
	.quad	l_OBJC_CLASS_NAME_
	.quad	0
	.quad	0
	.quad	__OBJC_$_INSTANCE_VARIABLES_SomeClass
	.quad	0
	.quad	0
```
从定义上看 `__OBJC_$_INSTANCE_VARIABLES_SomeClass`对应[objc-runtime-new.h](https://opensource.apple.com/source/objc4/objc4-680/runtime/objc-runtime-new.h)中的`struct ivar_list_t`加上一个`.long`（4 bytes）的overhead，其中

```shell
.long	1                               ; 0x1
.quad	_OBJC_IVAR_$_SomeClass._anIVar1
.quad	l_OBJC_METH_VAR_NAME_
.quad	l_OBJC_METH_VAR_TYPE_
.long	3                               ; 0x3
.long	8                               ; 0x8
```
对应objc-runtime-new.h中的`struct ivar_t`。最后一段表明`struct class_ro_t`中的`ivars`指向上面提到的`ivar_list_t`。注意，这里保存`ivars`的是`class_ro_t`而不是`struct objc_class`。猜想这可能和`struct swift_class_t`有关，这里不做更多展开。另外，如果我们有一个C++的`ivar`，并且它是template的，那么这个ivar的名字将会非常长，很不利与debug。下面例子是一个`std::vector<int> _vec`的ivar的名字

```
l_OBJC_METH_VAR_TYPE_.4:                ; @OBJC_METH_VAR_TYPE_.4
	.asciz	"{vector<int, std::allocator<int> >=\"__begin_\"^i\"__end_\"^i\"__end_cap_\"{__compressed_pair<int *, std::allocator<int> >=\"__value_\"^i}}"
```

### Methods

如果在类中加一个method

```objc
@implementation SomeClass
- (void)doSomething {}
@end
```
它所产生的的代码和`ivar`非常类似。编译器产生了一段代码用来保存method，对应objc-runtime-new.h中的`struct method_list_t`

```shell
__OBJC_$_INSTANCE_METHODS_SomeClass:
	.long	24                              ; 0x18
	.long	1                               ; 0x1
	.quad	l_OBJC_METH_VAR_NAME_
	.quad	l_OBJC_METH_VAR_TYPE_
	.quad	"-[SomeClass doSomething]"
```
前两个`.long`是8 bytes的overhead，后面则是`struct method_t`对象。同样，`__OBJC_$_INSTANCE_METHODS_SomeClass`保存在`struct class_ro_t`中，而不是`struct class_objc`中。

### Properties

接下来我们看property

```objc
#import <Foundation/Foundation.h>

@interface SomeClass : NSObject
@property(nonatomic, strong) NSString* aString;
@end

@implementation SomeClass
@end
```
编译器为property生成的代码较多，首先会为其自动生成getter和setter，并放到上面提到的`method_list_t`中

```shell
__OBJC_$_INSTANCE_METHODS_SomeClass:
	.long	24                              ; 0x18
	.long	2                               ; 0x2
	.quad	l_OBJC_METH_VAR_NAME_
	.quad	l_OBJC_METH_VAR_TYPE_
	.quad	"-[SomeClass aString]"
	.quad	l_OBJC_METH_VAR_NAME_.1
	.quad	l_OBJC_METH_VAR_TYPE_.2
	.quad	"-[SomeClass setAString:]"
```
其次，由于property是ivar的封装，上面提到`ivar_list_t`中会保存其对应的ivar

```shell
__OBJC_$_INSTANCE_VARIABLES_SomeClass:
	.long	32                              ; 0x20
	.long	1                               ; 0x1
	.quad	_OBJC_IVAR_$_SomeClass._aString
	.quad	l_OBJC_METH_VAR_NAME_.3
	.quad	l_OBJC_METH_VAR_TYPE_.4
	.long	3                               ; 0x3
	.long	8                               ; 0x8
```

同时，Objective-C的每个类也有一个`property_list_t`中，同样的，它也被保存在`struct class_ro_t`中

```shell
__OBJC_$_PROP_LIST_SomeClass:
	.long	16                              ; 0x10
	.long	1                               ; 0x1
	.quad	l_OBJC_PROP_NAME_ATTR_
	.quad	l_OBJC_PROP_NAME_ATTR_.5
__OBJC_CLASS_RO_$_SomeClass:
	.long	0                               ; 0x0
	.long	8                               ; 0x8
	.long	16                              ; 0x10
	.space	4
	.quad	0
	.quad	l_OBJC_CLASS_NAME_
	.quad	__OBJC_$_INSTANCE_METHODS_SomeClass
	.quad	0
	.quad	__OBJC_$_INSTANCE_VARIABLES_SomeClass
	.quad	0
	.quad	__OBJC_$_PROP_LIST_SomeClass
```
每个`property_list_t`保存的是一个`struct property_t`的object，其定义可参考objc-runtime-new.h。其中，property的name和attribute在汇编中均为C string

```shell
l_OBJC_PROP_NAME_ATTR_:                 ; @OBJC_PROP_NAME_ATTR_
	.asciz	"aString"

l_OBJC_PROP_NAME_ATTR_.5:               ; @OBJC_PROP_NAME_ATTR_.5
	.asciz	"T@\"NSString\",&,N,V_aString"
```
## Ivar Access

OC对ivar的访问和C/C++类似，都是使用offset。为了展示方便，我们先来看下C是如何访问ivar的

<div class="md-flex-h md-margin-bottom-24">
<div>
<pre class="highlight language-python md-no-padding-v md-height-full">
<code class="language-cpp">
struct SomeStruct {
  // giving x a nonzero offset.
  double dbl;             
  int x;
  int y;
};

int accessMember(struct SomeStruct *o){
  return o->x + o->y;
}

int accessArray(int o[4]) {
  return o[2] + o[3];
}
</code>
</pre>
</div>
<div class="md-margin-left-12">
<pre class="highlight md-no-padding-v md-height-full">
<code class="language-python">
_accessMember:
; %bb.0:
	ldp	w8, w9, [x0, #8]
	add	w0, w9, w8
	ret
_accessArray:
	.cfi_startproc
; %bb.0:
	ldp	w8, w9, [x0, #8]
	add	w0, w9, w8
	ret
</code>
</pre>
</div>
</div>

左边是一个简单的C struct，其中`accessMember`函数会访问`x`和`y`两个`ivar`。`accessArray`是用来做参照，通过对比汇编代码可以发现，struct对ivar的访问是用offset。具体来说，`x0`保存了`SomeStruct*`的地址，offset为8字节，因此`[x0, #8]`是找到`x`在内存中位置，`ldp`是load pair的意思，它可以一次load两个int。可以看到，这和数组访问的汇编代码相同。我们再来看看Objective-C的ivar access

<div class="md-flex-h md-margin-bottom-24">
<div>
<pre class="highlight language-python md-no-padding-v md-height-full">
<code class="language-cpp">
@implementation SomeClass{
int x;
  int y;
}
- (int)accessIvar
{
  return x + y;
}
@end
</code>
</pre>
</div>
<div class="md-margin-left-12">
<pre class="highlight md-no-padding-v md-height-full">
<code class="language-python">
"-[SomeClass accessIvar]":
; %bb.0:
	ldp	w8, w9, [x0, #8]
	add	w0, w9, w8
</code>
</pre>
</div>
</div>

生成的汇编和上面C/C++产生的汇编基本一致。

## Functions and Methods

这个可能是被误解最多的一块内容，大部分C++的程序员认为Objective-C的动态性来自于和C++类似的virtual table，但实际上这是不正确的，Objective-C的动态性来自"call by name"的设计，结合它的[runtime库](https://developer.apple.com/documentation/objectivec/objective-c_runtime?preferredLanguage=occ)，相比virtual table更加灵活，但却有更高的overhead。

C++中的一个virtual function call用的是virtual table，例如`o->doStuff()`会被编译成`o->vtbl->doStuffFunctionPointer()`，对应到汇编代码，基本上是一个load加上一个jump。下面我们OC中一个简单的function call和它的汇编代码如下所示

<div class="md-flex-h md-margin-bottom-24">
<div>
<pre class="highlight language-python md-no-padding-v md-height-full">
<code class="language-cpp">
void callIndirect(id o){
    [o doSomeStuff];
}
</code>
</pre>
</div>
<div class="md-margin-left-0">
<pre class="highlight md-no-padding-v md-height-full">
<code class="language-python">
"-[SomeClass callIndirect:]":
	.cfi_startproc
; %bb.0:
	mov	x0, x2
Lloh0:
	adrp	x8, _OBJC_SELECTOR_REFERENCES_@PAGE
Lloh1:
	ldr	x1, [x8, _OBJC_SELECTOR_REFERENCES_@PAGEOFF]
	b	_objc_msgSend
</code>
</pre>
</div>
</div>

基本上也是一个load加jump，`_objc_msgSend`定义在动态库中，其函数原型如下

```cpp
objc_msgSend(obj, @selector(message));
```
对应上面的汇编,`x0`保存了`o`的地址，`x1`保存了selector的地址。具体来说，`_OBJC_SELECTOR_REFERENCES`是一个非常大的数组，里面保存每个selector的pointer。`[x8, _OBJC_SELECTOR_REFERENCES_@PAGEOFF]`类似`OBJC_SELECTOR_REFERENCES[DO_STUFF_OFFSET]`，用来寻址具体的selector。

关于`objc_msgSend`的实现这里就不具体展开了，推荐阅读Resouce中Mike Ash的这篇文章。但是这里可以明显的看到OC的这种"call by name"的设计是有比较大的overhead的，虽然每个Object上面有method的cache，但是cache需要一个warm up的过程，比起C++这种纯静态的调用，还是会慢很多。而且也会带来更大的code size。

### Class Methods

C++中的静态类方法效率很高，基本上就是一个C function call，而Objective-C中的类方法的调用和成员方法类似，但确有更大的overhead，相比于成员函的调用，它多了一步fetch global class pointer - ` _OBJC_CLASSLIST_REFERENCES[DUMMY_CLASS_OFFSET]`

<div class="md-flex-h md-margin-bottom-24">
<div>
<pre class="highlight language-python md-no-padding-v md-height-full">
<code class="language-cpp">
void callDummy(){
    [Dummy dummy];
}
</code>
</pre>
</div>
<div class="md-margin-left-0">
<pre class="highlight md-no-padding-v md-height-full">
<code class="language-python">
__Z9callDummyv:
; %bb.0:
Lloh0:
  adrp	x8, _OBJC_CLASSLIST_REFERENCES_$_@PAGE
Lloh1:
  ldr	x0, [x8, _OBJC_CLASSLIST_REFERENCES_$_@PAGEOFF]
Lloh2:
  adrp	x8, _OBJC_SELECTOR_REFERENCES_@PAGE
Lloh3:
  ldr	x1, [x8, _OBJC_SELECTOR_REFERENCES_@PAGEOFF]
  b	_objc_msgSend
  .loh AdrpLdr	Lloh2, Lloh3
  .loh AdrpAdrp	Lloh0, Lloh2
  .loh AdrpLdr	Lloh0, Lloh1
  .cfi_endproc
</code>
</pre>
</div>
</div>

## Literals

Objective-C有很方便的literal syntax用来创建NSString, NSNumber, NSArray以及NSDictionary。其中大部分都是syntax sugar，会简介调用这些类的构造函数，但是这里有一个例外是NSString

<div class="md-flex-h md-margin-bottom-24">
<div>
<pre class="highlight language-python md-no-padding-v md-height-full">
<code class="language-cpp">
NSString *getLiteral()
{
  return @"Hello";
}
</code>
</pre>
</div>
<div class="md-margin-left-0">
<pre class="highlight md-no-padding-v md-height-full">
<code class="language-python">
	.section	__TEXT,__cstring,cstring_literals
l_.str:                                 ; @.str
	.asciz	"Hello"
	.section	__DATA,__cfstring
l__unnamed_cfstring_:
	.quad	___CFConstantStringClassReference
	.long	1992
	.space	4
	.quad	l_.str
	.quad	5
</code>
</pre>
</div>
</div>

`@"Hello"`被保存在了一个具有4个element的struct中，`___CFConstantStringClassReference`看着像`isa` pointer，第三个参数指向一个C string，第四个参数是字符串的长度。这说明`@"Hello"`并没有调用`[NSString alloc]`，而是用了更加efficient的一种方式。


## Resources

- [Objective-C runtime](https://developer.apple.com/documentation/objectivec/objective-c_runtime?preferredLanguage=occ) 
- [Objective-C Implementation and Performance Details for C and C++ Programmers](https://swolchok.github.io/objcperf/)
- [Non-fragile ivars](http://www.sealiesoftware.com/blog/archive/2009/01/27/objc_explain_Non-fragile_ivars.html)
- [objc_msgsend](https://www.mikeash.com/pyblog/friday-qa-2012-11-16-lets-build-objc_msgsend.html)



