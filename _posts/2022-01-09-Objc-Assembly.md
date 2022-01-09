---
list_title: 从汇编层面看Objective-C
title: 从汇编层面看Objective-C
layout: post
categories: ["C++", "Objective-C", "C", "Assembly"]
---

### Motivation

最近需要给组内分享一些iOS的知识，其中大部分听众是C/C++的工程师。由于FB还是在大量使用Objective-C，工作中一个误区是很多人认为Objective-C只是语法层面和C/C++不同，而实际上，这不完全正确。为了把Objective-C将清楚，本文尝试从汇编的角度来分析Objective-C的一些实现细节。一部分资料来自[Apple's Objective-C runtime open source release](https://opensource.apple.com/source/objc4/)，以及[Github上这个mirror](https://github.com/opensource-apple/objc4)。所有例子使用下面命令编译，需要本地安装XCode。

```shell
#!/usr/bin/env bash
xcrun --sdk iphoneos clang -arch arm64 -S -Os $@
```

### Class Metadata

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

#### Class artifacts

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

#### ivars

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
对应objc-runtime-new.h中的`struct ivar_t`。最后一段表明`struct class_ro_t`中的`ivars`指向上面提到的`ivar_list_t`。注意，这里保存`ivars`的是`class_ro_t`而不是`struct objc_class`。猜想这可能和`struct swift_class_t`有关，这里不做更多展开。





