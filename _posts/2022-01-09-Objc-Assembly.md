---
list_title: 从汇编层面看Objective-C
title: 从汇编层面看Objective-C
layout: post
categories: ["C++", "Objective-C", "C", "Assembly"]
---

## Introduction

本文从汇编的角度来分析Objective-C的一些实现细节。一部分资料来自[Apple's Objective-C runtime open source release](https://opensource.apple.com/source/objc4/)，以及[Github上这个mirror](https://github.com/opensource-apple/objc4)。所有例子使用下面命令编译，需要本地安装XCode

```shell
#!/usr/bin/env bash
xcrun --sdk iphoneos clang -arch arm64 -S -Os $@
```

## Class Metadata

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

# ----------

	.section	__TEXT,__text,regular,pure_instructions
	.build_version ios, 15, 0	sdk_version 15, 0
	.section	__DATA,__objc_imageinfo,regular,no_dead_strip
L_OBJC_IMAGE_INFO:
	.long	0
	.long	64

.subsections_via_symbols
```

### Class artifacts

`@implementation`会产生具体的汇编代码，我们先看一个Empty class

```objc
#import <Foundation/Foundation.h>

@interface SomeClass : NSObject
@end

@implementation SomeClass
@end
```
编译器会对上面的类产生下面代码

1. 一个C string的类名 (`L_OBJC_CLASS_NAME_`)
2. 


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

	.section	__DATA,__objc_imageinfo,regular,no_dead_strip
L_OBJC_IMAGE_INFO:
	.long	0
	.long	64

.subsections_via_symbols
```

