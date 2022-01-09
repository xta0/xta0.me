---
list_title: 从汇编层面看Objective-C
title: 从汇编层面看Objective-C
layout: post
categories: ["C++", "Objective-C", "C", "Assembly"]
---

本文从汇编的角度来分析Objective-C的一些实现细节。一部分资料来自[Apple's Objective-C runtime open source release](https://opensource.apple.com/source/objc4/)，以及[Github上这个mirror](https://github.com/opensource-apple/objc4)。所有例子使用下面命令编译，需要本地安装XCode

```
#!/usr/bin/env bash

xcrun --sdk iphoneos clang -arch arm64 -S -Os $@
```

## Class Metadata

Objective-C的类通常包含下面两部分`@interface`和`@implementation`。编译器对`@interface`并不产生有意义的汇编代码

<div class="md-flex-h md-margin-bottom-24">
<div>
<pre class="highlight language-python md-no-padding-v md-height-full">
<code class="language-cpp">
#import <Foundation/Foundation.h>

@interface Noop{
@private
  NSString *aIvar;
}
- (int)aMethod;
@property (nonatomic, strong) NSString *aProperty;
@end
</code>
</pre>
</div>
<div class="md-margin-left-12">
<pre class="highlight md-no-padding-v md-height-full">
<code class="language-python">
	.section	__TEXT,__text,regular,pure_instructions
	.build_version ios, 15, 0	sdk_version 15, 0
	.section	__DATA,__objc_imageinfo,regular,no_dead_strip
L_OBJC_IMAGE_INFO:
	.long	0
	.long	64

.subsections_via_symbols
</code>
</pre>
</div>
</div>

### Class artifacts

