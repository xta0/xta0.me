---
layout: post
title: Swift and Objective-C Interoperability Part 1
list_title: Swift and Objective-C Interoperability Part 1
categories: [iOS, Objective-C, Swift]
---

今天我们来聊聊Swift和OC代码混合使用的问题。按照常理来说，Swift作为独立的编程语言和OC应该是不存在交集的。这个问题

### Return Value

我们先来做几个函数返回值的例子的实验，这些实验可以很好的提供问题的表象，帮助我们先建立起感性认识。我们假设一个Swift工程中有一个OC类，定义如下, 在

```objc
#import <Foundation/Foundation.h>
#import "SomeOCProtocol.h"

NS_ASSUME_NONNULL_BEGIN

@interface SomeOCClass : NSObject
- (id)someMethod;
- (id<SomeOCProtocol>) someMethod
@end

NS_ASSUME_NONNULL_END
```



