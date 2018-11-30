---
list_title: 规避__unretain_unsafe指针 | Fix unretain-unsafe Pointer in iOS
title: Fix unretain-unsafe Pointer 
layout: post
categories: [iOS, Objective-C]
---

最近被迫在大量的使用__unsafe_unretained pointer，用起来坑太多了
先看一段必然会crash的代码：

```objc
// Do any additional setup after loading the view.
 _welcomeView = [[ATCSearchLivingWelcomeView alloc]initWithFrame:CGRectMake(0, 0, 100,100)];
 [self.view addSubview:_welcomeView];
 
 __unsafe_unretained MXSecondViewController* unsafeSelf = self;
 [_welcomeView startAnimatingWithCompletionBlock:^{
     [unsafeSelf internalMethod];
 }];
```

假设welcomeView的animation执行10秒，那么animation结束前，释放controller，必然会crash。原因也很简单，由于`unsafeSelf`的类型是`__unsafe_unretained`的，那么它所指向的对象在dealloc之后，自己也不会`nil`，因此，`unsafeSelf`变成了野指针。

通常这种问题解决的办法是当controller dealloc的时候将回调的block手动释放：

```objc
- (void)dealloc{
    _welcomeView.block = nil;
}
```

但是这种方法治标不治本而已，总不能将block全部暴露出来手动释放一遍，因此，我们需要一种通用的解决方法。

我们先来分析下，问题产生的原因：

多数情况下使用`__unsafe_unretained`指针是由于在iOS5.0下无法使用`__weak`，但又要解决使用block产生的retain-cycle。就上面那个例子来说，对象间的引用关系如左图，当Controller释放后，对象间关系变成了右图：

<div class="md-flex-h">
<div><img src="{{site.baseurl}}/assets/images/2013/11/retain-1.png"></div>
<div class="md-margin-left-12"><img src="{{site.baseurl}}/assets/images/2013/11/retain-2.png"></div>
</div>


解决这个问题，一种通用的解决方案来是mike ash的这边文章<a href="http://www.mikeash.com/pyblog/friday-qa-2010-07-16-zeroing-weak-
references-in-objective-c.html">MAZeroingWeakRef</a>。

使用方法如下：

```objc 
MXT_ZeroingWeakRef* ref = [[MXT_ZeroingWeakRef alloc]initWithTarget:self];
[_welcomeView startAnimatingWithCompletionBlock:^{

    [ref.target internalMethod];
}];
```
MXT_ZeroingWeakRef是我对MAZeroingWeakRef的精简和改写，去掉了CoreFoundation对象的兼容。但是思路基本上是一致的。这种用法和C++0x或BOOST库中的智能指针类似：

```cpp
std::weak_ptr<DataBase> DBObserver = self;
```

里面实现的思路有些绕，确实要花一点时间才能把它完全理清楚。最关键的一点是"isa-swizzling"，也就是apple实现<a href="/blog/?p=18">KVO</a>的办法，理解了这个，剩下的就迎刃而解了。

## Further Reading

- [Introduce to MAZeroingWeakRef](https://mikeash.com/pyblog/introducing-mazeroingweakref.html)
- [Github MAZeroingWeakRef](https://github.com/mikeash/MAZeroingWeakRef)