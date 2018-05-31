---
list_title: 规避__unretain_unsafe指针
layout: post
tag: Objective-C
categories: 随笔

---


<em>所有文章均为作者原创，转载请注明出处</em>

最近被迫在大量的使用__unsafe_unretained pointer，用起来坑太多了
先看一段必然会crash的代码：

```objc
// Do any additional setup after loading the view.
 _welcomeView = [[ATCSearchLivingWelcomeView alloc]initWithFrame:CGRectMake(0, 0, 100,100)];
 [self.view addSubview:_welcomeView];
 
 __unsafe_unretained MXSecondViewController* weakSelf = self;
 [_welcomeView startAnimatingWithCompletionBlock:^{
    
     [weakSelf internalMethod];
 }];
```

假设welcomeView的animation执行10秒，那么animation结束前，释放controller，必然会crash。
原因也很简单，由于weakSelf是unsafe的，那么它使dealloc之后，指针也不会nil，变成了野指针。

通常这种问题解决的办法是当controller dealloc的时候将回调的block手动释放：

```objc
- (void)dealloc
{
    _welcomeView.block = nil;
    
}
```

但是这种方法治标不治本而已，总不能将block全部暴露出来手动释放一遍，因此，我们需要一种通用的解决方法。

我们先来分析下，问题产生的原因：

多数情况下使用__unsafe_unretained指针是由于在iOS5.0下无法使用__weak，但又要解决使用block产生的retain-cycle。就上面那个例子来说，对象间的引用关系如下：

<a href="/assets/images/2013/11/retain-1.png"><img src="/assets/images/2013/11/retain-1.png" alt="retain-1" width="222" height="105" class="alignnone size-full wp-image-330" /></a>

当controller dealloc之后：

<a href="/assets/images/2013/11/retain-2.png"><img src="/assets/images/2013/11/retain-2.png" alt="retain-2" width="211" height="110" class="alignnone size-full wp-image-331" /></a>

一种通用的解决方案来是mike ash的这边文章<a href="http://www.mikeash.com/pyblog/friday-qa-2010-07-16-zeroing-weak-
references-in-objective-c.html">MAZeroingWeakRef</a>。

使用方法如下：

```objc 
MXT_ZeroingWeakRef* ref = [[MXT_ZeroingWeakRef alloc]initWithTarget:self];
[_welcomeView startAnimatingWithCompletionBlock:^{

    [ref.target internalMethod];
}];
```
MXT_ZeroingWeakRef是我对MAZeroingWeakRef的精简和改写，去掉了CoreFoundation对象的兼容。但是思路基本上是一致的。这种用法和C++0x或BOOST库中的智能指针类似：

```
std::weak_ptr<DataBase> DBObserver = self;
```

里面实现的思路有些绕，确实要花一点时间才能把它完全理清楚。最关键的一点是"isa-swizzling"，也就是apple实现<a href="/blog/?p=18">KVO</a>的办法，理解了这个，剩下的就迎刃而解了。

