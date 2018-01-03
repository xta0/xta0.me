---
title: 理解AFNetworking中的Runloop 
layout: post
tag: iOS
categories: 随笔

---


<em>所有文章均为作者原创，转载请注明出处</em>

最近大家都开始用<a href="https://github.com/AFNetworking/AFNetworking">AFNetworking</a>，今天看了下它网络请求的代码，采用的也是NSOperation+NSURLConnetion并发模型。一般使用这种模型都要解决一个问题：
NSURLConnection对象在下载完前，所在线程就退出了，NSOperation对象也就接收不到回调。

这个问题在<a href="http://stackoverflow.com/questions/9223537/asynchronous-nsurlconnection-with-nsoperation">stackoverflow</a>上已经讨论了N多次了，其原因也在apple的<a href="https://developer.apple.com/library/mac/documentation/Cocoa/Reference/Foundation/Classes/NSURLConnection_Class/Reference/Reference.html#//apple_ref/occ/instm/NSURLConnection/initWithRequest:delegate:startImmediately:">guide line</a>上写的很清楚了：NSURLConnection的delegate方法需要在connection发起的线程的runloop中调用。因此，当发起connection的线程exit了，delegate自然不会被调用，请求也就回不来了。

针对这个问题，通常有这么两种解法：


- 所有connection在主线程的runloop中发起，回调也都由主线程的runloop分发：

```objc
 NSRunLoop *runLoop = [NSRunLoop mainRunLoop];
[_connection scheduleInRunLoop:runLoop forMode:NSRunLoopCommonModes];
[_connection start];
```

- 让发起请求的线程不退出，通过内置一个runloop来实现
 
```c
NSRunLoop *runLoop = [NSRunLoop currentRunLoop];
[runLoop addPort:[NSPort port] forMode:NSRunLoopCommonModes];      
[_connection scheduleInRunLoop:runLoop forMode:NSRunLoopCommonModes];
[_connection start];       
[runLoop run];
```

这两种方法都不可取！

第一种方法是出于性能考虑，当并发的请求很多时，需要大量占用main runloop，会影响GUI性能。

第二种方法问题更大，connection虽然可以顺利完成，但由于线程一直被runloop占据，导致线程永远无法停止，线程池直接失去了对线程的控制，而由于线程无法退出，它的stackframe中引用的NSOperation对象也无法释放，并发数量上去后，无论是CPU的资源还是内存上，都会有问题。

AFNetworking解决这个问题采用了另一种方法：单独起一个global thread，内置一个runloop，所有的connection都由这个runloop发起，回调也都由它接收。这是个不错的想法，既不占用主线程，又不耗CPU资源： 

```objc
[self performSelector:@selector(operationDidStart) 
			  onThread:[[self class] networkRequestThread] 			withObject:nil 
		 waitUntilDone:NO 
		         modes:[self.runLoopModes allObjects]]
```
线程代码：

```objc
+ (void) __attribute__((noreturn)) networkRequestThreadEntryPoint:(id)__unused object {
    do {
        @autoreleasepool {
            [[NSRunLoop currentRunLoop] run];
        }
    } while (YES);
}

+ (NSThread *)networkRequestThread {
    static NSThread *_networkRequestThread = nil;
    static dispatch_once_t oncePredicate;
    
    dispatch_once(&amp;oncePredicate, ^{
        _networkRequestThread = [[NSThread alloc] initWithTarget:self selector:@selector(networkRequestThreadEntryPoint:) object:nil];
        [_networkRequestThread start];
    });
    
    return _networkRequestThread;
}
```

整个请求流程如下：

![Alt text](/assets/images/2012/11/afnetworking.png)

这个想法其实不是AFNetworking最早想出来的，是apple的一个demo：<a href="https://developer.apple.com/LIBRARY/IOS/samplecode/MVCNetworking/Introduction/Intro.html ">MVCNetworking</a>，AFNetworking简单粗暴的借鉴了这个demo。

不论是AFNetworking还是MVCNetworking，保持runloop的代码均为：

```objc 
do {
        @autoreleasepool 
        {
            [[NSRunLoop currentRunLoop] run];
        }
    } while (YES);
``` 

另一种优雅的方式是，直接塞一个input source:
 
```objc
@autoreleasepool
{        
	NSRunLoop *runLoop = [NSRunLoop currentRunLoop];
	[runLoop addPort:[NSPort port] forMode:NSDefaultRunLoopMode];
	[runLoop run];
}
```

更多关于Runloop的基础知识清参考[前一篇文章](http://vizlabxt.github.io/blog/2012/11/17/Understanding-Runloop/)