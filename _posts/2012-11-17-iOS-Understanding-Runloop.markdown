---
layout: post
list_title: 理解iOS中的Runloop
categories: 随笔
tag: Runloop

---

## Runloop的事件轮训

了解Runloop的工作方式很重要，以前一直以为Main Runloop是一个60fps的回调，后来发现这种理解是不对的，60fps是屏幕的刷新频率，工作在更底层，它和Runloop并没有直接对应的关系。Runloop实质上是一种由操作系统控制的事件轮训机制，类似的消息模型每个GUI操作系统都有，比如MFC的:

```c++

while(GetMessage(&msg,NULL,0,0))
{
   TranslateMessage (&msg);
   DispatchMessage (&msg) ;
}

```

根据Apple开放的Runloop代码来看，思路是一样的:

```c

void CFRunLoopRun(void) {	/* DOES CALLOUT */
    int32_t result;
    do {
        result = CFRunLoopRunSpecific(CFRunLoopGetCurrent(), kCFRunLoopDefaultMode, 1.0e10, false);
        CHECK_FOR_FORK();
    } while (kCFRunLoopRunStopped != result && kCFRunLoopRunFinished != result);
}

```
追着看下去，最终执行的函数是`__CFRunLoopRun`。

根据官方文档所载：

> A run loop receives events from two different types of sources. Input sources deliver asynchronous events, usually messages from another thread or from a different application. Timer sources deliver synchronous events, occurring at a scheduled time or repeating interval. Both types of source use an application-specific handler routine to process the event when it arrives.

Runloop只检测两种类型的消息，一种是Input Sources，通常是来自其它线程或进程。另一种是同步事件，来自Timer。这两种事件都可以在app中被获取并做处理：

<img src="/assets/images/2012/11/runloop.png" width="496" height="262">

如果想要获取Runloop每一次扫描的回调，可以注册observer:

```c

CFRunLoopObserverContext context = {
   0,
   (__bridge void *)(self),
   NULL,
   NULL,
   NULL
};
CFRunLoopObserverRef observerRef = CFRunLoopObserverCreate(kCFAllocatorDefault, kCFRunLoopAllActivities, YES, 0, &runloopCallback, &context);
CFRunLoopAddObserver(CFRunLoopGetCurrent(), observerRef, kCFRunLoopCommonModes);

```

上面代码以Main Runloop为例，注册了一个回调函数:`runloopCallback`,同时指定了要监听事件类型为`kCFRunLoopAllActivities`。在回调函数中，将被监听的事件打印出来：

```c

void runloopCallback(CFRunLoopObserverRef observer, CFRunLoopActivity activity, void *info)
{
    NSString* activityStr = @"unkown";
    if (((activity >> 0)&1) == 1) {
        activityStr = @"kCFRunLoopEntry";
    }
    else if (((activity >> 1)&1) == 1){
        activityStr = @"kCFRunLoopBeforeTimers";
    }
    else if (((activity >> 2)&1) == 1){
        activityStr = @"kCFRunLoopBeforeSources";
    }
    else if (((activity >> 5)&1) == 1){
        activityStr = @"kCFRunLoopBeforeWaiting";
    }
    else if (((activity >> 6)&1) == 1){
        activityStr = @"kCFRunLoopAfterWaiting";
    }
    else if (((activity >> 7)&1) == 1){
        activityStr = @"kCFRunLoopExit";
    }
    else
    {
        activityStr = @"kCFRunLoopAllActivities";
    }
    
    printf("[%.4f] activity:%s\n",CFAbsoluteTimeGetCurrent(),activityStr.UTF8String);
}

```

可以看出，RunloopActivity包含下面这6种事件:

- `kCFRunLoopEntry`：Runloop准备启动
- `kCFRunLoopBeforeTimers`：Runloop准备处理timer事件
- `kCFRunLoopBeforeSources`：Runloop准备处理input sources
- `kCFRunLoopBeforeWaiting`：Runloop准备进入休眠
- `kCFRunLoopAfterWaiting`：Runloop被唤醒准备处理消息
- `kCFRunLoopExit`：Runloop退出

如果程序监听了`kCFRunLoopAllActivities`，那么当Runloop每轮训到某个activity的时候，程序会收该activity事件的回调。

运行上面代码，输出下面的log：

```

[471507989.6447] activity:kCFRunLoopEntry
[471507989.6448] activity:kCFRunLoopBeforeTimers
[471507989.6448] activity:kCFRunLoopBeforeSources
[471507989.6456] activity:kCFRunLoopBeforeTimers
[471507989.6456] activity:kCFRunLoopBeforeSources
[471507989.6456] activity:kCFRunLoopBeforeWaiting

...


[471507991.0989] activity:kCFRunLoopAfterWaiting
[471507991.0993] activity:kCFRunLoopBeforeTimers
[471507991.0993] activity:kCFRunLoopBeforeSources
[471507991.0993] activity:kCFRunLoopBeforeWaiting

```

可以发现，Runloop会在启动的时候依次轮训Timer,Sources，过一阵后，如果没有任何的Input Source发生或Timer，Runloop将进入`kCFRunLoopBeforeWaiting`，即休眠状态。

这时候，如果有touch事件进来，相当于产生了一个Input Source，这时候：

1. Runloop会先发一个被唤醒的回调：`kCFRunLoopAfterWaiting`
2. 接着处理touch事件，`kCFRunLoopBeforeSources`会回调
3. 触发app中的 `- (void)touchesBegan:(NSSet* )touches withEvent:(UIEvent *)event`方法。
4. 重新进入休眠状态
5. 当touch end的时候又会唤醒Runloop，重复上面的操作的操作。

```c

[471512170.7319] activity:kCFRunLoopBeforeSources
VZRunloopSample[40478:3595823] touch began
[471512170.7327] activity:kCFRunLoopBeforeSources
[471512170.7328] activity:kCFRunLoopBeforeSources
[471512170.7328] activity:kCFRunLoopBeforeWaiting
[471512170.8271] activity:kCFRunLoopBeforeSources
VZRunloopSample[40478:3595823] touch end
[471512170.8277] activity:kCFRunLoopBeforeSources
[471512170.8277] activity:kCFRunLoopBeforeWaiting
[471512171.4833] activity:kCFRunLoopBeforeSources
[471512171.4833] activity:kCFRunLoopBeforeWaiting

```

类似的，如果有Timer进来，`kCFRunLoopBeforeTimers`会先回调，接着再执行Timer的callback函数。

## 什么是Input Sources

上面提到了很多次Input Sources，那么什么是Input Sources呢？文档的解释是:

> Input sources deliver events asynchronously to your threads. 

就是说Input Sources的作用是向Runloop所在线程异步的投递消息，消息源有两种，一种是Port-based的，它监听应用来自Mach Port发出的消息；另一种是自定义消息，它来自其它线程。无论是哪种消息源，都可以被Runloop监听。

### Port-based Sources

OSX提供了一种基于MachPort的线程或进程间的通信方式，这种方式需要添加一个NSMachPort对象到Runloop的Source中，当Port中有消息发送过来时，会唤醒Runloop来处理消息:

```c

// Delegates are sent this if they respond, otherwise they
// are sent handlePortMessage:; argument is the raw Mach message
- (void)handleMachMessage:(void *)msg
{
	
}

```
注意，参数`msg`的类型为`NSPortMessage`，这个类在OSX中可以正常使用，原因是OSX对app没有进程通信的限制，但是在iOS中，这个类只在`NSPort.h`中有一个前向声明，无法直接使用，推测涉及到沙盒安全的问题，因此还没找到iOS中使用`NSPort`进行线程通信的方法。

> 2015年12月14日，补充：在iOS8之后，有了Application Group的概念，在一个group里的application可以使用`CFMessagePortCreateLocal` API打开port进行通信

### 自定义的Input Sources

文档中给出的这张图已经说的很清楚了：

<img src="/assets/images/2012/11/runloop2.png" width="475" height="233">

如果想使用自定义的Input Source向Runloop投递消息，需要使用CoreFoundation的API，Foundation的API不提供创建自定义的Input Source，根据文档，使用Input Source需要如下步骤：

- 创建一个`CFRunloopContextRef`,实现三个回调函数:
	- `void	(*schedule)(void *info, CFRunLoopRef rl, CFStringRef mode);`
	- `void	(*cancel)(void *info, CFRunLoopRef rl, CFStringRef mode);`
	- `void	(*perform)(void *info);`

- 通过上面的context创建一个RunloopSourceRef:
	- `CFRunLoopSourceRef runLoopSource = CFRunLoopSourceCreate(NULL, 0, &context);` 

- 将创建的RunloopSource添加到当前的Runloop中:
	- `CFRunLoopRef runLoop = CFRunLoopGetCurrent();`
   - `CFRunLoopAddSource(runLoop, runLoopSource, kCFRunLoopDefaultMode);` 

- 向RunloopSource发送Signal，唤醒Runloop处理事件

```c
 	CFRunLoopSourceSignal(runLoopSource);
	CFRunLoopWakeUp(runloop);
```
	
使用这种方式可以实现线程间的通信，通常用于子线程向主线程通信，但是数据传递会很麻烦，暂时想不到应用场景。
	

### 使用PerformSelector

除了上面两种Input Sources，`performSelector`也是通过runloop执行的，`selector`对应的方法在Runloop中串行执行，`performSelector`通常用在主线程，如果要在其它线程使用，需要这个线程有一个活着的Runloop保持线程不退出。


## 理解Runloop Modes

官方文档对Runloop Modes的描述有只有三小段文字，但信息量确非常的大，说的也很晦涩。

首先是说Runloop Mode包含一组input sources，timers和observers的集合，只有mode中包含的Source才会被Runloop扫描到。

理解Runloop Mode，要先看下`CFRunLoop`的结构：

```c

struct __CFRunLoop {
    CFRuntimeBase _base;
    pthread_mutex_t _lock;			/* locked for accessing mode list */
    __CFPort _wakeUpPort;			// used for CFRunLoopWakeUp 
    Boolean _unused;
    volatile _per_run_data *_perRunData;              // reset for runs of the run loop
    pthread_t _pthread;
    uint32_t _winthread;
    CFMutableSetRef _commonModes;
    CFMutableSetRef _commonModeItems;
    CFRunLoopModeRef _currentMode;
    CFMutableSetRef _modes;
    struct _block_item *_blocks_head;
    struct _block_item *_blocks_tail;
    CFTypeRef _counterpart;
};

```

通过这个结构可以看到：

- 每一个`CFRunloop`中都包含一个`_commonModes`的集合，它用来保存Runloop支持的mode类型（包括系统默认的mode和自定义的mode），我们可以在运行时访问`__CFRunLoop`这个结构，来查看`_commonModes`中具体保存了哪些mode:

```c

CFRunLoopRef runloopRef = CFRunLoopGetCurrent();
struct __CFRunLoop* runloopStructPointer = runloopRef;

```
上面代码中，打印出的`_commonModes`结构如下：

```
common modes = <CFBasicHash 0x7f9f43601ea0 [0x1013297b0]>{type = mutable set, count = 2,
entries =>
	0 : <CFString 0x10244c270 [0x1013297b0]>{contents = "UITrackingRunLoopMode"}
	2 : <CFString 0x101349b60 [0x1013297b0]>{contents = "kCFRunLoopDefaultMode"}
}
```

说明当前Runloop支持两种mode：一个是`UITrackingRunLoopMode`,一个是`kCFRunLoopDefaultMode`

-  每一个`CFRunloop`中都包含一个`_commonModeItems`集合，它用来保存某个Mode中支持的消息源，包括`Timer/Input Source/Observers`, 类型为`CFRunLoopSourceRef`。每当有一个上述对象添加进来时，便向这个集合增加一个元素:

```c

CFSetAddValue(rl->_commonModeItems, rls);

```

默认情况下，当Runloop启动后，系统会自动在`_commonModeItems`中添加一些列Source和Observer。

- 每一个`CFRunloop`中还包含一个`_currentMode`表示当前运行在的Mode

- 通过`CFRunLoopCopyCurrentMode(CFRunloopRef runloop)`这个API可以看到系统默认预置进来的几种Mode:

	- UITrackingRunLoopMode,
	- GSEventReceiveRunLoopMode,
	- kCFRunLoopDefaultMode,
	- UIInitializationRunLoopMode,
	- kCFRunLoopCommonModes

	这些mode和官方文档上给出的并不一致，原因是iOS和MacOS对某些模式的命名不同，这些mode中，需要注意的有三种:kCFRunLoopDefaultMode, kCFRunLoopCommonModes和UITrackingRunLoopMode

### kCFRunLoopDefaultMode

在创建`CFRunLoopRef`的时候`_commonModes`默认填充了一个`kCFRunLoopDefaultMode`：

```c
static CFRunLoopRef __CFRunLoopCreate(pthread_t t) {

	CFRunLoopRef loop = NULL;
	CFRunLoopModeRef rlm;
	...
		
	loop->_commonModes = CFSetCreateMutable(kCFAllocatorSystemDefault, 0, &kCFTypeSetCallBacks);
	CFSetAddValue(loop->_commonModes, kCFRunLoopDefaultMode);
	loop->_commonModeItems = NULL;
	
	...
	return loop;
}

```
也就是说Runloop默认运行的mode是`kCFRunLoopDefaultMode`,官方文档对`kCFRunLoopDefaultMode`的解释是:

> The default mode is the one used for most operations. Most of the time, you should use this mode to start your run loop and configure your input sources.

在开源的Runloop代码中，并没有找到关于`kCFRunLoopDefaultMode`太多的代码，默认情况下它对应的`_commonModeItems`为空。

### kCFRunLoopCommonModes

除了`kCFRunLoopDefaultMode`系统为Runloop还提供了一种mode叫`kCFRunLoopCommonModes`,文档对它的解释是:

> This is a configurable group of commonly used modes. Associating an input source with this mode also associates it with each of the modes in the group. For Cocoa applications, this set includes the default, modal, and event tracking modes by default. Core Foundation includes just the default mode initially. You can add custom modes to the set using the CFRunLoopAddCommonMode function.

这个模式很特别，如果指定了Runloop的mode为`kCFRunLoopCommonModes`，系统默认将现在的已经存在的各种`Timers/Input Sources/Observers`默认添加到`_commonModeItems`这个集合里，也就是说它拥有各个Mode的特性，从源码中也能看出这一点:

以addTimer为例:

```c

void CFRunLoopAddTimer(CFRunLoopRef rl, CFRunLoopTimerRef rlt, CFStringRef modeName)
{
	...
		
	if (modeName == kCFRunLoopCommonModes)
	{
		CFSetRef set = rl->_commonModes ? CFSetCreateCopy(kCFAllocatorSystemDefault, rl->_commonModes) : NULL;
		
		if (NULL == rl->_commonModeItems) 
		{
			rl->_commonModeItems = CFSetCreateMutable(kCFAllocatorSystemDefault, 0, &kCFTypeSetCallBacks);
		}
		CFSetAddValue(rl->_commonModeItems, rls);
		if (NULL != set) 
		{
			CFTypeRef context[2] = {rl, rls};
			/* add new item to all common-modes */
			CFSetApplyFunction(set, (__CFRunLoopAddItemToCommonModes), (void *)context);
			CFRelease(set);
		}
	}
	...
}

```

每次有`Timer/Observer/Resource`添加到Runloop中的时候，如果Runloop运行在`kCFRunLoopCommonModes`模式，都会将`rls`添加到`_commonModeItems`中。

一个典型的场景就是：当Runloop运行在`kCFRunLoopDefaultModes`模式时，向Runloop添加了一个Timer：

```c

NSTimer* timer = [NSTimer timerWithTimeInterval:1.0f target:self selector:@selector(timerFireMethod:) userInfo:nil repeats:YES];
[[NSRunLoop mainRunLoop] addTimer:timer forMode:NSDefaultRunLoopModes];

```

如果此时界面上有ScrollView，当ScrollView滑动时，Runloop的模式切换到了`UITrackingRunLoopMode`上，由于此时的`_commonModeItems`中没有Timer，因此无法收到Timer的回调。解决办法是将timer添加到`NSRunLoopCommonModes`中，这样Timer就会被添加到`_commonModeItems`中:

```c
    
[[NSRunLoop mainRunLoop] addTimer:timer forMode:NSRunLoopCommonModes];

```


## 系统对Runloop的使用

大部分系统的行为都和Runloop有着直接或间接的关系，尤其是Main Runloop，下面举出了部分

### CATransaction

CATransaction注册了两个Runloop的Observer:

```
9 : <CFRunLoopObserver 0x7f9f43715c40 [0x1013297b0]>{valid = Yes, activities = 0xa0, repeats = Yes, order = 1999000, callout = _beforeCACommitHandler (0x101704a54), context = <CFRunLoopObserver context 0x7f9f436072a0>}

16 : <CFRunLoopObserver 0x7f9f43715da0 [0x1013297b0]>{valid = Yes, activities = 0xa0, repeats = Yes, order = 2001000, callout = _afterCACommitHandler (0x101704a99), context = <CFRunLoopObserver context 0x7f9f436072a0>}

```

- 第一个是`beforeCommit`,activity是`kCFRunLoopExit|kCFRunLoopBeforeWaiting`,优先级是`1999000`
- 第二个是`afterCommit`,activity是`kCFRunLoopExit|kCFRunLoopBeforeWaiting`,优先级是`2001000`

这里能看出CATransaction是等到Runloop空闲的时候才去做commit，这点非常重要，我们也可以自己写一段demo来验证：

我们定义一个Label，Override `drawTextInRect:`：


```c

- (void)drawTextInRect:(CGRect)rect
{
    CGContextRef context = UIGraphicsGetCurrentContext();
    UIImage* image = [UIImage imageNamed:@"sv.jpg"];
    CGContextDrawImage(context, rect, image.CGImage);
    [super drawTextInRect:rect];
}


```

可以看到调用堆栈如下:

```
frame #1: 0x014b4711 UIKit`-[UILabel drawRect:] + 98

...

frame #13: 0x010b435e QuartzCore`CA::Layer::layout_and_display_if_needed(CA::Transaction*) + 38
frame #14: 0x010a6e8b QuartzCore`CA::Context::commit_transaction(CA::Transaction*) + 317
frame #15: 0x010dae03 QuartzCore`CA::Transaction::commit() + 561
frame #16: 0x010db6c4 QuartzCore`CA::Transaction::observer_callback(__CFRunLoopObserver*, unsigned long, void*) + 92
frame #17: 0x0093c61e CoreFoundation`__CFRUNLOOP_IS_CALLING_OUT_TO_AN_OBSERVER_CALLBACK_FUNCTION__ + 30
frame #18: 0x0093c57e CoreFoundation`__CFRunLoopDoObservers + 398
frame #19: 0x00931728 CoreFoundation`CFRunLoopRunSpecific + 504
frame #20: 0x0093151b CoreFoundation`CFRunLoopRunInMode + 123
frame #21: 0x011ef854 UIKit`-[UIApplication _run] + 540

...

```

### AutoReleasePool

AutoReleasePool在回收对象的时候，也是依赖Runloop的回调:

```
13 : <CFRunLoopObserver 0x7f9f43715f80 [0x1013297b0]>{valid = Yes, activities = 0xa0, repeats = Yes, order = 2147483647, callout = _wrapRunLoopWithAutoreleasePoolHandler (0x1016d1c4e), context = <CFArray 0x7f9f43715d70 [0x1013297b0]>{type = mutable-small, count = 1, values = (
0 : <0x7f9f44800048>
)}}

14 : <CFRunLoopObserver 0x7f9f43715ee0 [0x1013297b0]>{valid = Yes, activities = 0x1, repeats = Yes, order = -2147483647, callout = _wrapRunLoopWithAutoreleasePoolHandler (0x1016d1c4e), context = <CFArray 0x7f9f43715d70 [0x1013297b0]>{type = mutable-small, count = 1, values = (
0 : <0x7f9f44800048>

```

这里可以看到AutoReleasePool注册了两个observer，activities分别为`0xa0`和`0x01`，翻译过来是`kCFRunLoopEntry|kCFRunLoopBeforeWaiting`和`kCFRunLoopExit`，优先级也是最高(IntMax)和最低(-IntMax)。推测就是：在Runloop开始的时候，最先进行一次对象回收，在Runloop结束的时候，再进行一次对象的回收。


## 其它

除了上面看到的两个Runloop Observer，再列举一下其它的：

- `__IOHIDEventSystemClientAvailabilityCallback`
- `_ZL20notify_port_callbackP12__CFMachPortPvlS1_`
- `_UIGestureRecognizerUpdateObserver`
- `PurpleEventCallback`
	- [关于PurpleEventCallback](http://stackoverflow.com/questions/12246627/what-is-purpleeventcallback) 
- `PurpleEventSignalCallback`
- `__IOMIGMachPortPortCallback`
- `_UIApplicationHandleEventQueue`
- `_ZN2CA11Transaction17observer_callbackEP19__CFRunLoopObservermPv`
- `FBSSerialQueueRunLoopSourceHandler`

## Further Reading

- [Apple Runloop Doc](https://developer.apple.com/library/ios/documentation/Cocoa/Conceptual/Multithreading/RunLoopManagement/RunLoopManagement.html#//apple_ref/doc/uid/10000057i-CH16-SW3)
- [Apple Open Source - CoreFoundation](http://www.opensource.apple.com/source/CF/)