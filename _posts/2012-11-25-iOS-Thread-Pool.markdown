---
layout: post
list_title: 理解iOS中的线程池 | Thread Management in iOS
title: 理解iOS中的线程池
categories: [iOS]
---

在 GCD 和 NSOperationQueue 之前，iOS 使用线程一般是用 NSThread，而 NSThread 是对<a title="POSIX THREAD" href="http://en.wikipedia.org/wiki/POSIX_Threads">POSIX thread</a>的封装,也就是 pthread，本文最后会面附上一段使用 pthread 下图片的代码，现在我们还是继续上面的讨论。使用 NSThread 的一个最大的问题是：直接操纵线程，线程的生命周期完全交给 developer 控制，在大的工程中，模块间相互独立，假如 A 模块并发了 8 条线程，B 模块需要并发 6 条线程，以此类推，线程数量会持续增长，最终会导致难以控制的结果。

GCD 和 NSOperationQueue 出来以后，developer 可以不直接操纵线程，而是将所要执行的任务封装成一个 unit 丢给线程池去处理，线程池会有效管理线程的并发，控制线程的生命周期。因此，现在如果考虑到并发场景，基本上是围绕着 GCD 和 NSOperationQueue 来展开讨论。GCD 是一种轻量的基于 block 的线程模型，使用 GCD 一般要注意两点：一是线程的 priority，二是对象间的循环引用问题。NSOperationQueue 是对 GCD 更上一层的封装，它对线程的控制更好一些，但是用起来也麻烦一些。关于这两个孰优熟劣，需要根据具体应用场景进行讨论：<a title="GCD vs NSOperation" href="http://stackoverflow.com/questions/10373331/nsoperation-vs-grand-central-dispatch">stackoverflow:GCD vs NSopeartionQueue</a>。

我们后面会以下载图片为例，首先来分析在非并发的情况下 NSOperationQueue 和 GCD 的用法和特性，然后分析在并发的情况下讨论 NSOperationQueue 对线程的管理。

> 测试图片来自[这里](http://www.collegedj.net/wp-content/uploads/)

### 异步下载

先从 NSOperationQueue 最简单的用法开始：

```c++
_opQueue = [[NSOperationQueue alloc]init];
[_opQueue addOperationWithBlock:^{
        NSData* data = [NSData dataWithContentsOfURL:[NSURL URLWithString:url3]];
        //_imgv3.image = [UIImage imageWithData:data];

        [[NSOperationQueue mainQueue] addOperationWithBlock:^{
            _imgv3.image = [UIImage imageWithData:data];
        }];
    }];
```

这种写法和使用 GCD 相比没任何优势，使用 GCD 代码写起来还更顺手：

```c++
- (void)downloadWithGCD
{
    dispatch_async(_gcdQueue, ^{
        NSData* data = [NSData dataWithContentsOfURL:[NSURL URLWithString:url4]];
        dispatch_async(dispatch_get_main_queue(), ^{
            self.imgv4.image = [UIImage imageWithData:data];
             NSLog(@"done downloading 3rdimage ");
        });
    });
}
```

## 线程同步

接下来我们再对比一下 GCD 和 NSOperation 对线程的控制性，假设我们有两张图要下载，第二张要在第一张完成后再去下载，显然这是一个线程同步的问题，我们先用 NSOperation 来实现

```c++
- (void)downloadWithNSOperationDependency
{
    ETOperation* op1 = [ETOperation new];
    op1.url = [NSURL URLWithString:url3];
    __weak ETOperation* _op1 = op1;
    [op1 setCompletionBlock:^{
        _imgv3.image = _op1.image;
    }];
    [op1 start];

    ETOperation* op2 = [ETOperation new];
    op2.url = [NSURL URLWithString:url4];
    __weak ETOperation* _op2 = op2;
    [op2 setCompletionBlock:^{
        _imgv4.image = _op2.image;
    }];
    [op2 addDependency:op1];
    [op2 start];
}
```

上述代码中`op2`将在`op1`执行完成后执行。接下来我们使用 GCD 来完成同样的任务，仅就上面的 case 来说，使用 GCD 有很多种方式，比如最常用的就是使用一个串行队列，当第一个下载 block 执行完后在启动第二个 block 进行下载，这种方式太过简单，这里就不做过多介绍，下面给出一种使用`dispatch_group`的方式，这种方式略显笨拙，但是可以展示如何使用 GCD 来做线程同步

```c++
- (void)downloadWithGCDGroups
{
    dispatch_group_t group = dispatch_group_create();
    dispatch_queue_t queue = dispatch_get_global_queue(0, 0);
    dispatch_group_async(group, queue, ^(){
        NSData* data = [NSData dataWithContentsOfURL:[NSURL URLWithString:url3]];
        dispatch_group_async(group, dispatch_get_main_queue(), ^(){
            self.imgv3.image = [UIImage imageWithData:data];
        });
    });
    // This block will run once everything above is done:
    dispatch_group_notify(group, dispatch_get_main_queue(), ^(){
        dispatch_async(_gcdQueue, ^{
            NSData* data = [NSData dataWithContentsOfURL:[NSURL URLWithString:url4]];
            dispatch_async(dispatch_get_main_queue(), ^{
                self.imgv4.image = [UIImage imageWithData:data];
            });
        });
    });
}
```

在实际项目中，如果是两个线程之间的同步问题，我们不会书类似写上面的代码。实际上`dispatch_group`的作用在于控制多个线程并发，并为这些线程提供一个线程同步点，即当`group`内的所有线程都执行完成后，再通知外部(类似 Java 中线程的`join`操作)。因此，通常情况下，`dispatch_group`的用法如下：

```c++
dispatch_queue_t queue = dispatch_get_global_queue( 0, 0 );
dispatch_group_t group = dispatch_group_create();

//run task #1
dispatch_group_enter(group);
dispatch_async( queue, ^{
    NSLog( @"task 1 finished: %@", [NSThread currentThread] );
    dispatch_group_leave(group);
} );

//run task #2
dispatch_group_enter(group);
dispatch_async( queue, ^{
    NSLog( @"task 2 finished: %@", [NSThread currentThread] );
    dispatch_group_leave(group);
} );

//sychronization point
dispatch_group_notify( group, queue, ^{
    NSLog( @"all task done: %@", [NSThread currentThread] );
} );
```

如果考虑控制线程，相比 GCD 来说 NSOperation 是个更好的选择，它提供了很多 GCD 没有的高级用法：

1. Operation 之间可指定依赖关系
2. 可指定每个 Operation 的优先级
3. 可以 Cancel 正在执行的 Operation
4. 可以使用 KVO 观察对任务状态：`isExecuteing`、`isFinished`、`isCancelled`

### NSOperationQueue 与线程池

下面我们在来观察并发的情况，这也是今天重点要讨论的。我们先从`NSOperationQueue`的并发模型开始：

这里是 apple 关于并发 NSOperationQueue 的[Guideline]("https://developer.apple.com/library/mac/documentation/general/conceptual/concurrencyprogrammingguide/OperationObjects/OperationObjects.html#//apple_ref/doc/uid/TP40008091-CH101-SW1");

总结一下，要点有这么几条：

1. 如果要求 concurrent，那么 NSOperation 的生命周期要自己把控
2. 并发的 operation 要继承 NSOperation 而且必须 override 这几个方法：
   - `start`，`isExecuting`，`isFinished`，`isConcurrent`
3. 复写`isExecuting`和`isFinished`要求：
   - 线程安全
   - 手动出发 kvo 通知

满足这三点，就可以使用 NSOperationQueue 并发了，我们先按照上面的要求创建一个`NSOperation`：

```c++
@interface MXOperation : NSOperation
{
    NSString*   _threadName;
    NSString*   _url;
    BOOL        executing;
    BOOL        finished;
}

@end

@implementation MXOperation

- (id)initWithUrl:(NSString*)url name:(NSString*)name;
{
    self = [super init];

    if (self) {
        if (name!=nil)
        _threadName = name;
        _url = url;
        executing = NO;
        finished = NO;

    }
    return self;
}

- (BOOL)isConcurrent {
    return YES;
}

- (BOOL)isExecuting {
    return executing;
}

- (BOOL)isFinished {
    return finished;
}

- (void)start
{
    [NSThread currentThread].name = _threadName;
    currentThreadInfo(@"start");

    if ([self isCancelled])
    {
        // Must move the operation to the finished state if it is canceled.
        [self willChangeValueForKey:@"isFinished"];
        finished = YES;
        [self didChangeValueForKey:@"isFinished"];
        return;
    }

    // If the operation is not canceled, begin executing the task.
    [self willChangeValueForKey:@"isExecuting"];

    executing = YES;

    //下载图片
    [NSData dataWithContentsOfURL:[NSURL URLWithString:_url]];

    //完成下载
    [self completeOperation];

    [self didChangeValueForKey:@"isExecuting"];
}

- (void)completeOperation {
    [self willChangeValueForKey:@"isFinished"];
    [self willChangeValueForKey:@"isExecuting"];

    executing = NO;
    finished = YES;

    [self didChangeValueForKey:@"isExecuting"];
    [self didChangeValueForKey:@"isFinished"];
}

- (void)dealloc
{
    dumpThreads(@"dealloc");
}

@end
```

首先我们按照要求完成了`MXOperation`的并发代码。其次我们在`start`的方法中，给当前线程增加了 name，方便观察。然后使用`NSData`去下载图片，图片下载完成后通过`KVO`通知`OperationQueue`任务完成。最后我们在`delloc`的方法中，观察当前`active`的线程情况。

`currentThreadInfo`和`dumpThreads`两个工具函数，涉及到了 kernel 的一些 API，作用是用来查看当前线程的状态：

```c++
static inline void currentThreadInfo(NSString* str)
{
    if (str)
        NSLog(@"---------%@----------",str);

    NSThread* thread = [NSThread currentThread];
    mach_port_t machTID = pthread_mach_thread_np(pthread_self());
    NSLog(@"current thread num: %x thread name:%@", machTID,thread.name);

    if (str)
        NSLog(@"-------------------");
}


static inline void dumpThreads(NSString* str) {

    NSLog(@"---------%@----------",str);
    currentThreadInfo(nil);
    char name[256];
    thread_act_array_t threads = NULL;
    mach_msg_type_number_t thread_count = 0;
    task_threads(mach_task_self(), &amp;threads, &amp;thread_count);
    for (mach_msg_type_number_t i = 0; i &lt; thread_count; i++) {
        thread_t thread = threads[i];
        pthread_t pthread = pthread_from_mach_thread_np(thread);
        pthread_getname_np(pthread, name, sizeof name);
        NSLog(@"mach thread %x: getname: %s", pthread_mach_thread_np(pthread), name);
    }
    NSLog(@"-------------------");
}
```

然后我们来并发下载 4 张图片，图片大小在 100kb 左右：

```c++
// Do any additional setup after loading the view, typically from a nib.
   NSArray* urls = @[@"http://www.collegedj.net/wp-content/uploads/2010/10/6.jpg",
                     @"http://www.collegedj.net/wp-content/uploads/2010/10/Rihanna.jpg",
                     @"http://www.collegedj.net/wp-content/uploads/2010/10/chris-brown.jpg",
                     @"http://www.collegedj.net/wp-content/uploads/2010/10/dj_scary.jpg",
                   ];
   _queue = [NSOperationQueue new];
   for (int i=0; i<urls.count; i++)
   {
       MXOperation* operation = [[MXOperation alloc]initWithUrl:urls[i] name:[NSString stringWithFormat:@"%d",i]];
       [_queue addOperation:operation];
   }

```

观察日志输出:

```shell
---------start----------
---------start----------
---------start----------
---------start----------
current thread num: 1403 thread name:0
current thread num: 3307 thread name:1
current thread num: 3603 thread name:2
current thread num: 3703 thread name:3
-------------------
-------------------
-------------------
-------------------
---------dealloc----------
current thread num: 1403 thread name:0
mach thread a0b: getname:
mach thread d03: getname:
mach thread 1403: getname: 0
mach thread 3307: getname: 1
mach thread 3603: getname: 2
mach thread 3703: getname: 3
mach thread 3f03: getname: com.apple.NSURLConnectionLoader
mach thread 4007: getname:
mach thread 4707: getname:
mach thread 6203: getname:
mach thread 6303: getname: com.apple.CFSocket.private
-------------------
---------dealloc----------
current thread num: 3307 thread name:1
mach thread a0b: getname:
mach thread d03: getname:
mach thread 1403: getname: 0
mach thread 3307: getname: 1
mach thread 3603: getname: 2
mach thread 3703: getname: 3
mach thread 3f03: getname: com.apple.NSURLConnectionLoader
mach thread 4007: getname:
mach thread 4707: getname:
mach thread 6203: getname:
mach thread 6303: getname: com.apple.CFSocket.private
-------------------
---------dealloc----------
current thread num: 3603 thread name:2
mach thread a0b: getname:
mach thread d03: getname:
mach thread 1403: getname: 0
mach thread 3307: getname: 1
mach thread 3603: getname: 2
mach thread 3703: getname: 3
mach thread 3f03: getname: com.apple.NSURLConnectionLoader
mach thread 4007: getname:
mach thread 4707: getname:
mach thread 6203: getname:
mach thread 6303: getname: com.apple.CFSocket.private
-------------------
---------dealloc----------
current thread num: 3703 thread name:3
mach thread a0b: getname:
mach thread d03: getname:
mach thread 1403: getname: 0
mach thread 3307: getname: 1
mach thread 3603: getname: 2
mach thread 3703: getname: 3
mach thread 3f03: getname: com.apple.NSURLConnectionLoader
mach thread 4007: getname:
mach thread 4707: getname:
mach thread 6203: getname:
mach thread 6303: getname: com.apple.CFSocket.private
-------------------
```

有种眼花缭乱的感觉，静下心慢慢看：

1. 我们首先并发了 4 个线程：

   ```shell
   ---------start----------
   current thread num: 1403 thread name:0
   ---------start----------
   current thread num: 3307 thread name:1
   ---------start----------
   current thread num: 3603 thread name:2
   ---------start----------
   current thread num: 3703 thread name:3
   ```

线程`id`是系统分配的，线程名字是我们自定义的，用`0-3`去标识

2. 然后，图片下载完成后，`MXOperation`被释放掉：

   ```shell
   ---------dealloc----------
   current thread num: 1403 thread name:0
   mach thread a0b: getname:
   mach thread d03: getname:
   mach thread 1403: getname: 0
   mach thread 3307: getname: 1
   mach thread 3603: getname: 2
   mach thread 3703: getname: 3
   mach thread 3f03: getname: com.apple.NSURLConnectionLoader
   mach thread 4007: getname:
   mach thread 4707: getname:
   mach thread 6203: getname:
   mach thread 6303: getname: com.apple.CFSocket.private
   ```

这个时候可以看到：当前线程`id`为`1403`，我们标识其为 0 号，同时存在的线程还有 1，2，3 和一些包括主线程在内，获取不到名字的线程。然后有两个是网络请求的线程。

到目前为止，结果复合预期，没什特别的地方。接着我们再下载两张图：

```objc
NSArray* urls = @[ @"http://www.collegedj.net/wp-content/uploads/2010/10/3-150x150.jpg",
                   @"http://www.collegedj.net/wp-content/uploads/2010/10/3-300x199.jpg"];

 for (NSString* url in urls)
 {
     MXOperation* operation = [[MXOperation alloc]initWithUrl:url name:nil];
     [_queue addOperation:operation];
 }
```

注意，这里并没有指定其 name，我们要验证 thread id，观察日志输出结果：

```shell
--------again-----------
---------start----------
---------start----------
current thread num: 1403 thread name:
current thread num: 3307 thread name:
-------------------
-------------------
---------dealloc----------
current thread num: 1403 thread name:
mach thread a0b: getname:
mach thread d03: getname:
mach thread 1403: getname:
mach thread 3307: getname:
mach thread 3603: getname: 2
mach thread 3703: getname: 3
mach thread 3f03: getname: com.apple.NSURLConnectionLoader
mach thread 4007: getname:
mach thread 4707: getname:
mach thread 6203: getname:
mach thread 6303: getname: com.apple.CFSocket.private
mach thread 6903: getname:
mach thread 6a03: getname:
mach thread 6b03: getname:
-------------------
---------dealloc----------
current thread num: 3307 thread name:
mach thread a0b: getname:
mach thread d03: getname:
mach thread 1403: getname:
mach thread 3307: getname:
mach thread 3603: getname: 2
mach thread 3703: getname: 3
mach thread 3f03: getname: com.apple.NSURLConnectionLoader
mach thread 4007: getname:
mach thread 4707: getname:
mach thread 6203: getname:
mach thread 6303: getname: com.apple.CFSocket.private
mach thread 6903: getname:
mach thread 6a03: getname:
mach thread 6b03: getname:
-------------------
```

我们发现重新下载的两条线程 `id` 分别为：

```shell
---------start----------
current thread num: 1403 thread name:
---------start----------
current thread num: 3307 thread name:
```

这说明 `NSOperationQueue` 的线程池起了作用，`1403` 和 `3307` 线程在下载完后，率先进入休眠状态，有新任务来时，两条线程再次被唤醒，而不是重新再起线程。为了验证这个的判断，我们来改一下 `NSOperationQueue` 的并发数：

```c++
_queue.maxConcurrentOperationCount = 3;

```

然后我们下载 6 张图：

```c++
NSArray* urls = @[    @"http://www.collegedj.net/wp-content/uploads/2010/10/1-150x150.jpg",
                      @"http://www.collegedj.net/wp-content/uploads/2010/10/Rihanna.jpg",
                      @"http://www.collegedj.net/wp-content/uploads/2010/10/chris-brown.jpg",
                      @"http://www.collegedj.net/wp-content/uploads/2010/10/dj_scary.jpg",
                      @"http://www.collegedj.net/wp-content/uploads/2010/10/3-150x150.jpg",
                      @"http://www.collegedj.net/wp-content/uploads/2010/10/3-300x199.jpg"
                    ];
```

再次观察日志:

```
---------start----------
---------start----------
---------start----------
current thread num: 1403 thread name:0
current thread num: 3307 thread name:1
current thread num: 3603 thread name:2
-------------------
-------------------
-------------------
---------start----------
current thread num: 1403 thread name:3
-------------------
---------start----------
current thread num: 3307 thread name:4
-------------------
---------dealloc----------
---------start----------
current thread num: 3603 thread name:2
current thread num: 3d07 thread name:5

```

由于并发数为 3，率先有 3 条线程并发出去，由于 1403 最先去下载，我们特意为其安排一个 150x150 的小图，其下载完成后，立刻处于休眠状态，然后第 4 张图片被下载，1403 又被唤醒，同理，3307 也是相同的情况。而当第 6 张图要去下载时，我们看到是一条新的线程 id 为 3d07，不在 1403，3307，3603 之内。这说明线程池当前没有可调度的线程了，只好创建一个新线程。

最后，我们图片全部下载完，清空这个线程池：

```c++
[_queue cancelAllOperations];
    _queue = nil;
    dumpThreads(@"finish");
```

结果为：

```
---------finish----------
current thread num: a0b thread name:
mach thread a0b: getname:
mach thread d03: getname:
mach thread 3c03: getname: com.apple.NSURLConnectionLoader
mach thread 5b03: getname: com.apple.CFSocket.private
-------------------
```

这个结果也在我们意料之中，所有线程池创建的线程全被销毁，只留下一个主线程，一个不知道名字的线程，两个网络请求的线程。

## Resource

//附：pthread 代码：

```c
//使用ptrhead
struct threadInfo {
    unsigned char* url;
    size_t count;
};

struct threadResult {

    unsigned char* imageRawData;
    unsigned short int imageLength;
};

void * downloadImage(void *arg)
{
    struct threadInfo const * const info = (struct threadInfo *) arg;

    unsigned char* url = info-&gt;url;

    NSURL* nsUrl  =[ NSURL URLWithString:[NSString stringWithUTF8String:(char*)url]];
    NSData* data = [NSData dataWithContentsOfURL:nsUrl];

    struct threadResult * const result = (struct threadResult *) malloc(sizeof(*result));
    result -&gt; imageRawData = (unsigned char*)data.bytes;
    result -&gt; imageLength = data.length;

    return result;
}

```

...

```objc

- (void)downloadImageWithPthread
{
    //线程参数结构体
    struct threadInfo * const info = (struct threadInfo *) malloc(sizeof(*info));
    info-&gt;url = (unsigned char*)url1.UTF8String;

    //
    pthread_t tid;
    int err_create = pthread_create(&amp;tid, NULL, &amp;downloadImage, info);
    NSCAssert(err_create == 0, @"pthread_create() failed: %d", err_create);

    // Wait for the threads to exit:
    struct threadResult * results;

    int err_join = pthread_join(tid, (void**)(&amp;results));
    NSCAssert(err_join == 0, @"pthread_join() failed: %d", err_join);

    NSData* imgData = [NSData dataWithBytes:results-&gt;imageRawData length:results-&gt;imageLength];
    _imgv1.image = [UIImage imageWithData: imgData];

    free(results);
    results = NULL;
}
```
