---
layout: post
list_title: iOS中的一些小技巧 | Tips and Tricks in iOS
title: iOS中的一些小技巧 | Tips and Tricks in iOS
categories: [iOS]
---



<h3>Mac OS 卸载 MySQL</h3>

```
sudo rm /usr/local/mysql
sudo rm -rf /usr/local/mysql*
sudo rm -rf /Library/StartupItems/MySQLCOM
sudo rm -rf /Library/PreferencePanes/My*
vim /etc/hostconfig  (and removed the line MYSQLCOM=-YES-)
rm -rf ~/Library/PreferencePanes/My*
sudo rm -rf /Library/Receipts/mysql*
sudo rm -rf /Library/Receipts/MySQL*
sudo rm -rf /var/db/receipts/com.mysql.*
```

<h3>Rake Commands</h3>

[Answer](http://jonathanhui.com/rake-command)

<h3>解决10.9下cocoapods的bug</h3>

<a href="http://blog.cocoapods.org/Repairing-Our-Broken-Specs-Repository/">Answer</a>

<h3>使用appleDoc生成文档</h3>

<em>update @2013/11/16</em> 

- 首先代码注释要规范，通常以/**开头，以*/结尾。推荐xcode自动注释插件：<a href="https://github.com/onevcat/VVDocumenter-Xcode ">VVDocumenter</a>
- 从github上clone appleDoc工程：

```shell
git clone git://github.com/tomaz/appledoc.git
cd appledoc
sudo sh install-appledoc.sh
```

- 创建appleDoc的html文件，注意工程的绝对路径要带出来

```shell
appledoc --no-create-docset --output ./docNew --project-name xxx --project-company "xxx" --company-id "com.xxx.xxx" ~/Desktop/tbcity-mvc/xxx/Universal/Libraries/xxx/Core/
```

<h3>解决xcode找不到libtool问题</h3>

<em>update @2013/10/1 </em>

高版本的OSX，如果在app store上更新了Xcode，系统会把它当做普通的app，这个修改会导致和原来xcode的安装路径不一致。因此xcode会找到不到libtool。解决办法：<mark>将`usr/bin/libtool`考到xcode要求那个目录</mark>

<h3>Framework和静态库</h3>

- Framework和静态库的区别：

本质都是相同的，都是静态的代码。framework使用更方便，可以将头文件,资源文件等打包使用。

静态库的.a只是.m文件打包的二进制代码，因此，还需要自己将.h引入进来才能编译过，同样，资源文件也需要单独引入。

- xcode打Framework的方法：

使用开源项目:https://github.com/kstenerud/iOS-Universal-Framework

里面有两种方案：

1. fakeFramework ：思路是创建一个bundle，将bundle中的文件格式改成和framework相同的格式。
2. real Framework：运行一段脚本，给Xcode打补丁，重启xcode后即可看到有framework的模版出来。

- xcode打静态库的方法：

参见[Creating a Static Library in iOS Tutorial](http://www.raywenderlich.com/41377/creating-a-static-library-in-ios-tutorial)

<h3>NSString的hash问题</h3>

<em>update @2013/05/10 </em> 

我们知道如果两个object是equal的，那么他们的hash value一定相同。
但是hash value相同的两个object却不一定是equal的。NSString就是个例子：

> 当string长度长度大于96个字符时，NSString会为其生成相同的hash value

这种情况可以用sha1算法代替，[参考这里](http://www.abakia.de/blog/2012/12/05/nsstring-hash-is-bad/)

<h3>使用NSCache</h3>

<em>update @2012/12/5 </em> 

- 当系统内存紧张时，NSCache会自动释放一些资源
- 线程安全
- NSCache不会copy存入object的key

开源项目SDWebImage就是直接使用NSCache来缓存图片：

```objc
@interface SDImageCache ()

@property (strong, nonatomic) NSCache *memCache;
@property (strong, nonatomic) NSString *diskCachePath;
@property (SDDispatchQueueSetterSementics, nonatomic) dispatch_queue_t ioQueue;

@end
```

但实际项目中，为了查询方便，通常还会提供一个list来保存image的key:

```objc

@interface ETImageCache()<NSCacheDelegate>
{
    //mutable keyset
    NSMutableSet* _keySet;
    
    //internal memory cache
    NSCache* _memCache;
    
    // an async write-only-lock
    dispatch_queue_t _lockQueue;
}
```

基本上使用NSCache可以解决大部分的问题：你可以尽情的向cache中塞图片，当内存不足时，你可以选择手动释放掉NSCache中所有图片，也可以默认NSCache自己的策略：根据LRU规则释放掉最不活跃的图片。当app退到后台时，NSCache会自动释放掉图片，腾出空间给其它app。

<h3>Objective-C中的copy</h3>

UIKit和Foundation对象的copy都是shallow copy（浅拷贝）。比如UIImage：

```objc
UIImage* img = [UIImage imageNamed:@"pic.jpg"];
UIImage* img_copy = [img copy];
NSLog(@"%p,%p,%p,%p",img,img_copy,&img,&img_copy);
```

结果为：

```
0xc135b90,0xc135b90,0xbfffedc0,0xbfffedbc
```

他们指向的heap地址是相同的，他们各自在stack上的地址是不同的，相差4字节。

`mutableCopy`也是`shallow copy`。

对于自定义对象，比如`ETSomeItem:NSObject`，这种对象要实现`<NSCopy>`，对这种对象的`copy`操作为`deep copy`

### NSNull和Nil

```objc
NSString* null = [NSString stringWithFormat:@"%@",[NSNull null]];
NSString* nill = [NSString stringWithFormat:@"%@",nil];
```
输出为：

- `"<null>"`
- `"(null)"`

<h3>NSLog:C语言中的可变参数</h3>

<em>update @2012/07/05/</em>

NSLog的实现用到了C语言中的可变参数：`void NSLog(NSString *format, ...) `，我们可以自己实现一个NSLog：

 
```c
void mySBLog(NSString* format,...)
{
    va_list ap;
    va_start(ap, format);
    NSString* string = [[NSString alloc]initWithFormat:format arguments:ap];
    va_end(ap);
    printf("!![SBLog]-->!!%s \n",[string UTF8String]);
}
```

### 关于@Synchronized

@synchronized{...}相当于使用NSLock：

```objc
 @synchronized(self) {
  	NSLog(@"");
 }
```
相当于

```objc
[NSLock lock];
NSLog(@"");
[NSLock unlock]
```

### XCode编译C++

使用XCode编译C++头文件，例如`Header.h`:

```c++
namesapce TP{
}
```

如果直接编译会报错:`unknown type name namespace`

原因是XCode的`compile source AS`这一项指定的是:`According to File Type`，这意味着，XCode会根据文件类型编译源码，因此，`Header.h`不会作为C++代码去编译。

解决这个问题，有两个方法:

- 将`compile source AS`改成`Objective-C++`，这种方式将所有源码按照`OC++`编译
- 使用`__cplusplus`做条件编译，只有`C++/Objective-C++`类型的文件，才能编译`Header.h`：

将`Header.h`修改为：

```c++
#ifdef __cplusplus
extern "C"{
#endif
  namespace TP{
  }
    
#ifdef __cplusplus
 }
#endif
```
- 对于非`C++/Objective-C++`类型的文件:

```c++
#ifdef __cplusplus
#include "Header.h"
#endif
```

- 对于``C++/Objective-C++`类型的文件，直接import即可

例如在`ViewController.mm`中可以直接`import “Header.h”`

