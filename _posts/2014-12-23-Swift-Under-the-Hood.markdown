---
layout: post
list_title:  Swift中的Object研究 | Swift Object Under the hood
title: Swift中的Object
categories: [swift, iOS]
---

## Memory Layout

```c
class MySwiftClass
{
    let a : UInt64 = 10
    func method(){ }
}
```

我们首先定义一个类`MySwiftClass`，不指定其父类。接着我们想来观察这类的memory layout(x86_64)。

>所谓Memory Layout是指一个对象在内存中的存储形式。如果熟悉C++，那么对这个概念不会陌生，即使不熟悉C++，Objective-C对象在内存中也有固定的[存储形式](https://www.mikeash.com/pyblog/friday-qa-2009-03-13-intro-to-the-objective-c-runtime.html)。


```c
//creat an instance of MySwiftClass
let obj = MySwiftClass()
//get it's pointer in memory
var obj_ptr:UnsafePointer<Void> = unsafeAddressOf(obj)
//size of obj
let l:UInt = malloc_size(obj_ptr)
//memory layout of obj
let d = NSData(bytes: obj_ptr, length: (Int)(l));
println("%@",d)
```

- `obj`的大小为32byte
- `obj`在内存中的存放格式为`<100d8719 01000000 0c000000 01000000 0a000000 00000000 00000000 00000000>`

在x86_64上，数据按照little-endian存储，低位在前，高位在后，数据按照64bit对齐内存。前16字节为[isa pointer]():`100d8719 01000000 0c000000 01000000`。后16字节为成员变量`a`。

从这个结果来看Swift对象的内存格式和Objective-C对象是相同的。


## Swift Object

### `objc_class`在哪?

在Objective-C中，`isa`实际上是`objc_class`类型的结构体。因此，最先首先想到的办法当然是在Swift的各种framework中看看能否找到`objc_class`的定义。翻了半天，只找到了这样一句:

`/* Use 'Class' instead of 'struct objc_class *' */`

这是不是说明Swift的`objc_class`不存在呢？通过上一节的分析，应该可以确定`isa`是存在的，只是不允许被直接访问了。
网上有一篇很神奇的文章[Inside Swift](http://www.eswick.com/2014/06/inside-swift/)。他是通过逆向Mach-O文件，在`__objc_classlist`段，找到了`objc_class`

```c
struct objc_class {
    uint64_t isa;
    uint64_t superclass;
    uint64_t cache;
    uint64_t vtable;
    uint64_t data;
};
```
其中`vtable`能给我们带来一些启示，写过C++的人都知道，这个`vtable`叫做[虚表](http://en.wikipedia.org/wiki/Virtual_method_table)，用来在运行时决虚函数地址，但是Swift的`vtable`却和C++的`vtable`有着本质的却别，我们后面会详细解释这个问题。

### 对象间通信：

Swift作为一种静态语言，对象间通信是不需要使用Runtime的，这样也间接的[提升了性能](https://www.mikeash.com/pyblog/friday-qa-2014-07-04-secrets-of-swifts-speed.html)。但有一个问题，如果一个Swift对象需要和Objective-C对象通信怎么办？例如，有这样一段OC代码，它向`s`发消息:

```objc
//MySwiftClass是Swift的Class
MySwiftClass* s = [MySwiftClass new];
if([s responseToSelector(@selector(method))]){
	[s method];
}
```
此时`s`需要实现`responseToSelector`这样的方法，还需要检查`mehtod`这个方法是否存在。这说明`s`仍然需要具备在运行时introspect的能力，而我们又没有在`MySwiftClass`中定义任何OC的方法，这是怎么做到的呢？

### 神奇的SwiftObject

接着上面的问题，Swift对象为了和OC对象通信，必须要兼容OC的runtime，那么我们如何来验证呢？首先想到的就是在运行时反射出Swift对象的一些信息，由于上文已经创建好了`obj`：

```c
let obj = MySwiftClass()
```
我们接下来的任务就是反射出`obj`更多的信息，关于在Swift中如何拿到这些信息，Mike写了一个[非常牛逼的工具](https://github.com/mikeash/memorydumper/blob/master/memory.swift)，这个工具的思路是根据对象address和size，通过`dladdr`将里面的内容符号化。这份代码对于理解Swift和C有着很好的帮助。但是仅从实现这个功能来说，不需要那么复杂，我上传了一份比较精简的[代码](https://github.com/akaDealloc/blog/tree/gh-pages/code/swift/swift-basic/chap14-Runtime/chap14.playground)。总之，不论用哪种方法，都能得到下面的结果:


```
class:Optional("TestSwiftRuntime.MySwiftClass")
superclass:Optional("SwiftObject")
ivar:Optional("magic")  type:Optional("{SwiftObject_s=\"isa\"^v\"refCount\"q}")
ivar:Optional("a") type:Optional("")
property:Optional("hash") type:Optional("TQ,R")
property:Optional("superclass") type:Optional("T#,R")
property:Optional("description") type:Optional("T@\"NSString\",R,C")
property:Optional("debugDescription") type:Optional("T@\"NSString\",R,C")
method:Optional("zone") type:Optional("^{_NSZone=}16@0:8") 
method:Optional("doesNotRecognizeSelector:") type:Optional("v24@0:8:16") 
method:Optional("description") type:Optional("@16@0:8") 
method:Optional(".cxx_construct") type:Optional("@16@0:8") 
method:Optional("retain") type:Optional("@16@0:8") 
method:Optional("release") type:Optional("v16@0:8") 
method:Optional("autorelease") type:Optional("@16@0:8") 
method:Optional("retainCount") type:Optional("Q16@0:8") 
method:Optional("dealloc") type:Optional("v16@0:8") 
method:Optional("isKindOfClass:") type:Optional("B24@0:8#16") 
method:Optional("hash") type:Optional("Q16@0:8") 
method:Optional("isEqual:") type:Optional("B24@0:8@16") 
method:Optional("_cfTypeID") type:Optional("Q16@0:8") 
method:Optional("respondsToSelector:") type:Optional("B24@0:8:16") 
method:Optional("self") type:Optional("@16@0:8") 
method:Optional("performSelector:") type:Optional("@24@0:8:16") 
method:Optional("performSelector:withObject:") type:Optional("@32@0:8:16@24") 
method:Optional("conformsToProtocol:") type:Optional("B24@0:8@16") 
method:Optional("performSelector:withObject:withObject:") type:Optional("@40@0:8:16@24@32") 
method:Optional("isProxy") type:Optional("B16@0:8") 
method:Optional("isMemberOfClass:") type:Optional("B24@0:8#16") 
method:Optional("superclass") type:Optional("#16@0:8") 
method:Optional("class") type:Optional("#16@0:8") 
method:Optional("debugDescription") type:Optional("@16@0:8") 
```

我们先来分析一下上面的信息：

- 类名：运行时`obj`的类名变成了`TestSwiftRuntime.MySwiftClass`。格式为: `Target名称.类名`。也许有人会注意到，上面的结果和[这篇文章](https://www.mikeash.com/pyblog/friday-qa-2014-07-18-exploring-swift-memory-layout.html)得到的结果不一样，它得到类名是混淆过的结果：`_TtC16TestSwiftRuntime12MySwiftClass`。产生这个问题的原因是使用API的差别:`String(UTF8String:ptr)`和`String.fromCString(ptr)`，但这并不是根本原因，根本原因是[Name Mangling](http://en.wikipedia.org/wiki/Name_manglin)。

- 父类：`SwiftObject`，貌似一个新OC基类出现了，实现了`<NSObject>`，因此具备了和其它OC对象通信的能力，但是，和`NSObject`不同的是，它还有一个成员变量叫`magic`，从TypeEncoding的结果来看，它是一个结构体，有两个成员:`isa`，`refCount`。功能上貌似是用来做引用计数。更多关于`magic`的内容还有待研究。
	
- 关于成员变量`a`：我们为`MySwiftClass`定义了一个`UInt32`类型的成员`a`，但是我们却无法获取它的TypeEncoding，为什么呢？这说明Objective-C的runtime是无法获取到Swift变量的类型的。

- 关于成员方法: 由于`SwiftObject`实现了`<NSObject>`，因此它实现了一大堆OC的方法。

- 关于成员方法`method()`：我们发现`method()`根本没有被反射出来，原因在上一节末尾也提到了，Swift对象的method被放到了`vtable`里，而Objective-C的运行时是无法发现`vtable`的。

能输出这样的结果意味着，这些信息可以被Objective-C在运行时发现，所以我们的Swift对象:`obj`

1. 要么自己就是一个Objective-C对象：这点显然不可能了，因为`objc_class`的结构是不同的。

2. 要么它自己重新实现了上面的方法：这种可能性比较大，按照我的理解，`SwiftObject`为了和OC对象通信，对OC所有runtime API的函数原型做了不同的实现，例如上面的例子`[s method]`,实际上是`objc_msgSend(s,@selector(method),nil)`，系统首先会判断`s`的`objc_class`类型，如果是Swift对象，则在`s`的`vtable`中找到`method`的地址然后执行。如果是OC对象，则还是老一套。

### 寻找vtable

接下来我们讨论两个Swift对象的通信，上面已经提到，Swift对象的通信基本上是靠`vtable`，但是`vtable`有无法用OC的runtime反射出来，因此找到`vtable`是个比较艰难的问题，这时候，一种办法是依靠Mike的牛逼的工具：

```c
let obj = MySwiftClass()
var obj_ptr:UnsafePointer<Void> = unsafeAddressOf(obj)
dumpmem(obj_ptr)
```
`dumpmem`会在运行时dump出`objc_class`中所有的符号：

```
...
Symbol _TFC16TestSwiftRuntime12MySwiftClassg1aVSs6UInt64
Symbol _TFC16TestSwiftRuntime12MySwiftClasss1aVSs6UInt64
Symbol _TFC16TestSwiftRuntime12MySwiftClassm1aVSs6UInt64
Symbol _TFC16TestSwiftRuntime12MySwiftClass6methodfS0_FT_T
Symbol _TFC16TestSwiftRuntime12MySwiftClasscfMS0_FT_S0
...
```

我们根据Name mangling规则找到了`obj`的成员方法，先忽略这些诡异的符号，我们后面会详细解释name mangling的问题。从上到下依次为：

- `a`的`setter`方法
- `a`的`getter`方法
- 一个未知的`m1a`方法？
- `a`的`method`方法
- `a`的`init`方法

由于这些方法编译器是提前编译好的，因此，还有一种dump 符号的方法是通过查看目标文件的符号表，打开命令行输入:

```
xcrun swiftc -emit-library -o TestSwiftRuntime -
class MySwiftClass{

var a:UInt64 = 10
func method(){}

}

^D

```

此时会在当前目录下生成一个TestSwiftRuntime的目标文件:`TestSwiftRuntime`


```
 xcrun nm -g TestSwiftRuntime 
```

可以得到和上面相同的结果

```
00000000000015a0 T __TFC16TestSwiftRuntime12MySwiftClass6methodfS0_FT_T_
0000000000001680 T __TFC16TestSwiftRuntime12MySwiftClassCfMS0_FT_S0_
00000000000015c0 T __TFC16TestSwiftRuntime12MySwiftClassD
0000000000001660 T __TFC16TestSwiftRuntime12MySwiftClasscfMS0_FT_S0_
00000000000015b0 T __TFC16TestSwiftRuntime12MySwiftClassd
00000000000015f0 T __TFC16TestSwiftRuntime12MySwiftClassg1aVSs6UInt64
0000000000001630 T __TFC16TestSwiftRuntime12MySwiftClassm1aVSs6UInt64
0000000000001610 T __TFC16TestSwiftRuntime12MySwiftClasss1aVSs6UInt64

```

### Name Mangling的规则

关于Name mangling，wikipedia上面的解释非常详细，读一遍基本就能明白了。这东西主要是编译器为了区分同名的数据结构设计的一种编码方式，例如下面代码:

```
int foo(int a) { return a * 2; }    
int foo(double a) { return a * 2.0; }

int main() { return foo(1) + foo(1.0); }

```

编译器为了区别这两个方法，需要为他们生成不同的signature，生成的方法就是通过Name mangling。
上面的代码，我们使用C++的编译器来编译：

```
0000000100000f30 T __Z3food
0000000100000f10 T __Z3fooi
0000000100000000 T __mh_execute_header
0000000100000f60 T _main
                 U dyld_stub_binder

```

可以看到编译器为两个`foo`方法生成了不同的signature。关于C++编译器Name Mangling的规则可以参考[这里](http://mentorembedded.github.io/cxx-abi/abi.html#mangling)。

对于Swift，采用了和C++类似的规则,我们以
`__TFC16TestSwiftRuntime12MySwiftClass6methodfS0_FT_T_`
这个方法为例：

1. Swift中所有的符号都以`_`开头。

2. `_T` 用来标识这个符号是全局的。

3. `F` 用来标识这个符号代表的是一个函数。

4. `C` 代表它是从属于一个类的方法。

5. `16TestSwiftRuntime`代表`module name`，16为字符长度。

6. `12MySwiftClass`代表类名，12为字符长度。

7. `6method`代表方法名称，6为字符长度。

8. `f`代表这个方法是`uncurried function`，第一个参数是一个隐式参数`self`。

9. `S0_FT_T_`不确定，按照规律来看应该是用来标识入参和返回值的。


demangling可以用下面的命令：

```

xcrun swift-demangle __TFC16TestSwiftRuntime12MySwiftClass6methodfS0_FT_T_

//_TFC16TestSwiftRuntime12MySwiftClass6methodfS0_FT_T_ ---> TestSwiftRuntime.MySwiftClass.method (TestSwiftRuntime.MySwiftClass)() -> ()

```

或者用下面API：

```c

//name mangling:
println(_stdlib_getTypeName(obj)) //_TtC16TestSwiftRuntime12MySwiftClass

//demangling:
println(_stdlib_demangleName(_stdlib_getTypeName(obj))) //TestSwiftRuntime.MySwiftClass

```


## 更多

我们上面研究了Swift中对象的内存模型和类结构，而Swift中还有很多非对象类型，比如Struct，Optional对象，继承了NSObject的Swift对象，等等...它们有着各自有趣的特性。这方面Mike做了非常详细的分析和研究，感兴趣的可以直接去阅读他的文章。


## 总结

最后我们再来梳理一遍上面的内容：

- 首先我们有一个Swift对象：`obj`，他没有任何父类，我们通过查看它我在内存中的布局，发现了`isa`。

- 然后我们通过OC的运行时，发现`obj`有一个父类`SwiftObject`，它实现了`<NSObject>`的接口，使`obj`有了和OC对象通信的能力

- 然后通过查看目标代码的符号表，找到了`vtable`里面的方法。

- 最后我们解释了Name Mangling。


## Further Reading

- [Swift funtime](https://realm.io/news/altconf-boris-bugling-swift-funtime/)
- [Inside Swift](http://www.eswick.com/2014/06/inside-swift/)
- [Mike's Blog](https://www.mikeash.com/pyblog/)




