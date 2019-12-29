---
layout: post
list_title: 链接与加载 | Linkers and Loaders | 静态链接与静态库 | Static Library 
title: 静态链接与静态库
categories: [C,C++]
---

### 静态库

无论是动态库还是静态库都是为了解决代码重用的问题。静态库可以理解为一系列目标文件的集合，link的时候静态库中的symbol会和目标文件中的symbol一起链接，如下图所示

<img src="{{site.baseurl}}/assets/images/2015/07/static-linking.png">

这种方式的好处是简单粗暴，静态库中的代码和目标工程中的代码一起编译连接，linker可以做全局的symbol级别的optimization，比如strip掉dead code等。使用静态库的劣势在于它会增加binary的大小，更重要的是如果静态库中的代码更新了，整个工程需要重新编译，不够灵活。

### 静态链接与`-dead_strip`

静态链接相对来说比较简单，如上图中展示了一个静态库被链接进一个executable的全过程，对于这种情况，binary中最终只会链接静态库中被用到的symbols，如下图所示

<img src="{{site.baseurl}}/assets/images/2015/07/static-linking-selectiveness.png">

上图中，假设我们的binary只需要三角形和菱形两个symbol，这时候如果linker开启了优化模式，即使静态库中有多个symbol，最终被链接进来的也只有这两个。这也是为什么静态库的size很大，但最终的binary的size却很小的原因。为了加深理解，我们可以看下面的代码

<div class="highlight md-flex-h md-margin-bottom-24">
<div>
<pre class="highlight language-cpp md-no-padding-v md-height-full">
<code class="language-cpp">
//a.cpp
int __attribute__((noinline)) a_foo() { 
    int buf[5000]; 
    return 1; 
}
int __attribute__((noinline)) a_bar() { 
    int buf[5000]; 
    return 1; 
}
</code>
</pre>
</div>
<div class="md-margin-left-12">
<pre class="highlight language-cpp md-no-padding-v md-height-full">
<code class="language-cpp">
//main.cpp
extern int a_foo();
int main(){
    int x = a_foo();
    std::cout<<x<<std::endl;
    return 0;
}
</code>
</pre>
</div>
</div>

上面代码中，`a.cpp`包含了两个函数，`a_foo`和`a_bar`，而`main.cpp`中只用到了`a_foo`，而且`a_foo`是被声明的，main函数并不知道去哪里找这个函数。接着我们分别编译这两个文件，得到各自的目标文件

```shell
> clang -static -c main.cpp //main.o
> clang -static -c a.cpp //a.o
```
此时我们观察两个目标文件中的symbol

<div class="highlight md-flex-h md-margin-bottom-24">
<div>
<pre class="highlight language-cpp md-no-padding-v md-height-full">
<code class="language-cpp">
>  nm a.o | c++filt

00000050 T a_bar()
00000000 T a_foo()
                 U ___stack_chk_fail
                 U ___stack_chk_guard
</code>
</pre>
</div>
<div class="md-margin-left-12">
<pre class="highlight language-python md-no-padding-v md-height-full">
<code class="language-python">
> nm main.o | c++filt

000001e0 short GCC_except_table3
                 U __Unwind_Resume
                 U a_foo()
...
000000a0 T _main

</code>
</pre>
</div>
</div>

观察目标文件中的symbol，其中`main.o`中的`a_foo`标记为`U`，符合我们的预期。接着我们手动的将这两个目标文件link起来产生最终的binary `a.out`，我们在MacOS下使用默认的linker - `ld`

```shell
ld -o a.out main.o a.o -lc++ -L/usr/local/lib -lSystem
```
> 也可以直接 `clang++ main.cpp a.cpp`

接着我们查看`a.out`中的符号

```shell
> nm a.out | c++filt

100000f68 short GCC_except_table3
                 ...
100000e70 T a_bar()
100000e20 T a_foo()
...
100000c40 T _main
                 U dyld_stub_binder

```
我们发现，`a.o`中没用的`a_bar`也被link进来了，这显然是我们不希望看到的，此时我们可以通过`-dead_strip`来告诉`ld` strip掉无用代码

```shell
ld -o a.out main.o a.o -lc++ -L/usr/local/lib -lSystem -dead_strip
```
> 也可以直接 `clang++ main.cpp a.cpp -Wl,-dead_strip`

此时我们再查看`a.out`的符号表则会发现`a_bar()`已经不在了。

### `ld64.lld`

自然而然的我们会想否可以将上面的linker优化技术应用到静态库上，即给你一个很大的静态库，是否可以通过linker的帮助来裁剪掉无用的代码。为了回答这个问题我们要想一下`-dead_strip`是怎么工作的。显然对于每个executable都有一个`main()`函数，这个main函数是整个应用程序的entry point，也就是说我们可以从main函数中用到的symbols出发来trace所有用到的symbol并把他们记录下来，然后strip掉那些没有用的symbol。比如上面例子中main函数中发现了`a_foo`，是一个undefined symbol，这时linker再去寻找`a_foo`，而`a_foo`也是一个函数，它又用了别的symbol，通过这样的不断搜索，便可以找出所有用到的symbol。

回到静态库问题上，通常对于静态库，我们会提供public APIs，这些API即可作为我们的entry point，作为trace的起点。接下来的问题是，我们需要一个linker来帮我们完成trace + dead_code strip。不幸的是，MacOS上默认的`ld`不能strip目标文件，`-dead_strip`只对executable或者动态库有效。由于静态库只是目标文件的合集，我们需要一种linker可以帮我们strip 目标文件 - `ld64.lld`。

> 需要注意的是ld64.lld目前已经处于不被维护的状态，请慎重使用

`ld64.lld`是LLVM toolchain里的一种linker，使用它我们需要自行编译LLVM

```shell
> git clone https://github.com/llvm/llvm-project.git
> mkdir build-release && cd build-release
> cmake -G Ninja ../llvm -DCMAKE_BUILD_TYPE=Release -DLLVM_ENABLE_PROJECTS="clang;lld"
> ninjia
```
有了ld64.lld之后，我们需要先为其提供一个`symbol list`，里面包含我们要保留的symbol。还是上面的例子，假如我们要保留`a_foo`，则我们可以用下面命令

```shell
> echo "__Z5a_foov" > exported.syms
> LD=~/LLVM/build-release/bin/ld64.lld
> $LD -dead_strip -o a_lite.o -exported_symbols_list ./exported.syms -r a.o
```
此时得到的`a_lite.o`只保留了`a_foo`。可以看到`ld64.lld`和`ld`不同的地方在于，它支持`-r`，即可以将目标文件作为`-dead_strip`的输入，同时输出可以是一个monolithic 目标文件。


### vtable的问题

虽然`-dead_strip`可以帮我们strip掉无用代码，但它却不是万能的，对于虚函数，它貌似无能为力

```cpp
//a.cpp
#include <iostream>
class A{
public:
    virtual void bark(){
        std::cout << "from A" << std::endl;
    }
    void print();
};
//main.cpp
#include "./a.h"
int main(){
    A b;
    b.print();
    return 0;
}
```

上述代码中，我们创建了一个`A`对象，并调用了它的成员方法`print()`。如果我们编译上述代码，并查看`a.out`的符号表，我们会发现`A`的虚函数`bark()`依然存在，其原因是`-dead_strip`貌似无法strip掉类的virtual table，进而strip不掉虚方法

```shell
> clang++ main.cpp a.cpp -Wl,-dead_strip -o a.out
> nm a.out | c++filt

...
0000000100000e40 unsigned short A::bark()
0000000100001b70 T A::print()
0000000100000e00 unsigned short A::A()
0000000100000e20 unsigned short A::A()
...
00000001000020e8 short vtable for A
               U vtable for __cxxabiv1::__class_type_info
...
```

### The C++ Registry Pattern

C++中有一种很常见的Registry Pattern，即通过定义一个无用的全局变量来执行一段初始化代码，我们还是来看一个具体的例子

```cpp
//main.cpp
class AA {
public:
    AA(){}
    void foo(){}
};
class BB {
public:
    void bar(){}
}
auto REG_a = AA();

int main(){
    BB b = BB();
    b.bar();
    return 0;
}
```
上述例子中我们的`main()`函数定义了`b`并调用了`bar()`，按照我们对`-dead_strip`的理解，`AA`应该会被strip掉，因为除了`auto REG_a = AA()`这句之外没有其它的Call Site，而`REG_a`也没有在`main`函数中出现，因此应该同样被strip掉，我们可以编译一下看看结果是否符合预期

```cpp
> clang++ main.cpp -Wl,-dead_strip
> nm a.out | c++filt | grep AA

0000000100000db0 unsigned short AA::AA()
0000000100000f10 unsigned short AA::AA()
```
我们发现`AA`和`REG_a`并没有被strip掉，也就是说对于这种情况，linker会保留`AA`的构造函数，以及构造函数的transitive closure。但是对于`AA::foo()`却不会保留。

## Resources

- [Linkers and Loaders](https://www.amazon.com/Linkers-Kaufmann-Software-Engineering-Programming/dp/1558604960)
- [Advanced C and C++ compiling](https://www.amazon.com/Advanced-C-Compiling-Milan-Stevanovic/dp/1430266678)