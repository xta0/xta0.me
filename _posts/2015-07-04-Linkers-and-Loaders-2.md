---
layout: post
list_title: 链接与加载 | Linkers and Loaders | 静态库 | Static Library 
title: 静态库与动态库
categories: [C,C++]
---

## 静态库

无论是动态库还是静态库都是为了解决代码重用的问题。静态库可以理解为一系列目标文件的集合，link的时候静态库中的symbol会和目标文件中的symbol一起链接，如下图所示

<img src="{{site.baseurl}}/assets/images/2015/07/static-linking.png">

这种方式的好处是简单粗暴，静态库中的代码和目标工程中的代码一起编译连接，linker可以做全局的symbol级别的optimization，比如strip掉dead code等。使用静态库的劣势在于它会增加binary的大小，更重要的是如果静态库中的代码更新了，整个工程需要重新编译，不够灵活。

### 静态链接

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
ld -o a.out main.o a.o -lc++ -L/usr/local/lib -lSystem $path_to_libclang_rt.osx.a
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
ld -o a.out main.o a.o -lc++ -L/usr/local/lib -lSystem $path_to_libclang_rt.osx.a -dead_strip
```
> 也可以直接 `clang++ main.cpp a.cpp -Wl,-dead_strip`

此时我们再查看`a.out`的符号表则会发现`a_bar()`已经不在了。

### 静态库的优化




### Whole Archive的问题


## Resources

- [Linkers and Loaders](https://www.amazon.com/Linkers-Kaufmann-Software-Engineering-Programming/dp/1558604960)
- [Advanced C and C++ compiling](https://www.amazon.com/Advanced-C-Compiling-Milan-Stevanovic/dp/1430266678)