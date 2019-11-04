---
layout: post
list_title: 链接与加载 | Linkers and Loaders | 符号的可见性问题 | Introduction to Symbol visibility
title: 符号的可见性问题
categories: [C,C++]
---

我们继续讨论动态库的问题，接下来我们将把关注点放在符号的可见性问题上。符号的可见性对于动态库至关重要，默认情况下，动态库的符号都是全局可见的，那么如果一个进程加载了两份动态库，恰好有重名的符号，那么则会有符号覆盖的问题。此外，控制动态库中符号的数量也对动态库的加载速度有帮助

### 符号的可见性问题

我们还是以一个例子来看符号可见性的问题，假如我们有下面代码

<div class="highlight md-flex-h md-margin-bottom-24">
<div>
<pre class="highlight language-cpp md-no-padding-v md-height-full">
<code class="language-cpp">
//a.c
int myintvar = 5;
 
int func0 () {
  return ++myintvar;
}
 
int func1 (int i) {
  return func0() * i;
}

</code>
</pre>
</div>
<div class="md-margin-left-12">
<pre class="highlight language-cpp md-no-padding-v md-height-full">
<code class="language-cpp">
//main.cpp
extern int myintvar;
int main(){
    printf("%d",myintvar); 
}
</code>
</pre>
</div>
</div>

我们将`a.c`编译为动态库，并查看其中的符号

```shell
> clang -fPIC -shared a.c -o a.so
> nm a.so

0000000000000f70 T _func0
0000000000000f90 T _func1
0000000000001000 D _myintvar
                 U dyld_stub_binder
```

我们发现三个符号的类型均为大写字母T或D，说明他们是global的符号，全局可见。此时如果有另一个文件`b.c`中有这样一行代码

```c
int myintvar = 10;
```
那么我们将`b.c`,`a.so`一起和`main.c`进行编译，并观察输出结果

```shell
> clang a.so b.c main.c
> ./a.out //10
```
我们发现输出结果为10，也就是说动态库中的符号被覆盖掉了。接下来，假如我们的动态库只需要导出`func1`，而隐藏`func0`和`myintvar`，该怎么做呢？我们至少有三种方法，包括使用`static`关键字，定义符号的`GNU visibility`，以及使用exported symbol list。每种方式都有各自的优缺点，我们接下来一一讨论

### 使用static关键字

在C/C++中被`static`声明的变量符号类型会变成local，也就是说禁止该符号被外部链接。因此用`static`修饰的变量是永远不会被link的。编译器不会为该符号生成任何信息，因此这种方式是一种最简单的方式，我们修改`a.c`如下

```cpp
static int myintvar = 5;
 
static int func0 () {
  return ++myintvar;
}
 
int func1 (int i) {
  return func0() * i;
}
```
重新编译动态库，并查看符号表

```shell
> clang -fPIC -shared a.c -o a.so
> nm a.so

0000000000000fa0 t _func0
0000000000000f80 T _func1
0000000000001000 d _myintvar
                 U dyld_stub_binder

```

我们发现`_func0`和`_myintvar`的符号类型变成了小写的`t`和`d`，说明这两个符号变成了`local`的。

虽然`static`可以隐藏符号，但是它同样限制了符号的作用域，`func0`和`myintvar`只可以在`b.c`中使用，即被`static`修饰的符号，只可在定义它们的文件中使用。因此`main.c`中将无法看到`myintvar`，当编译`main.c`时会有link错误。

使用`static`这种方式更多的是用于控制文件内的符号可见性，而不用于控制低级别的符号可见性。实际上，大多数函数或者变量不会依赖于static来控制符号可见性。

### 使用`visibility`关键字

更常用的方法是使用GNU的visibility关键字，常用的有两个

- `default`，符号将被导出，默认可见
- `hidden`，符号不被导出，不能被其它对象使用

我们修改`a.c`的代码如下

```c
int myintvar __attribute__ ((visibility ("hidden")));
int __attribute__ ((visibility ("hidden"))) func0 () {
  return ++myintvar;
}
```