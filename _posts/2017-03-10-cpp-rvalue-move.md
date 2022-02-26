---
layout: post
list_title: C++中的右值引用与std::move
title: C++中的右值引用与std::move
categories: [C++]
---

在C++11以前，只有左值和右值，左值很好理解，右值可以被绑定到一个常量的左值引用，即`const T&`，当不能绑定到非常量引用`T&`的左值。在C++11之后，出现右值引用的类型`&&`和`std::move`，此时一个右值可以被绑定到一个具有`T&&`的类型的左值，此时这个左值在术语叫做xvalue。它具有像左值一样可以进行取地址的操作，同时也具有右值的特性，即可以被移动。根据这份[Value Category](https://en.cppreference.com/w/cpp/language/value_category)的描述，更准确的分类应该是如下图所示

```shell
              expression
              /       \
        glvalue       rvalue
       /      \      /      \
lvalue         xvalue        prvalue
```

其中lvalue和prvalue比较好理解，lvalue是纯左值，它在等号左边，有标识符，能取地址，不能被移动。而纯右值也很好理解，它在等号的右边，没有标识符，不能取地址，但是可以被移动。xvalue比较特殊，它有标识符，可以出现在等号左边，可以被取地址，同时还可以被移动。上面三种类型常见的用法可参考上面的[Value Category](https://en.cppreference.com/w/cpp/language/value_category)
、 。    
### 右值引用

一个右值引用既可能是左值也可能是右值，区分标准在于如果他有标识符，那他就是左值，如果没有，则是右值。可见右值引用是一种xvalue。假设我们有一个Dummy类如下，它重载了拷贝构造函数和移动构造函数

```cpp
class Dummy {
public:
    string x = "100";
    Dummy(string i):x(i){}
    Dummy(const Dummy& d):x(d.x){}
    Dummy(Dummy&& d):x(std::move(d.x)){}
    ~Dummy(){
        std::cout<<__func__<<std::endl;
    }
};
```
我们先看右值引用充当左值的情况

```cpp
void foo(Dummy&& dm){
 Dummy d = dm; // calls the copy constructor
}
```
此时虽然`dm`有标识符，是一个充当右值引用的左值，`d`会通过`Dummy`的拷贝构造函数创建。原因是`dm`在`foo`中为左值，生命周期和`foo`函数一致，这意味着在后面的代码中可能会被访问到，因此它不可能将自己的内存交给`d`，否则如果被后面代码修改，则将会造成错误的结果。

接下来我们看右值引用充当右值的情况

```cpp
Dummy&& dummy(){
  return std::move(Dummy());
}
void bar(){
 Dummy dm = dummy(); //calls the move constructor
}
```
此时在`bar()`中，`dm`会通过移动构造函数创建。因为`dummy()`返回的是一个右值prvalue，通过`std::move(Dummy())`将其变成了一个没有标识符的右值引用。

这里需要注意，在`dummy()`stack中创建的`Dummy()`对象是一个prvalue，它在`dummy()`函数执行完成后就被释放了。因此在`bar()`中的`dm`虽然调用了移动构造，但是由于之前的对象已经释放，这里`dm`中的`x`将指向一个无效的内存地址。

> 通常情况下，不建议将右值引用作为返回值

我们将上面的例子稍作修改

```cpp
Dummy&& dummy(const Dummy& dm){
  return std::move(dm);
}
```

此时`std::move(dm)`接受一个左值`dm`，将它转化成了一个右值引用，这个过程同样没有拷贝，此时返回的右值引用指向的仍是左值`dm`的地址。

另一个用比较常见的用法是子类继承父类的拷贝构造函数

```cpp
Base(Base const& rhs); //copy constructor
Base(Base&& rhs); //move constructor

Derived(Derived const& rhs): Base(rhs){

}
Derived(Derived&& rhs) 
  : Base(rhs) // wrong: rhs is an lvalue
{
  // Derived-specific stuff
}
Derived(Derived&& rhs) 
  : Base(std::move(rhs)) // good, calls Base(Base&& rhs)
{
  // Derived-specific stuff
}
```
上面`Derived`的移动构造如果不使用`std::move()`，则会触发基类的拷贝构造而非移动构造，因此此时`rhs`是左值。

### prvalue的生命周期

在生命周期方面，prvalue对象在表达式执行完成后立即释放，xvalue则在作用域结束后释放。如下面代码所示

```cpp
Dummy foo(){
  return Dummy();
}

int main(){
      foo(); //prvalue
      std::cout<<__func__<<std::endl;
}
```
上面`foo()`返回了一个prvalue，当`foo()`执行完成后，prvalue会立即释放，则我们看到的log顺序为

```
~Dummy
main
```
但是如果我们将一个prvalue绑定到一个上面代码修改为

```cpp
Dummy&& dm = foo();
```
则我们发现`dm`在`main`执行完后才被析构。这是因为我们将一个prvalue绑定到了一个右值引用上面，该引用值的生命周期将持续到作用域结束。

注意，这里有一个坑，即一个xvalue是无法被右值引用的

```cpp
Dummy&& dm = std::move(foo());
```
此时在`foo()`执行完成后，临时对象`prvalue`便会释放，因此`dm`将绑定到一个不可用的内存地址，此时`dm`的行为将是undefined behavior


### std::move解决什么问题

简单的说move解决大对象的拷贝问题，大对象包括容器和一些占内存较大的类对象。我们假设上面的`Dummy`类hold一个指向一块较大内存对象的指针，`m_pResource`，则下面代码将触发该对象的拷贝

```cpp
Dummy dm("dm");
dm = Dummy("dm2");
```
考虑上面最后一行代码，它做了三件事

- 拷贝临时对象持有的`m_pResource`
- 释放`dm`原来持有的`m_pResource`
- 释放临时对象持有的`m_pResource`

如果`m_pResource`指向的是一个很大的对象，上述行为这显然效率不高，如果我们能将历史对象的`m_pResource`直接transfer给`dm`，那么性能将会有极大的提升，这也是`std::move`的基本实现原理。因此对于`Dummy`的移动构造和移动复制函数，我们需要做的是实现资源的交换

```cpp
Dummy& Dummy::operator=(Dummy&& rhs)
{
  // [...]
  // swap this->m_pResource and rhs.m_pResource
  // [...]  
}
```
可见我们需要用`swap`来交换两个对象的resource，这意味着`rhs`将拥有`dm`的resource。


### 支持move semantics

要让某个对象支持`std::move`需要做下面几件事情

- 支持拷贝构造和移动构造函数
- 实现`swap`成员函数，支持和另外一个对象快速交换成员
- 实现一个全局的 `swap` 函数，调用成员函数 `swap` 来实现交换。
- 实现移动赋值 `operator=`
- 上面各个函数如果不抛异常的话，应当标为`noexcept`

###  编译器对函数返回值的优化

如果一个函数返回一个对象，编译器可直接将其在调用栈上创建，因此并不会多调用一次拷贝构造，如果强行用`std::move`还会破坏这个编译器优化，比如下面代码

```cpp
Dummy dummy(){
  Dummy dm;
  // do something to dm
  return dm; //won't call the copy constructor
  // return std::move(dm); // making it worse!
}
void main(){
  Dummy dm = dummy();
}
```

## Resources

- [C++ Primer]()
- [Effective Modern C++]()
- [Rvalue References Explained, by Thomas Becker](http://thbecker.net/articles/rvalue_references)