---
layout: post
list_title: C++ Essentials | RValue and Move Semantics
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
    Dummy(Dummy&& d):x(std::move(d.x)) noexcept{}
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

### `&&`修饰成员函数

我们可以将某个成员函数标记为`&&`，则表示该成员函数只能被一个右值对象调用

```cpp
struct Foo {
  auto func() && {}
};

auto a = Foo{};
a.func(); //error, a is lvalue
std::move(a).func(); // compiles
Foo{}.func(); // compile
```

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

## `std::move`解决什么问题

简单的说move解决内存拷贝问题，通常情况下如果一个对象hold了一块比较大的，并且是heap-allocated的资源，使用`std::move`比copy效率更高。

```cpp
Dummy dm(a_big_vector);
Dummy dm2(std::move(dm))
```
上面代码中如果将`dm`直接copy给`dm2`则cost会很高，如果用`std::move`则将`dm2`中vetor的指针直接指向`dm`中的vector，效率很高

### Rule of Zero/Five

如果要完整的支持move，一个类需要实现下面五个函数

- copy constructor
- copy assignment operator
- move constructor
- move assignment operator
- destructor

默认情况下，编译器会为每个类自动生成上面五个函数，但是我们如果override了其中一个，则compiler将不会自动生成其它的四个。此时，由于move构造没有被生成，所有该类的对象将支持copy。

### A common pitfall - moving non-resources

使用系统默认的构造函数，如果类中有foundamental type的成员(`int`, `float`, `bool`)，则这些成员不会被move，而是被copy。这回带来一些异常的情况

```cpp
struct S4 {
    std::vector<float> _data;
    int _index{-1};
    S4 (const std::vector<float>& data):_data(data){}
    void set_index(int index) {
      _index = index;
    }
    float select_item() const {
      return _data[_index];
    }
};

int main() {
  std::vector<float> data{1.0, 1.1, 1.2};
  S4 s4(data);
  s4.set_index(2);
  S4 s44(std::move(s4));
  auto x1 = s44.select_item(); //OK, 1.2
  auto x2 = s4.select_item(); //UB: undefined behavior. The underlying data has gone
}
```
上面代码中，对于`s44`来说，`_index`被copy到自己的`_index`中，`_data`被move到自己的`_data`中，因此`s44`没有任何问题。但是对于`s4`来说，move后`_data`已经不存在了，但是`_index`由于是copy，因此还保存着原来的值`2`，此时`_data[_index]`将会crash。解决办法是重载move constructor和move operator

```cpp
S4(const S4&& other) noexcept{
  std::swap(_data, other.data);
  std::swap(_index, other._index);
}

auto& oeprator=(S4&& other) noexcept{
  std::swap(_data, other.data);
  std::swap(_index, other._index);
}
```

需要注意的是我们需要将move constructor和move assignment operator标记成`noexcept`。如果不加这个mark，一些静态库仍会使用copy构造

### move构造函数的参数

对于移动构造函数的传参，我们可以使用**pass-by-value-then-move**的pattern。这会减少一次对参数的copy

```cpp
class Widget {
  std::vector<int> data_;
public:
  Widget(std::vector<int> x) : data_{std::move(x)}{}
  // ...
}
```

## Resources

- [C++ Primer]()
- [Effective Modern C++]()
- [Rvalue References Explained, by Thomas Becker](http://thbecker.net/articles/rvalue_references)