## 使用Objective-C++ 的 Pitfalls

### C++对象在Block中的Copy


我们有一个C++的类：`FluxAction`，然后我们在stack上创建一个`FluxAction`对象和一个`block`，并在`block`中使用这个`FluxAction`对象，代码如下所示：


```cpp

FluxAction action = {
   view_action,100,@"abc",nil
};
    
void(^block)(void) = ^{

   NSLog(@"%@",action.payload);
   
};
block();

```

代码执行期间会对`FluxAction`对象进行3次构造，3次析构，分别是:

- `action`对象创建调用初始化构造函数
- `action`对象被block copy，调用了两次`action`的拷贝构造函数
- `block()`执行完释放3个`action`对象，调用3次析构函数

这里有个问题，为什么`block`会对`action`进行两次copy，这点我们在后面讨论。

如果将`action`用`__block`声明:

```cpp

__block FluxAction action = {
	view_action, 100 , @"abc", nil;
};

block();

```