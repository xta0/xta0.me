---
layout: post
list_title: 理解iOS中的"isa"指针 | "isa" pointer in iOS
title: 理解iOS中的"isa"指针
categories: [Objective-C,iOS]
---

这是个很奇怪的问题，这个问题困扰我的时间也最长，要先从id说起：

```objc
typedef struct objc_object *id;
struct objc_object {
    Class isa;
};
typedef struct objc_class *Class;
```

OC中的对象都可以用`id`来表示，`id`是一个`objc_object`结构体指针，这个结构体中只有一个成员是`isa`。`isa`代表了这个对象的类型：
这句代码：

```objc
Class clz = [obj Class];
```

等价于：

```objc
objc_class* clz = obj->isa
```

`objc_class`这个结构体的定义为：

```c
struct objc_class {
    Class isa  
    Class super_class                                       
    const char *name                                        
    long version                                           
    long info                                               
    long instance_size                                    
    struct objc_ivar_list *ivars                         
    struct objc_method_list **methodLists                  
    struct objc_cache *cache                             
    struct objc_protocol_list *protocols                   
} 
```

那么就有:

```objc
objc_class* clz_Meta = obj->isa->isa。
```

Apple对`clz_Meta`的描述是MetaClass，也就是`Class`的class，这也不难理解，因为`clz`同样可以接受message：`[clz new]`,也有自己的类方法。

上面描述的结构如下图所示：

![](/assets/images/2012/02/class_hierarchy.png)

然后问题是：`obj->isa->isa->isa`指向哪里？

想不明白就来动手试试：
 
```objc
@interface SomeClass : NSObject
@property(nonatomic,assign) int a;
@end

@implementation SomeClass
@end

int main(int argc, char *argv[])
{
    SomeClass* obj = [SomeClass new];    
    [obj release];
    obj = nil;
    return 0;
}
```

然后我们用GDB调试：

```shell
(gdb) p (Class)[obj class]
$3 = (Class) 0x3668
(gdb) p obj->isa
$2 = (Class) 0x3668
```

这说明`[obj Class]`等价于`obj->isa`，复合我们的预期，然后我们看看`objc_class`这个结构体中的状态：

```shell
(gdb) p *obj->isa
$7 = {
  isa = 0x3654, 
  super_class = 0x11d4bc0, 
  name = 0x718e150 "\003", 
  version = 18697404, 
  info = 119070944, 
  instance_size = 13968, 
  ivars = 0x7fa62c, 
  methodLists = 0x11d9e40, 
  cache = 0x11d4cbc, 
  protocols = 0x3558
}
```

我们看到了MetaClass也就是`obj->isa->isa`指向`了0x3654`，我们用GDB调试下：

```shell
(gdb) p obj->isa->isa
$8 = (Class) 0x3654
```

接着，我么再看看MetaClass的objc_class结构体：

```shell
(gdb) p *obj->isa->isa
$6 = {
  isa = 0x11d4bd4, 
  super_class = 0x11d4bd4, 
  name = 0x718e130 "\003", 
  version = 18697404, 
  info = 119070912, 
  instance_size = 13908, 
  ivars = 0x11d4bc0, 
  methodLists = 0x718e150, 
  cache = 0x11d4cbc, 
  protocols = 0x718e0e0
  }
```

看到MetaClass对象的`isa`指向了`0x11d4bd4`，即`obj->isa->isa = 0x11d4bd4`。这个时候我们发现super_class也是`0x11d4bd4`，也就是说MetaClass的`isa`指向了其父类，我们看看它父类是什么？

```shell
(gdb) po obj->isa->isa->isa
NSObject
```
MetaClass对象的`isa`指向了super_class，那`super_class`的父类，也就是`NSObject`的`MetaClass`对象的`super_class`指向哪里？

```shell
(gdb) po obj->isa->isa->isa->super_class->super_class
Can't print the description of a NIL object.
```

`Nil`是我们期待的结果，到这里，我们应该把上面的图再改一改：

![](/assets/images/2012/02/class_hierarchy-2.png)

{% include _partials/post-footer-1.html %}



