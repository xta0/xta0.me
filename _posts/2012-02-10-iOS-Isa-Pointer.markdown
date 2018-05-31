---
layout: post
list_title: 理解isa
tag: Objective-C
categories: 随笔

---

<em>所有文章均为作者原创，转载请注明出处</em>

这是个很奇怪的问题，这个问题困扰我的时间也最长，要先从id说起：

```objc
typedef struct objc_object *id;
struct objc_object {
    Class isa;
};
typedef struct objc_class *Class;
```

OC中的对象都是id，id是一个objc_object结构体指针，这个结构体中只有一个成员是isa。isa代表了这个对象的类型：
这句代码：

```
Class clz = [obj Class];
```

等价于：

```
objc_class* clz = obj->isa

```

objc_class这个结构体的定义为：

```objc

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

```
objc_class* clz_meta = obj->isa->isa。
```

Apple对clz_meta的描述是metaClass，也就是`Class`的class，这也不难理解，因为`clz`同样可以接受message：[clz new],也有自己的类方法。

上面描述的结构如下图所示：

<a href="/assets/images/2012/02/class_hierarchy.png"><img src="/assets/images/2012/02/class_hierarchy.png" alt="class_hierarchy"width="511" height="181"/></a>

然后问题是：

```
obj->isa->isa->isa
```

指向哪里？

想不明白就来动手试试：
 
<pre class="theme:tomorrow-night lang:objc decode:true " >@interface SomeClass : NSObject

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
}</pre> 

然后我们用GDB调试：

```
(gdb) p (Class)[obj class]
$3 = (Class) 0x3668
(gdb) p obj->isa
$2 = (Class) 0x3668

```

这说明[obj Class]等价于obj->isa

然后我们看看objc_class这个结构体中的状态：

```
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

我们看到了metaClass也就是obj->isa->isa指向了0x3654，我们用GDB调试下：

```
(gdb) p obj->isa->isa
$8 = (Class) 0x3654
```

接着，我么再看看metaClass的objc_class结构体：

```
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

看到metaClass对象的isa指向了0x11d4bd4，即obj->isa->isa = 0x11d4bd4。这个时候我们发现super_class也是0x11d4bd4，也就是说metaClass的isa指向了其父类，我们看看它父类是什么？

```
(gdb) po obj->isa->isa->isa
NSObject
```

That's it !! That solved mystery!!

But wait ! MetaClass对象的isa指向了super_class，那super_class的父类，也就是NSObject的MetaClass对象的super_class指向哪里？

```
(gdb) po obj->isa->isa->isa->super_class->super_class
Can't print the description of a NIL object.

```

Nil是我们期待的结果

到这里，我们应该把上面的图再改一改：

<a href="/assets/images/2012/02/class_hierarchy-2.png"><img src="/assets/images/2012/02/class_hierarchy-2.png" alt="class_hierarchy-2" width="525" height="169"/></a>

That's all for today!

