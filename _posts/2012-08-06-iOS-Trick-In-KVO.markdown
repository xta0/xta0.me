---
list_title: 理解Objective-C中的KVO
layout: post
tag: Objective-C
categories: 随笔

---

<em>所有文章均为作者原创，转载请注明出处</em>

如果一个对象被key-value-observed ：

```objc

@interface KVOTestClass : NSObject

@property(nonatomic,strong) NSString* observedObj;
@property(nonatomic,strong) NSString* unobservedObj;

@end

@implementation KVOTestClass

@end

int main(int argc, char * argv[])
{
    @autoreleasepool {
    
   //normal class
    KVOTestClass* normalClass = [KVOTestClass new];

    //KVOed class
    KVOTestClass* kvoClass = [KVOTestClass new];

    [kvoClass addObserver:normalClass forKeyPath:@"observerObj" options:NSKeyValueObservingOptionNew   context:NULL];

    const char* metaClassName = object_getClassName(kvoClass);
    NSString* className = NSStringFromClass(kvoClass.class);

    NSLog(@"\n kvo object's metaclass : %s \n kvo object's class : %@",metaClassName,className);
        
        }
}
```

输出为：

<span style="color: #ff6600;">kvo object's metaclass : NSKVONotifying_KVOTestClass</span>

<span style="color: #ff6600;"> kvo object's class : KVOTestClass</span>


有两个问题：

（1）为什么这两个api结果不一样？

（2）NSKVONotifying_KVOTestClass怎么出来的？

第一个问题不属于kvo的讨论范畴，第二个问题是apple的一个trick：
>
“So how does that work, not needing any code in the observed object? Well it all happens through the power of the Objective-C runtime. When you observe an object of a particular class for the first time, the KVO infrastructure creates a brand new class at runtime that subclasses your class. In that new class, it overrides the set methods for any observed keys. It then switches out the isa pointer of your object (the pointer that tells the Objective-C runtime what kind of object a particular blob of memory actually is) so that your object magically becomes an instance of this new class.”

当kvoClass被某个观察者(normalClass)观察时，runtime会为kvoClass动态生成一个subclass（NSKVONotifying_KVOTestClass）来代替kvoClass，方法是改变kvoClass的isa。并且重载了被观察property（observedObj）的setter方法，那么当observedObj的值发生变化时，就可以通知观察者。

但为什么NSStringFromClass或class_getName返回的是KVOTestClass呢？

That's another trick：

>
Apple really doesn't want this machinery to be exposed. In addition to setters, the dynamic subclass also overrides the -class method to lie to you and return the original class! If you don't look too closely, the KVO-mutated objects look just like their non-observed counterparts.

原因是这个动态生成的subClass（NSKVONotifying_KVOTestClass）同样override了class方法，来返回它原来类型。

因此使用者根本感觉不出kvoClass这个instance的类型已经发生变化了。

下面这段代码可以看的更清楚：

 
```objc
static NSArray *ClassMethodNames(Class c)
{
    NSMutableArray *array = [NSMutableArray array];
    
    unsigned int methodCount = 0;
    Method *methodList = class_copyMethodList(c, &amp;methodCount);
    unsigned int i;
    for(i = 0; i &lt; methodCount; i++)
        [array addObject: NSStringFromSelector(method_getName(methodList[i]))];
    free(methodList);
    
    return array;
}
``` 

然后修改下main函数：

```objc
NSArray* normalClassMethods = ClassMethodNames(object_getClass(normalClass));
NSArray* kvoClassMethods    = ClassMethodNames(object_getClass(kvoClass));
        
NSLog(@"\n normal class : \n %@ kvo class: %@ \n",normalClassMethods,kvoClassMethods);
```

输出为：

```objc
 normal class: 
 (
    observedObj,
    "setObservedObj:",
    unobservedObj,
    "setUnobservedObj:",
    ".cxx_destruct"
) 
 kvo class: 
 (
    "setObservedObj:",
    class,
    dealloc,
    "_isKVOA"
)
```

这就一目了然了，normalClass只有getter，setter方法

kvoClass是其子类，由于observedObj被观察，因此override了observedObj的setter方法，同样，class方法也被override用来返回“KVOTestClass”，dealloc方法被override原因是为了释放观察者，而_isKVOA，从命名上看，属于apple的似有api，大概是用来标识这个对象是动态生成的kvo对象。

That's all

