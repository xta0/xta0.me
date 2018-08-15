---
layout: post
list_title: iOS中的Method Overloading | Method Overloading in iOS
title: iOS中的Method Overloading
categories: [Objective-C,iOS]
---

Objective-C的`[...]`语法来源于smallTalk，其设计初衷是希望它是一种动态的，在运行时决定函数的入口地址。由于objective-C本质上也还是C语言，因此[...]语法在编译后实际上是一条标准C函数：

```c
id objc_msgSend(id self, SEl OP, ...)
```

而C语言是一种静态语言，也就是说，函数的入口在编译的时候就已经确定了(static binding)。例如这样一段代码：

```c
__attribute__((noinline))void searchProduct()
{
    printf("searching product");
}

__attribute__((noinline))void searchShop()
{
    printf("searching shop");
}

void search_static(int type)
{
    if (type == 1) {
        searchProduct();
    }
    else
        searchShop();
}
```

我们不让编译器去自动inline代码，searchProduct和searchShop在编译的时候地址已经确定了：

```
.globl _search_static
_search_static: ## @search_static
.cfi_startproc
Lfunc_begin2:
.loc 1 48 0
pushq %rbp

...................

Ltmp20:</span>
 ## BB#1:
.loc 1 50 0
popq %rbp
jmp _searchProduct ## TAILCALL
Ltmp21:</span>
LBB2_2:</span>
.loc 1 53 0
popq %rbp
jmp _searchShop ## TAILCALL
Ltmp22:</span>
Lfunc_end2:
.cfi_endproc
```

如果把上面代码改写成下面这样：

```c
__attribute__((noinline))void searchProduct()
{
    printf("searching product");
}

__attribute__((noinline))void searchShop()
{
    printf("searching shop");
}

void search_dynamic(int type)
{
    void(*func)();
    
    if (type == 1) {
        func = searchProduct;
    }
    else
        func = searchShop;
    
    func();
}
```

这样就变成了运行时决定函数地址了。这种方式好处是很灵活，你甚至可以在运行时修改程序执行的结果。但是坏处也很明显，就是需要计算，慢。回退到1986年，OC和C++都刚刚问世，同是面向对象，C++因为其在编译器做大量的事而速度更快，OC由于受当时硬件条件，这种动态绑定的做法效率往往很低。

基本上OC的函数执行都是遵循上面这种模式，判断执行那个函数的条件是objc_msgSend的SEL参数，它是一个opaque string:

```objc
SEL someSelector = NSSelectorFromString(@"dealloc")
```
而函数的入口地址保存在了：

```objc
struct objc_method_list **methodLists 
```
为了效率考虑，这些地址被调用过后会cache住

```objc
struct objc_cache *cache  
```
例如：

```shell
2   CoreFoundation                      0x01a0c903 -[NSObject(NSObject) doesNotRecognizeSelector:] + 275
3   CoreFoundation                      0x0195f90b ___forwarding___ + 1019
4   CoreFoundation                      0x0195f4ee _CF_forwarding_prep_0 + 14
5   libobjc.A.dylib                     0x014c6275 _class_initialize + 599
6   libobjc.A.dylib                     0x014cd0f1 lookUpImpOrForward + 158
7   libobjc.A.dylib                     0x014cd04e _class_lookupMethodAndLoadCache3 + 55
8   libobjc.A.dylib                     0x014d512f objc_msgSend + 139
9   UIKit                               0x0034b9d4 -[UIViewController loadViewIfReq
```

虽然有cache或有一些优化在，但效率毕竟还是低于static binding，但对于现在的硬件条件来讲，这个也不算是瓶颈。

### Overloading的问题

使用Java或者C++，overloading是种很常见的策略，或者很多人习惯通过overloading的参数来区别方法的意思。但是这种技术在OC中却行不通，道理也很简单，overloading也是一种static binding。

以Java代码为例：

```java
class testObj
{
	public void searchProduct(String name)
	{
		System.out.println(name);
	}
	
	 public void searchProduct(int type)
	 {
			System.out.format("%d\n",type);
	 }
}

public class main {

	public static void main(String[] args) {
		// TODO Auto-generated method stub
		testObj obj = new testObj();
		
		obj.searchProduct(0);
		obj.searchProduct("iphone");
	}
}
```

可以看到main.class中，方法的入口地址也确定了。

```shell
public static void main(java.lang.String[]);
  Code:
  Stack=2, Locals=2, Args_size=1
  0:	new	#15; //class testObj
  3:	dup
  4:	invokespecial	#17; //Method testObj."<init>":()V
  7:	astore_1
  8:	aload_1
  9:	iconst_0
  10:	invokevirtual	#18; //Method testObj.searchProduct:(I)V
  13:	aload_1
  14:	ldc	#22; //String iphone
  16:	invokevirtual	#24; //Method testObj.searchProduct:(Ljava/lang/String;)
```
为什么OC不行呢？

对同名的方法，参数类型不同，仅靠SEL是无法区别的：

```objc
@selector(searchProduct:)
```
如果有同名方法，谁知道该调用哪一个呢？






