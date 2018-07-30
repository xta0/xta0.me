---
list_title: 理解Apple的LLVM(1)
layout: post
tag: iOS
categories: 随笔

---

<em></em>

最近在用LLDB，感到很乏力，和GDB的命令差很多，因此想好好研究下iOS代码的编译环境——<a href="llvm.org">LLVM</a>（low level virtual machine）。

其实LLVM跟virtual machine没有必然的联系，也不能直接说它是编译器，它自己给自己的定义是一个infrastructure，也就是编译器体系结构，里面包含了许多子模块，比如前端语法编译器（front-end）<a href="http://clang.llvm.org/">Clang</a>，后端机器码生成工具（back-end）<a href="http://dragonegg.llvm.org/">dragonegg</a>,调试器<a href="http://lldb.llvm.org/">LLDB</a>等。

WWDC2011:session307介绍了LLVM的历史，在LLVM问世之前，apple是一直用GCC的，但由GCC对很小众的Objective-C支持很慢，直白的说就是你给人家提需求，人家不改，导致Objective-C的一些新特性无法应用（我试着在windows下用GCC编译Objective-C代码，还必须要@synthesize property）。后来apple将GCC的前端编辑器单独抽出来，架在LLVM上，形成了GCC-LLVM的编译体系。但是由于GCC是一个从源代码编译到生成机器码（machine code）的完整过程，耦合性很高，想单独优化某一块很困难，因此这条路也没走通。后来苹果痛下决心，决定自己单独搞一套，找来了当时还在读本科的<a href="http://www.programmer.com.cn/9436/">Chris Lattner</a>,这哥们几乎以一己之力完成了LLVM的雏形，奠定了现在<a href="http://www.aosabook.org/en/llvm.html">LLVM现在的架构</a>。

LLVM的体系结构如下：

<a href="/assets/images/2013/11/llvm.png"><img src="/assets/images/2013/11/llvm.png" alt="llvm" width="491" height="67"/></a>

这个结构和GCC相比最大的优点是模块化，这意味着，每一块都可以做单独的优化和扩展，比如你想增加一种language（e.g.C,C++,Objective-C等），只需要对frontEnd进行扩展，如果你想增加一种新的平台（e.g. Intel,ARM,PowerPC等），只需要对backEnd进行扩展。

由于从源码到机器码这个编译过程是一个很复杂的过程，我们先从Clang开始。

Clang顾名思义,是C-language的统称，Apple对它的解释是，命令行上和使用gcc一样，但比gcc智能的多，尤其在错误的提示上和解决上。
我们可以先看看编译一个.m文件需要哪几步：

假设我们有一个hello.m的文件：

```c
#import "Foundation/Foundation.h";

int main(int argc, char *argv[]) {
	printf("hello\n");
	return 0;
}

```
```
clang -ccc-print-phases hello.m 
0: input, "hello.m", objective-c
1: preprocessor, {0}, objective-c-cpp-output
2: compiler, {1}, assembler
3: assembler, {2}, object
4: linker, {3}, image
5: bind-arch, "x86_64", {4}, image

```

###预处理

编译源码的第一步就是预处理，简单的说就是将宏展开，比如我们
`#import "Foundation/Foundation.h"`
编译器会把Foundation.h中的所有.h都展开：

```
clang -E hello.m | less
# 1 "/usr/include/sys/cdefs.h" 1 3 4
# 406 "/usr/include/sys/cdefs.h" 3 4
# 1 "/usr/include/sys/_symbol_aliasing.h" 1 3 4
# 407 "/usr/include/sys/cdefs.h" 2 3 4
# 472 "/usr/include/sys/cdefs.h" 3 4
# 1 "/usr/include/sys/_posix_availability.h" 1 3 4
# 473 "/usr/include/sys/cdefs.h" 2 3 4
# 76 "/usr/include/sys/types.h" 2 3 4
# 1 "/usr/include/machine/types.h" 1 3 4
# 35 "/usr/include/machine/types.h" 3 4
# 1 "/usr/include/i386/types.h" 1 3 4
# 70 "/usr/include/i386/types.h" 3 4
# 1 "/usr/include/i386/_types.h" 1 3 4
# 37 "/usr/include/i386/_types.h" 3 4
typedef signed char __int8_t;
.....

```

<h3>符号化</h3>

预处理完了之后，需要将这些string符号化:

我们先把<code>#import "Foundation/Foundation.h"</code>注释掉，然后：

```
clang -Xclang -dump-tokens hello.m

int 'int'	         [StartOfLine]	Loc= hello.m:3:1 
identifier 'main'	 [LeadingSpace]	Loc= hello.m:3:5
l_paren '('	                	Loc= hello.m:3:9
int 'int'		                Loc= hello.m:3:10
identifier 'argc'	 [LeadingSpace]	Loc= hello.m:3:14
comma ','		                Loc= hello.m:3:18 
char 'char'	  [LeadingSpace]	Loc= hello.m:3:20
star '*'	  [LeadingSpace]	Loc= hello.m:3:25
identifier 'argv'	        	Loc= hello.m:3:26
l_square '['		                Loc= hello.m:3:30
r_square ']'		                Loc= hello.m:3:31 
r_paren ')'		                Loc= hello.m:3:32 
l_brace '{'	 [LeadingSpace]	        Loc= hello.m:3:34 
identifier 'printf'	 [StartOfLine] [LeadingSpace]
Loc= hello.m:4:2 

```

我们可以看到，clang列出了符号所在的位置，当出错的时候，clang会指出具体的位置。

<h3>生成语法树</h3>

有了这些token后，clan会将这些token解析成语法树，假设我们把代码修改一下： 

```c
#import &lt;Foundation/Foundation.h&gt;

@interface Objayc
- (void)hello;
@end

@implementation Objayc
- (void)hello
{
	NSLog(@"hello");
}

@end

int main(int argc, char *argv[]) {
	
	Objayc* obj = [Objayc new];
	[obj hello];
	return 0;
}

``` 

然后使用命令：

```
clang -Xclang -ast-dump -fsyntax-only hello.m

```
忽略掉头文件：

```
-ObjCInterfaceDecl 0x1041eb290 <hello.m:3:1, line:5:2> Objayc
| |-super ObjCInterface 0x10236e430 'NSObject'
| |-ObjCImplementation 0x1041eb430 'Objayc'
| `-ObjCMethodDecl 0x1041eb3a0 <line:4:1, col:14> - hello 'void'
|-ObjCImplementationDecl 0x1041eb430 <line:7:1, line:13:1> Objayc
| |-ObjCInterface 0x1041eb290 'Objayc'
| `-ObjCMethodDecl 0x1041eb4d0 <line:8:1, line:11:1> - hello 'void'
|   |-ImplicitParamDecl 0x1041eb590 <<invalid sloc>> self 'Objayc *'
|   |-ImplicitParamDecl 0x1041eb5f0 <<invalid sloc>> _cmd 'SEL':'SEL *'
|   `-CompoundStmt 0x1041eb778 <line:9:1, line:11:1>
|     `-CallExpr 0x1041eb730 <line:10:2, col:16> 'void'
|       |-ImplicitCastExpr 0x1041eb718 <col:2> 'void (*)(id, ...)' <FunctionToPointerDecay>
|       | `-DeclRefExpr 0x1041eb648 <col:2> 'void (id, ...)' Function 0x10234cd10 'NSLog' 'void (id, ...)'
|       `-ImplicitCastExpr 0x1041eb760 <col:8, col:9> 'id':'id' <BitCast>
|         `-ObjCStringLiteral 0x1041eb6a0 <col:8, col:9> 'NSString *'
|           `-StringLiteral 0x1041eb670 <col:9> 'char [6]' lvalue "hello"
`-FunctionDecl 0x1041eb910 <line:16:1, line:21:1> main 'int (int, char **)'
  |-ParmVarDecl 0x1041eb7c0 <line:16:10, col:14> argc 'int'
  |-ParmVarDecl 0x1041eb840 <col:20, col:31> argv 'char **'
  `-CompoundStmt 0x1041ebb40 <col:34, line:21:1>
    |-DeclStmt 0x1041eba78 <line:18:2, col:28>
    | `-VarDecl 0x1041eb9e0 <col:2, col:27> obj 'Objayc *'
    |   `-ObjCMessageExpr 0x1041eba48 <col:16, col:27> 'Objayc *' selector=new class='Objayc'
    |-ObjCMessageExpr 0x1041ebad0 <line:19:2, col:12> 'void' selector=hello
    | `-ImplicitCastExpr 0x1041ebab8 <col:3> 'Objayc *' <LValueToRValue>
    |   `-DeclRefExpr 0x1041eba90 <col:3> 'Objayc *' lvalue Var 0x1041eb9e0 'obj' 'Objayc *'
    `-ReturnStmt 0x1041ebb20 <line:20:2, col:9>
      `-IntegerLiteral 0x1041ebb00 <col:9> 'int' 0

```

@interface开始，每个节点依次展开，生成了语法树。

语法树规则可以参考<a href="http://clang.llvm.org/docs/IntroductionToTheClangAST.html">官方指南</a>

<h3>静态语法检查</h3>

语法树生成后，clang会做静态的语法检测，包括数据类型，方法调用等一些编译器就能确定的事情。
然而一些runtime的事情在这时候是无法检查的。
一些高级的检测，比如某个局部变量创建了但没有使用或类似这种warn_arc_perform_selector_leaks的warning等等。

<h3>汇编代码</h3>

我们再一次简化代码： 

```c
#include "stdio.h";

int main(int argc, char *argv[]) {

	printf("hello!");	
	return 0;
}

```

<code>xcrun clang -S -o - hello.m | open -f</code>

<h3>Mach-O 文件</h3>

有了汇编代码就离机器码不远了，然后我们

<code>xcrun clang hello.m</code>

直接生成a.out的可执行文件<a href="https://developer.apple.com/library/mac/documentation/DeveloperTools/Conceptual/MachORuntime/Reference/reference.html ">文件格式</a>。

mach-o文件由Header和Commands和Data三个部分组成：

1，header ： 标识它是mach-o 文件
2，Load commands ： 标识data的存放规则
3，Data : 数据存放

Data部分又分为两部分：

1，Text Section ： 存放代码段，常量段等read-only数据
2，Data Section :     数据段，如静态变量，全局变量等，是writable的

分析mach-O文件，通常用otool。

<code>xcrun size -x -l -m a.out</code>

得到的commands如下：

```
Segment __PAGEZERO: 0x100000000 (vmaddr 0x0 fileoff 0)
Segment __TEXT: 0x1000 (vmaddr 0x100000000 fileoff 0)
	Section __text: 0x37 (addr 0x100000f10 offset 3856)
	Section __stubs: 0x6 (addr 0x100000f48 offset 3912)
	Section __stub_helper: 0x1a (addr 0x100000f50 offset 3920)
	Section __cstring: 0x7 (addr 0x100000f6a offset 3946)
	Section __unwind_info: 0x48 (addr 0x100000f71 offset 3953)
	Section __eh_frame: 0x40 (addr 0x100000fc0 offset 4032)
	total 0xe6
Segment __DATA: 0x1000 (vmaddr 0x100001000 fileoff 4096)
	Section __nl_symbol_ptr: 0x10 (addr 0x100001000 offset 4096)
	Section __la_symbol_ptr: 0x8 (addr 0x100001010 offset 4112)
	Section __objc_imageinfo: 0x8 (addr 0x100001018 offset 4120)
	total 0x20
Segment __LINKEDIT: 0x1000 (vmaddr 0x100002000 fileoff 8192)
total 0x100003000
```

运行时vm将segment映射到物理地址空间中，最后是mach-o文件的二进制机器码（opcode）：
例如：__TEXT下的__text：

<code>xcrun otool -s __TEXT __text a.out </code>

```
(__TEXT,__text) section
0000000100000f10 55 48 89 e5 48 83 ec 20 48 8d 05 50 00 00 00 c7 
0000000100000f20 45 fc 00 00 00 00 89 7d f8 48 89 75 f0 48 89 c7 
0000000100000f30 b0 00 e8 11 00 00 00 b9 00 00 00 00 89 45 ec 89 
0000000100000f40 c8 48 83 c4 20 5d c3 
```

汇编代码到机器码的对应关系可以在<a href="http://download.intel.com/products/processor/manual/325462.pdf">这里查到</a>

Further Reading：
<a href="http://llvm.org/docs/">llvm docset</a>
<a href="http://llvm.org/docs/GettingStarted.html#example-with-clang">clang example</a>\