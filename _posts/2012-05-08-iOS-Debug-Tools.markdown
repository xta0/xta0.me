---
update: "2016-07-30"
layout: post
title: XCode的调试工具
list_title: XCode 调试 | XCode Debug Tools
categories: [XCode,iOS]
---

## LLDB

### 常用命令

- 列出和某个命令相关的命令：`apropos thread`
- 列出所有break points: `(lldb) br l`
- 删掉break: `br delete 1`
- 开关break: `br e 1`,`br di 1`
- 设置break point: `b MyViewController.m :30`
- 继续:`c`
- 断点后但不执行`n`, 
- 进入某个函数:`s`
- 结束:`finish`
- 执行表达式`expr`

```shell
#隐藏view
expr self.view.hidden = yes
#variable:
expr int $a = 100 定义一个a变量值为100
```

### 增加符号断点

- 给所有viewdidload打断点:`br set -n viewDidLoad`
- 根据condition打断点：`br mod -c "totalValue > 1000" 3`
- 进入命令行模式：

```shell
(lldb) br com add 2
Enter your debugger command(s).  Type 'DONE' to end.
> bt
> continue
> DONE
```

### 函数堆栈,线程相关

- `bt` = `thread backtrace`
- `br all`
- `thread list`
- `frame info`
- `frame select <num>`
- `frame variable`

```shell
(lldb) frame variable
(TBCityStoreMenuViewController *const) self = 0x09646820
(SEL) _cmd = "onCoverFlowLayoutClicked:"
(UIButton *) btn = 0x09665d40
(lldb) frame variable self
(TBCityStoreMenuViewController *const) self = 0x09646820
```
- `frame variable -F self`
- `watchpoint list`
- `watchpoint delete 1`

```shell
#监控变量_x是否发生变化
watchpoint set variable _x

(lldb) watchpoint set variable _type
Watchpoint created: Watchpoint 1: addr = 0x0955c8d0 size = 4 state = enabled type = w
    watchpoint spec = '_type'
    new value: kGrid
```
如果x发生变化：

```shell
Watchpoint 1 hit:
old value: kGrid
new value: kCover
```

### 添加Python script

- LLDB contains an embedded python interperter，The entire API is exposed through python scripting bindings:

```shell
(lldb) script print(sys.version)
2.7.5 (default, Aug 25 2013, 00:04:04) 
[GCC 4.2.1 Compatible Apple LLVM 5.0 (clang-500.0.68)]
```

- The script command parses raw Python commands:

```shell
(lldb) script
Python Interactive Interpreter. To exit, type 'quit()', 'exit()' or Ctrl-D.
>>> a = 3
>>> print a
3
```

- Run python scripts from a breakpoint
	- LLDB creates a Python function to encapsulate the scripts
	- if yout want to access the script variables outside the breakpoint,you must declare thhem as global variables

- 通过打断点来调用python函数：

```shell
(lldb) breakpoint command add -s python 1(break point的id)
(lldb) breakpoint command add -s python
Enter your Python command(s). Type 'DONE' to end.
def function(frame,bp_loc,internal_dict):
    """frame: the SBFrame for the location at which you stopped
       bp_loc: an SBBreakpointLocation for the breakpoint location information
       internal_dict: an LLDB support object not to be used"""

	//填写python函数体：
	variables = frame.GetVariables(False,True,False,True)
	for i in range(0,variables.GetSize()):
		variable = variables.GetValueAtIndex(i)
		print variable
		DONE
```

- 将方法定义在python脚本中：

```shell
def breakpoint_func(frame,bp_loc,dict):

frame : current stack frame of the breakpoint
bp_loc : the current breakpoint location
dict : the python session dictionary

#函数返回false，则lldb继续执行
#函数返回true，则lldb暂停执行
```

- import exsting scripts to be used during your debugging session
	- 将python文件引入进来`(lldb) command script import ~/my_script.py`
	- 将breakpoint和python函数关联起来：`(lldb) breakpoint command add -F my.breakpoint_func`
	- `command script import "/Users/moxinxt/Desktop/tbcity-ipad/iCoupon4Ipad/iCoupon4Ipad/breakpoint.py"`

- 使用Facebook [Chisel](https://github.com/facebook/chisel)

### Attach到某个进程

我们也可以使用LLDB来debug某个进程：`process attach --name WWDCDemo --waitfor` //等待WWDCDemo启动，并将lldb挂到WWDCDemo上
	
## debug二进制

### Clang

- `clang -c test.c`
	- `-E` Preprocess,but don't compile
	- `-S` Compile, but don't assemble
	- `-c` Asseble, but don't link

- `clang -o test test.c` 编译`test.c`，生成`test`
- `clang -o test test.m -framework foundation` 带上link的framework

###  Lipo

- `lipo -info` 二进制包
- `lipo xx.a -thin arm64`

### otool

otool用来查看Mach-O文件格式
- `otool -h`:查看头部信息
- `otool -l`:查看LOAD CMD
- `otool -t`:Dump binary information
- `otool -t -V`: Disassmble the binary

### dSYM & DWARF

XCode自带了导出二进制符号表的工具:`dsymutil`

- 提取.dSYM:`dsymutil xx.app/xx -o xx.dSYM`
- 查看dSYM中的UUID: `dwarfdump --uuid VZMachOTools.dSYM/Contents/Resources/DWARF`
- 查看dSYM中的info信息: `dwarfdump -e --debug-info VZMachOTools.dSYM/Contents/Resources/DWARF > info.txt`
- 查看dSYM中的line信息: `dwarfdump -e --debug-line VZMachOTools.dSYM/Contents/Resources/DWARF > line.txt`

### Security & CodeSign

`Security`用来执行与证书签名相关的命令：
- 显示出所有可以用于签名的证书：`$ security find-identity -v -p codesigning `

### lldb

		 
### optool

- [官方文档](https://github.com/alexzielenski/optool)
- 作用：
	- appstore 二进制 去壳：`./optool [-w] -t 二进制`
	- 向二进制中注入dylib 