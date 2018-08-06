---
layout: post
title: XCode的一些技巧
list_title: XCode的一些技巧 | XCode Tips
categories: [XCode,UNIX]
---

## LLDB

### 资料

- [官方文档](http://lldb.llvm.org/tutorial.html)

### 常用命令

- 列出和某个命令相关的命令：`apropos thread`
- 列出所有break points: `(lldb) br l`
- 删掉break: `br delete 1`
- 开关break: `br e 1`,`br di 1`
- 设置break point: `b MyViewController.m :30`
- 继续:`c`,下一条:`n`,进入某个函数:`s`
- 结束:`finish`
- `expr`:

```
//隐藏view
expr self.view.hidden = yes

//variable:
expr int $a = 100 定义一个a变量值为100
```

### 增加符号断点

- 给所有viewdidload打断点:`br set -n viewDidLoad`
- 根据condition打断点：`br mod -c "totalValue > 1000" 3`
- 进入命令行模式：

```
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
- `frame variable`:

```

(lldb) frame variable
(TBCityStoreMenuViewController *const) self = 0x09646820
(SEL) _cmd = "onCoverFlowLayoutClicked:"
(UIButton *) btn = 0x09665d40
(lldb) frame variable self
(TBCityStoreMenuViewController *const) self = 0x09646820
```
- `watchpoint list`
- `watchpoint delete 1`

```
//监控变量_x是否发生变化
watchpoint set variable _x

(lldb) watchpoint set variable _type
Watchpoint created: Watchpoint 1: addr = 0x0955c8d0 size = 4 state = enabled type = w
    watchpoint spec = '_type'
    new value: kGrid
```
如果x发生变化：

```
Watchpoint 1 hit:
old value: kGrid
new value: kCover

```

### 添加Python script

- LLDB contains an embedded python interperter，The entire API is exposed through python scripting bindings:

```

(lldb) script print(sys.version)
2.7.5 (default, Aug 25 2013, 00:04:04) 
[GCC 4.2.1 Compatible Apple LLVM 5.0 (clang-500.0.68)]

```

- The script command parses raw Python commands:

```

(lldb) script
Python Interactive Interpreter. To exit, type 'quit()', 'exit()' or Ctrl-D.
>>> a = 3
>>> print a
3
>>> 
```

- Run python scripts from a breakpoint

	- LLDB creates a Python function to encapsulate the scripts

	- if yout want to access the script variables outside the breakpoint,you must declare thhem as global variables

- 通过打断点来调用python函数：

```
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

```
def breakpoint_func(frame,bp_loc,dict):

frame : current stack frame of the breakpoint
bp_loc : the current breakpoint location
dict : the python session dictionary

函数返回false，则lldb继续执行
函数返回true，则lldb暂停执行
```


- import exsting scripts to be used during your debugging session

	- 将python文件引入进来`(lldb) command script import ~/my_script.py`
	- 将breakpoint和python函数关联起来：`(lldb) breakpoint command add -F my.breakpoint_func`
	- `command script import "/Users/moxinxt/Desktop/tbcity-ipad/iCoupon4Ipad/iCoupon4Ipad/breakpoint.py"`

### 使用Chisel

- [Github](https://github.com/facebook/chisel)


### debug二进制

- 新版的macos在shell里面直接输入`lldb`即可启动lldb

- debug某个进程：`process attach --name WWDCDemo --waitfor` //等待WWDCDemo启动，并将lldb挂到WWDCDemo上
- 设置断点：`breakpoint set --func_regex fact`

## dSYM & DWARF

XCode自带了导出二进制符号表的工具:`dsymutil`

- 提取.dSYM:`dsymutil xx.app/xx -o xx.dSYM`

- 查看dSYM中的UUID: `dwarfdump --uuid VZMachOTools.dSYM/Contents/Resources/DWARF`

- 查看dSYM中的info信息: `dwarfdump -e --debug-info VZMachOTools.dSYM/Contents/Resources/DWARF > info.txt`

- 查看dSYM中的line信息: `dwarfdump -e --debug-line VZMachOTools.dSYM/Contents/Resources/DWARF > line.txt`

## Security & CodeSign

`Security`用来执行和证书，签名相关的命令：
- 显示出所有可以用于签名的证书：`$ security find-identity -v -p codesigning `

## Clang

- `clang -c test.c`
	- `-E` #preprocess,but done compile
	- `-S` #Compile, but don't assemble
	- `-c` #Asseble, but don't link

- `clang -o test test.c` 编译`test.c`，生成`test`
- `clang -o test test.m -framework foundation` 带上link的framework
		 

## Lipo

- `lipo -info 二进制包`
- `lipo -detailed_info 二进制包`

## otool

otool用来查看Mach-O文件格式
- `otool -h`:查看头部信息

## optool

- [官方文档](https://github.com/alexzielenski/optool)

- 作用：
	- appstore 二进制 去壳：`./optool [-w] -t 二进制`
	- 向二进制中注入dylib 