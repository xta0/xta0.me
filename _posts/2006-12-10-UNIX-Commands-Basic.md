---
layout: post
title: UNIX Commands
categories: [unix,cheatsheet,shell]
tags: [Shell,UNIX]
---

<em>所有文章均为作者原创，转载请注明出处</em>

## 常用命令


- `echo`：输出字符
	- `echo hello world > ~/text`: `>`为输出流标志，输出文本到text中

- `man`: 命令手册

- 目录操作:	
	
	- `ls`：查看目录下文件
		- `ls -al`:查看隐藏文件
		- `ls -l`：长列表形式显示
		- `ls ~/c*`:只显示`c`开头的文件
		- 别名：`ll`

	- `cd`：进入目录
	- `find dir file`:在`dir`的路径中查找`file`
	- `pwd`: 当前路径

- 文件操作:

	- `touch`:创建空文件
	
	- `cat/more/less`：查看文件
		- `cat`：查看短文件
		- `more/less`：查看长文件
		- `less`：
		 	
	- `cp`：copy文件
	- `mv`：移动文件,重命名
	
	- `rm`：删除文件
		- `rm`：删除文件
		- `rm -d`：删除空文件夹
		- `rm -rd`：强制递归删除文件夹中的文件
		- `rm -rf`
		- `rm -rf test1/test2 -v`：查看删除过程

- 别名
	- `alias ll` = `ls -alF`  

##使用GCC##

.c编译完成后会生成.o文件，多个.o文件和一些lib，一起link得到可执行文件，下面是GCC的一些编译选项：

- gcc：直接执行% gcc  main.c 会得到 一个 a.out的可执行二进制文件。运行时需要带路径：% ./a.out

-  -c : 生成目标文件.o。例如得到main.o：

```
% gcc -c main.c
```

使用<code>nm main.o</code>

可以看到文件内容

- -o : 生成可执行文件。例如得到可执行文件p：

<code> % gcc -o p main.o module1.o module2.o </code>

<code> % gcc -o p main.o module1.c module2.c </code>

- -g : 编译器在输出文件中包含debug信息。例如:<code> % gcc -g main.c</code>

- -Wall:编译器编译时打出warning信息,强烈推荐使用这个选项。例如:<code> % gcc -Wall main.c</code>

- -I<em>dir</em>: 除了在main.c当前目录和系统默认目录中寻找.h外，还在dir目录寻找，注意，dir是一个绝对路径。

例如：<code> % gcc main.o ./dir/module.o -o p </code>



下面我们看一个完整的例子：

----./test/main.c , main.h  ,  module_1.h  ,  module_1.c  

----./test/ext/module_2.h  ,  module_2.c

main.h : 

```c 
 #include <stdio.h>;

int main(void);

```

main.c:

```c

 #include "main.h" 
 #include "module_1.h"
 #include "module_2.h"

int  main(void)
{
	printf("hello world");
	
	int ret1 = module_1_Func(100,20); 
	printf("\n%d\n",ret1);
	
        int ret2 = module_2_Func(200,100);
	printf("\n%d\n",ret2);
	

	return 0;
}
```

下面我们要编译出main.c的可执行文件p：

- 使用-o选项，一行搞定：

<code> % gcc -o p main.c module_1.c ./ext/module_2.c -I./ext </code>

- 先单独编译成.o，在link

	- <code> % gcc -c module_1.c </code>生成module_1.o

	- <code> % cd ./ext	% gcc -c module_2.c </code>生成module_2.o

	- <code> % gcc -c main.c -I./ext </code>生成main.o

	- <code> % gcc -o p main.o module_1.o ./ext/module_2.o </code>生成p


## 关于NM命令

命令格式:`nm[-AaefgnoPprsuvx][-t format]`

`nm`用来查看目标文件中的符号，目标文件包括：

- `.obj`结尾的文件，可能是Object Module Format格式或者是Common Object File Format格式
- `.lib`结尾文件，包含一个或多个`.obj`文件
- Windows可执行文件`.exe`

注意，`nm`命令不会列出DLL的entry point，除非有和它关联的符号表。

默认情况，`nm`列出按照字母顺序列出符号类型:

- A :absolute symbol, global
- a :absolute symbol, local
- B :uninitialized data (bss), global
- b :uninitialized data (bss), local
- D :initialized data, global
- d :initialized data, local
- F :file name
- l :line number entry (see -a option)
- N :no defined type, global; this is an unspecified type, compared to the undefined type U
- n :no defined type, local; this is an unspecified type, compared to the undefined type U
- S :section symbol, global
- s :section symbol, local
- T :text symbol, global
- t :text symbol, local (static)
- U :undefined symbol
- ? :unknown symbol

<h3>Further Reading:</h3>

- <a href="http://cslibrary.stanford.edu/107/UnixProgrammingTools.pdf">UNIX Programming Tools</a>
- <a href="https://www.mkssoftware.com/docs/man1/nm.1.asp">nm command</a>
