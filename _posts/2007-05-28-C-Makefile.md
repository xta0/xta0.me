---
title: C Makefile
layout: post
---

## C Makefile

当一个工程很大，有很多文件时，使用gcc去编译就局限了。这个时候通常使用makefile，makefile中，需要把这些文件组织到一起。
makefile是一个纯文本文件，实际上它就是一个shell脚本(关于Shell脚本可以参考[这里]())，并且对大小写敏感，里面定义一些变量。重要的变量有三个：

- CC ： 编译器名称
	- `% gcc  main.c` 在makefile中的写法为：`$(CC) main.c`
- CFLAGS ： 编译参数，也就是上面提到的gcc的编译选项。这个变量通常用来指定头文件的位置，常用的是-I, -g。
	- `%gcc -g -I./ext -c main.c ` 在makefile中的写法为：`CFLAGS = -g -I./ext`
- LDFLAGS ：链接参数，告诉链接器lib的位置，常用的有-I,-L，

一个简单的makefile：

```
CC = gcc
main:main.c main.h
$(CC) main.c
```

必须要包含这三部分

main.o: main.c main.h

这句话的意思是main.o必须由main.c，main.h来生成

`$(CC)main.c`

是shell命令，前面必须加tab

针对上面的例子，我们可以写一个makefile 文件


```
C = gcc
CFLAGS = -g -I./ext/

PROG = p
HDRS = main.h module_1.h ./ext/module_2.h
SRCS = main.c module_1.c ./ext/module_2.c

$(PROG) : main.h main.c module_1.h ./ext/module_2.h
	$(CC) -o $(PROG) $(SRCS) $(CFLAGS)

```