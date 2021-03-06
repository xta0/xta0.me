---
layout: post
list_title: Java基础 | 概述 | Language Overview 
title: Java概述
categories: [Java]
---

## Java 概述

### 发展过程

- 1990年 SUN "Green"--开发家用电器软件
- 1994年 改进C++,发明Oka语言
- 1995年, Java语言
	- SUN公布第一版编译器JDK1.0
	- Sun：Stanford University Network 
- 1998年 Java2, JDK 1.2
- 2000-2006年，改进 JDK 1.6
	- JDK 1.4
		- assert、logging、Java2D、NIO、正则表达式
	- Java 5
		- 泛型、增强的foreach、自动装箱拆箱、枚举、可变长参数、静态引入、注记、printf、StringBuilder
	- Java 6
		- Compiler API(动态编译)、脚本语言支持、WebService支持
StringBuilder
- 2010 Oracle收购SUN
- 2011 JDK 1.7 改进
	- Java 7
		- 常量等的写法、带资源的try、重抛异常
- 2014 JDK 1.8 较大改进
	- Java8 
		- Lambda Expressions

### Java分为三大平台

- Java SE 标准版（J2SE，Java 2 Platform Standard Edition）
	- 独立应用，PC应用
- Java EE 企业版（J2EE，Java 2 Platform, Enterprise Edition）
	- Web应用
- Java ME 微型版（J2ME，Java 2 Platform Micro Edition）
	- 手机应用


### Java 语言概述

- OOP
- 语法结构和C/C++类似，但是更简单
	- 无指针操作
	- 自动内存管理
	- 数据类型长度固定
	- 不用头文件
	- 不包包含struct和union
	- 不支持宏
	- 不用多重继承
	- 无全局变量
	- 无goto
- 特点
	- 纯面向对象，没有独立于对象之外的函数
	- 平台无关
	- 安全稳定

### Java三种核心机制 

- Java 虚拟机(Java Virtual Machine)
	- Java虚拟机(JVM)读取并处理经编译过的字节码class文件 。
	- Java虚拟机规范定义了：
		- 指令集
		- 寄存器集
		- 类文件结构
		- 堆栈
		- 垃圾收集堆
		- 内存区域 
- 代码安全性检测(Code Security)
 	- JRE (The Java Runtime Environment) = JVM + API（Lib )
	- JRE运行程序时的三项主要功能：
		- 加载代码：由class loader 完成；
		- 校验代码：由bytecode verifier 完成；	
		- 执行代码：由 runtime interpreter完成。
	- 垃圾收集机制(Garbage collection) 
		- 系统级线程跟踪存储空间的分配情况
		- 在JVM的空闲时，检查并释放那些可被释放的存储器空间
		- 程序员无须也无法精确控制和干预该回收过程 

## Java程序构成

### 基本构成

- `package`语句
- `import`语句
-  文件名就是类名
- 一个文件中只能有一个public类

## 编译Java代码

- `classpath`: 源码路径
- `path`: java命令路径
- 编译：
	- `javac main.java` -> `main.class`
	- 命令行指定classpath： `javac -cp libxx.jar`
	- 批量编译: `javac -d classes src/chap1/*.java src/chap2/* .java`

- 运行:
	- `java main`
	- 命令行指定classpath: `java -cp libxx.jar`

### 使用Jar打包

- 编译 `javac A.java`
- 打包 `jar cvfm A.jar A.man A.class`
	- `c`表示创建create， `v`表示详情，`f`表示指定文件名，`m`表示清单信息文件
	- `A.man`是manifest:
		- `Mainfest-Version:1.0`
		- `Class-Path:.`
		- `Main-Class:A` 
	- 清单文件可以任意命名，常见的是`MANIFEST.MF`

### 使用JavaDoc生成文档

- `javadoc -d`目录名 xxx.java
- `/** */`标记格式
	- `@author`:  表明开发该模块的作者
	- `@version`: 类说明，版本
	- `@see` 参考其他类
	- `@return` 方法返回值
	- `@param` 方法参数说明
	- `@exception` 抛出异常的说明

### 使用javap

- 使用`javap`查看类的信息
	- `javap`类名
- 使用`javap`反汇编
	- `javap -c`类名 

## 命令行输入输出

### java.util.Scanner

- 使用`nextInt()`
- 使用`nextDouble()`
- `next()`得到下一个单词

### java.io

- `System.in.read()`	
- `System.out.print()`或者 `println,printf`

## Resources

- [Core Java Volumn 1](http://ptgmedia.pearsoncmg.com/images/9780137081899/samplepages/0137081898.pdf)
- [Core Java Volumn 2](http://ptgmedia.pearsoncmg.com/images/9780137081608/samplepages/013708160X.pdf)
