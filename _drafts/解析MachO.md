# 代码注入

## 先介绍一种debug iOS 二进制的方法

最近工作需要，为了排查一些诡异问题，使用了一种的debug iOS二进制的方法（注：该方法的原创不是我，是另一个同事），思路大概是这样的：

- 对于iOS企业包,MachO中的`cryptid`为`0`,没有“壳”(Fairplay),可以随意用`class-dump`来dump头文件,二进制可以直接拿来debug

- 向MachO中注入`dylib`。原理是在MachO Load Commands中插入一条`LC_LOAD_DYLIB`命令拉起注入的代码，方法是直接用了[optool](https://github.com/alexzielenski/optool)的`install`命令

- 将注入`dylib`后的二进制替换掉原来的二进制

- 在`dylib`中写代码hook出错的方法，或者使用lldb打符号断点，再用`expr`，`po`等命令来排查问题

这种方法主要针对于iOS企业包的调试，因此并不需要越狱的机器，对于appStore的包, 则需要先对其脱壳后再注入代码。


## MachO

MachO是iOS的二进制文件，里面包含代码的可执行文件和load和link规则，





## AppStore去壳 