---
layout: default
title: Vim Cheat Sheet
categories: [unix,cheatsheet,shell,vim]
meta: Vim简明操作
---

 
##Vim Cheet Sheet

###模式

- Normal Mode: 默认模式，通过`ESC`可以到达这个模式
- Insert Mode: 编辑模式，通过`I`到达这个模式
- Visual Mode: 阅读模式，通过`V`到达这个模式

### 保存修改

- `:q`: 退出
- `:q!`: 不保存退出
- `:e dir`: 打开文件
- `w`: 写保存
- `w!`: 强行写保存
- `:set number`设置行号

### 移动

- `f<char>`: 光标移动到下一个`char`的位置
- `F<char>`: 光标移动到前一个`char`的位置
- `t<char>`: 光标移动到下一个`char`的前一个位置
- `T<char>`: 光标移动到前一个`char`的后一个位置
- `w` 光标移动到下一个词
- `b` 光标移动到前一个词的开始位置
- `e` 光标移动到当前或下一个词的末尾
- `gg` 光标移到文件开始
- `G` 光标移到文件末尾
- `{` 光标移到段落开始
- `}` 光标移到段落末尾 
- `)` 移动到下一句
- `(` 移动到上一句
- `^E` 向下滚动

### 复制粘贴

在Visual模式下:

- `yy`复制光标所在行
- `2yy`复制2行数据
- `yw`复制光标所在字符
- `d`在光标的位置粘贴内容

### 撤销

- `u`

### 删除

- `d` + `w` 删除当前词
- `d` + (number) + `w` 删除光标开始出number个词
- `dd`或`shift+d` 删除当前行
- `5dd`删掉5行数据
- `dt+(word)`删除光标到word之间的字符
- `x` 删除光标位置的字符
- `X` 删除光标前一个位置的字符

###撤销 

- `u` 撤销上一次修改
- `U` 撤销所有修改

###进入编辑模式

- `s` 在光标处，删除一个词，并 进入编辑模式
- `a` 在光标的下一个位置进入编辑模式
- `o` 在光标处开启下一行进入编辑模式

###重复上次的命令

- `.`

###搜索

- Visual模式下:`/`+ 关键字，向后搜索
- Visual模式下:`?`+ 关键字，向前搜索
- `n`下一条搜索结果, `N`前一条搜索结果

###查找并批量替换

- 使用`sed`
- 默认对当前行查找

###文件内定位

- `:set number`显示出行号
- `G`跳转到最后一行
- `xG`跳转到第x行
- `ctrl-d`先下滚动半屏
- `ctrl-u`向上滚动半屏

###屏幕内定位

- `H`跳到屏幕开始的位置
- `M`跳到屏幕中间位置
- `L`跳到屏幕最后的位置


## 参考资料

- [Learn VIM Progressively](http://yannesposito.com/Scratch/en/blog/Learn-Vim-Progressively/)


---

layout: default
categories: UNIX
tags: [Tools,UNIX]
title: Emacs Cheet Sheet

---


##Emacs Cheet Sheet

- [文档](http://www.gnu.org/software/emacs/tour/)

### 常用命令：

- C-x C-c: 退出
- C-g: 撤销当前指令
- C-x C-f: 打开文件
- C-x C-s: 保存
- 进入命令行: C-c, C-x
	- 使用命令行load文件:use "section1.sml";
	
	