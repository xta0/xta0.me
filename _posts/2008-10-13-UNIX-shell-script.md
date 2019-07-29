--- 
list_title: UNIX 脚本语法 | Shell Script Basics
layout: post
title: UNIX Shell Script 
categories: [UNIX]
---

# Unix Shell Part 3

## Shell Scripts

- [Overview](#1)
- [变量](#2)
- [字符串操作](#3)
- [数学运算](#4)
- [浮点运算](#5)
- [条件判断](#6)
- [循环语句](#7)
- [函数](#8)
- [输入输出](#9)
- [Shell运行控制](#10)
- [Wildcards](#11)
- [Log&Debug](#12)
- [UNIX环境](#13)


<h2 id="1">OverView</h2>

###  历史和版本

- 第一个：Thompson shell: Unix shell 1971@Bell Lab
- 标准的Unix Shell: Bourne shell 1978@Bell Lab
- Linux发行版的默认shell:bash shell(Bourne-Again shell)，其他shell类型：ash,korn,tcsh,zsh,dash
- 查看当前系统使用的Shell:`echo $SHELL`	
- 查看系统支持的shell: `cat /etc/shells `

	```
	/bin/bash
	/bin/csh
	/bin/ksh
	/bin/sh
	/bin/tcsh
	/bin/zsh
	```

###  Shell分类

- 交互式Shell：在终端上执行，shell等待你的输入，并且立即执行你提交的命令。
- 非交互式Shell：以shell script(非交互)方式执行。在这种模式 下，shell不与你进行交互，而是读取存放在文件中的命令,并且执行它们。当它读到文件的结尾EOF，shell也就终止了

查看是否是交互式shell，可以通过打印`$-`变量的值（代表着当前shell的选项标志），查看其中的“i”选项（表示interactive shell）来区分交互式与非交互式shell。

```
> echo $-
569JNRXZghiklms #交互式

> ./test.sh
echo $-
hB #非交互式
```

或者使用环境变量

```
if [[ -n $PS1 ]]; then
    : # These are executed only for interactive shells
    echo "interactive"
else
    : # Only for NON-interactive shells
fi
```

- 登录Shell: 是需要用户名、密码登录后才能进入的shell（或者通过”–login”选项生成的shell）。通过执行`logout`或`exit`命令退出shell。bash是 login shell 时，其进程名为`-bash` 而不是`bash`。 可通过`echo $0`查看
	
- 非登录式Shell: 不需要输入用户名和密码即可打开的Shell，例如：直接命令“bash”就是打开一个新的非登录shell。通过`exit`退出shell，bash是非login shell 时，其进程名为`/bin/bash` 而不是`-bash`。 可通过`echo $0`查看

查看是否是login shell，LINUX下可以使用`shopt`命令:

```
if shopt -q login_shell ; then
    : # These are executed only when it is a login shell
    echo "login"
else
    : # Only when it is NOT a login shell
    echo "nonlogin"
fi
```

注：`:` — 脚本中这个符号没有意义，只为了填充空行而使用

###  Shebang
- `#!/bin/bash`
- 如果不指定`shebang`，系统用默认的shell


<h2 id="2">变量</h2>

- 用户自定义的变量
- 系统环境变量
	- 局部环境变量
	- 全局环境变量

###  自定义变量规则

- 合法字符，数字，字母，下划线
-  <= 20个字符
-  大小写敏感
- 声明：变量名=变量值，**注意等号前后没有空格，变量名通常用大写** 如`VAL=2`
- 访问: `$VAL `
- 变量使用：
	- 如果变量在单引号内则原封不动的输出
	- 如果变量在双引号内，使用`$VAL`或`${VAL}`，解析出变量的值输出

```
> val="bash"
> echo 'I like the $val shell' -> I like the $val shell
> echo "I like the $val shell" -> I like the bash shell
>  echo "I am ${val}ing on my keyboard" -> I am bashing on my keyboard
```
- 使用反引号或者括号将变量的内容返回

```
> date
2007年 7月12日 星期日 16时48分21秒 CST
> VAL=date
> echo $VAL //输出date

//使用反引号: 
> VAL=`date`
> echo $val //输出date的内容：2007年 7月12日 星期日 16时49分32秒 CST

//使用括号
> VAL=$(date)
> echo $val //输出date的内容：2007年 7月12日 星期日 16时49分32秒 CST
```
- 格式化字符串:

```
>var=`date +%Y%m%d`
>echo $var //20150712
```

- 接受用户输入作为参数:`read -p "PROMPT" VARIABLE`

	
<h2 id="3">字符串操作</h2>

- 字符串链接
- 字符串长度
- 查找字符串中的字符位置
- 字符串截断
- 字符串匹配

###  字符串的连接

- 将字符串写到一起即可连接字符串

```
>str=hello     
>str1=world    
>str=$str$str1
>echo $str  --> helloworld

>str=$str" "$str1
>echo $str --> helloworld world
```
- 使用`""`做字符串拼接

```
str="Test $str"
echo $str
Test hello:
```

###  字符串常用操作

- 查看字符串长度:`expr lenght "$str"`
- 查找字符串中字符位置: `expr index $str CHARs`
	- `expr index "abcd" 'b'`索引从1开始
- 获取子字符串:`expr substr POS LENGTH`
	- `expr substr "$str" 6 10` 
- 字符串匹配:
	- `expr $str: REGEXP`
	- `expr "$str" : '.*\([0-9]\{5\}\)'` 匹配`$str`中的数字部分，长度为5。注意`:`左右要求有空格，特殊字符要`\`转义。
	- `expr match $str REGXP`


<h2 id="4">数学运算</h2>

###  逻辑运算

- `& |  < > = != <= >=`

###  数值运算

- `+ - * / % ()` 得到的结果都是整数

###  运算表达式

- `expr expression`
- `result=$[expression]`

主要使用第二种

```
> num1=7 
> num2=13
> num3=7
> result=$[num1 < 19]
> echo $result --> 1

> result=$[num1 == num3] 
> echo $result --> 1
```

<h2 id="5">浮点运算</h2>

- 内建计算器bc，bc能够识别：
	- 数字（整形，浮点型）
	- 变量
	- 注释（以`#`开始的行）
	- 表达式
	- 编程语句（如条件判断: `if-then-else`）
	- 函数

```
> bc
bc 1.06
Copyright 1991-1994, 1997, 1998, 2000 Free Software Foundation, Inc.
This is free software with ABSOLUTELY NO WARRANTY.
For details type `warranty'. 
12.5 * 0.3
3.7
100/3
33 //默认scale =0，则没有小数点后的余数
scale=4
100/3
33.3333

```

使用变量

```
num=12
num1=3;num2=4;
num1/num2
.7500

```

###  脚本中使用`bc`

- 使用管道命令，管道将前一个命令的输出作为后一个命令的输入

```shell
var = `echo "options; expresssion" | bc`
```
上面的例子，将`options;expression`作为`bc`的输入

```shell
> var=`echo "scale=4;3.44/5" | bc`
> echo $var
.6880
```

- 使用管道没法进行复杂的运算，还有一种方式是使用输入重定向:

```shell
result=`bc << EOF
options
statements
expressions
EOF
```

例如：

```shell
> result=`bc << EOF
bquote> scale=4
bquote> var1=20
bquote> var2=3
bquote> var1/var2
bquote> EOF
bquote> `
> echo $result
6.6666
```

<h2 id="6">条件判断</h2>

- 条件
- 循环

###  `if-then`语句

- 单条件:

```shell
if command

then 
	commands
elif
	command2
then
	commands
else
	commands
fi
```

当`if`后面的`command`返回`0`时，执行`then`后面的语句:

```shell
#!/bin/sh

if date
then
echo "worked"
fi
```

###  条件判断

- 三类条件：
	- 数值比较
	- 字符串比较
	- 文件比较

- 使用`test`命令

```shell
if test condition
then
	commands
fi
```
- 使用`[]`，注意condition前后有空格

```shell
if [ condition1 ]
then
	command N
elif [ condition2 ]
then
	command Q
else
	command M
fi
```
###   常见的条件判断

- 与或:
	- `[ condition1 ] &&[ condition2 ]`
	- `[ condition1 ] || [ condition2 ]`
	
- 数值比较
	- 相等:`n1 -eq -n2`
	- 大于：`n1 -gt -n2`
	- 小于：`n1 -lt -n2`
	- 大于等于：`n1 -ge n2`
	- 小于等于：`n1 -le n2`
	- 不等于：`n1 -ne n2` 

```shell
#! /bin/sh
echo "input number: "
read p
if [ $p -gt 10 ]
then
 echo "the number is greater than 10"
fi
if [ $p -lt 10 ] && [ $p -gt 1 ]
then
 echo "the number is less than 10 and greater than 1"
fi
```
- 字符串比较:
	- 相同:`str1 = str2`
	- 不同：`str1 != str2`
	- 大于：`str1 > str2` //比较asc码
	- 小于: `str1 < str2`
	- 是否长度非0: `-n str1`
	- 是否长度为0: `-z str1`

```

#! /bin/sh

str1="ab"
str2="bc"

if [ $str1 \< $str2 ]//转义字符
then
 echo "str1 is less than str2"
elif [ $str1 \> $str2 ]
then
 echo "str1 is greater than str2"
elif [ $str1 = $str2 ]
then
 echo "str1 is equal to str2"
fi

```

- 文件比较
	- 检查file是否存在，并且在同一个目录: `-d file`
	- 检查file是否存在: `-e file`
	- 检查file是否存在，并且是一个文件: `-f file`
	- 检查file是否存在，并非空: `-s file`
	- 检查file是否存在，是否是可执行文件: `-x file`

	- 检查file1比file2新：`file1 -nt file2`
	- 检查file1比file2旧：`file1 -ot file2`


- 高级判断

	- `(( expression ))` 高级数学表达式
	- `[[ expression ]]` 高级字符串比较, 正则表达式匹配

```

#! /bin/sh

val1=10

if (( $val1 ** 2 > 90 ))
then
	(( val2 = $val1 ** 2))
	echo "The square of $val1 is $val2"
fi

```

```

#! /bin/sh

if [[ $USER == m* ]]
then
ehco "The current use is $USER"
fi


```

###  case语句

- case语法

```

case "$VAR" in 
	pattern_1)
		# Commands go here
		;;
	pattern_N) 
		# commands go here
		;;

	*) # default case
		# commands go here
	 	;;
esac

```

- 使用数字

```

#! /bin/sh
echo "input a number: "
read num

case $num in
1)
 echo "The month is January";;
2)
 echo "The month is Feb";;
*)
 echo "The month is unknown";;
esac

```

- 使用字符串

```

#! /bin/sh
echo "input YES|NO: "
read var

case $var in

YES | yes | y | Y)
 echo "You said YES"
 echo "OK!"
 ;;

NO | n | N | no)
 echo "You said NO";;
*)
 echo "The input is unknown";;
esac  
  
```

<h2 id="7">循环</h2>

- `for,while,until,break,continue`

### for循环

- 语法1：将list中的每一项赋值给val

```

for VARIABLE_NAME in ITEM_1 ITEM_N
do
	command 1
	command 2
	command N
done

```

- demo1:

```

#!/bin/bash
for COLOR in red green blue
do
	echo "COLOR: $COLOR"
done

--------------------
output:

> COLOR:red
> COLOR:green
> COLOR:blue

```

- demo2:重命名当前目录下的图片

```

#!/bin/bash

PICTURES=$(ls *jpeg)
DATE=$(date +%F)

for PICTURE in $PICTURES
do
   echo "Renaming ${PICTURE} to ${DATE}-${PICTURE}"
   mv ${PICTURE} ${DATE}-${PICTURE}
done

```

- 语法2：使用`((...))`

```

for(( i=1; i<10; i++))
do 
	commands
done

```

### list来源

- 直接创建列表

```

for val in Jan Feb Mar Apr May
do
  echo "month name is $val"
done

```
- 从变量读取

```

list="Jan Feb Mar Apr May"
for val in $list
do
  echo "Month name is $val"
  done

```
- 从命令读取

```

for val in `cat monthlist`
do
  echo "Month name is $val"
done

```
- 从目录读取

```

#文件通配符
for val in ~/Documents/Coding/Shell/*
do
echo "$val"
done  
   
```

### shell中默认的分隔符

- 空格
- 制表符
- 换行符
- 修改分隔符

	- `IFS=$";"` 以分号分割

### while和until

- while语法

```

while [ CONDITION_IS_TRUE ] command
do
	command 1
	command 2
	command N
done

```

- example:loop 9 times

```

#loop 9times

val=1
while [ $val -lt 10 ]
do
	echo "$val"
	val=$[ $val+1 ] # ((val++))
done

```

- exmpel: read input

```

while [ "$CORRECT" != "y" ]
do 
	read -p "Enter your name:"  NAME
	read -p "Is ${NAME} correct? " CORRECT
end

```

- example: Reading a file, line by line

```

LINE_NUM=1
while read LINE
do
	echo "${LINE_NUM}: ${LINE}"
	((LINE_NUM++))
done < /etc/fstab

FS_NUM=1
grep xfs /etc/fstab | while read LINE do #读xfs文件中的每一行
	echo "xfs: ${LINE}"
end

FS_NUM=1
grep xfs /etc/fstab | while read FS MP REST #读xfs，第一行放入FS变量，第二行放入MP，其它行放入REST
do
	echo "${FS_NUM}: file system: ${FS}"
	echo "${FS_NUM}: mount point: ${MP}"
	((FS_NUM++))
done 

```

- 使用continue

```

mysql -Bne 'show databases' | while read DB
do
	db-backed-up-recently $DB
	if [ "$?" -eq "0" ]
	then
		continue
	fi
		backup $DB
done
 
```

- until语法

```

until [ CONDITION_IS_TRUE ] command
do
	command
done

```

- example

```

val=1
util [ $val -eq 10 ]
do
echo "$val"
val=$[ $val+1 ]
done

```


<h2 id="8">函数</h2>

###  定义

- 定义1:

```

function name {
	commands
}

```

- 定义2

```

name() {
	commands
}

```

- 函数调用

```

#!/bin/bash

function hello(){
	echo "HELLO!"
	now #函数调用
}

#hello #错误，因为now还没被声明

function now(){
	echo "It's $(date+%r)"
}

hello #正确，函数调用，没有括号

```
- 函数参数传递: 
	- 引用命令行参数做变量：`> script.sh parm1 param2 param3`在脚本里怎么访问 
		- 所有参数 `$@`
		- 参数个数 `$#`
		- 第1个参数 `$1`
		- 第2个参数 `$2`

- example1:

```

#! /bash/bin

function hello(){
	echo "Hello $1"
}

hello Jason

#Output : Hello Jason

```

- example2:

```

#!/bin/bash

echo "Excuting script: $0"
for USER in $@ do
	echo "Achiving user: $USER"
	#lock the account
	passwd -l $USER
	#create an achive of the home dir
	tar cf /archives/${USER}.tar.gz /home/${USER}
end

```

- 函数的变量
	- 全局变量: 默认变量都是全局的 
	- 局部变量: 只能在函数内访问的变量，变量前加`local`，只有函数能使用local variable。
	- 定义在函数里的变量，只有当函数被执行后，变量才变成全局的
	
	```
	
	#!/bin/bash
	
	my_function(){
		GLOBAL_VAR=1
	}
	#GLOBAL_VAR not available yet.
	echo $GLOBAL_VAR
	my_function
	#GLOBAL_VAR is NOW available. 
	echo $GLOBAL_VAR
	
	```

- 行数返回值
	- 显式指定：`return <RETURN CODE>`
		- 0：success 
	- 默认上一条命令的执行结果 		
	- example:
	
	```
	
	function backup_file(){
		if [ -f $1 ]
		then
			local BACK="/tmp/$(basename ${1}).$(date +%F).$$" #$$代表当前running script的PID
			echo "Backing up $1 to ${BACK}"
			cp $1 $BACK
		else
			return 1
		fi
	}
	backup_file /etc/hosts
	if [ $? -eq 0 ]
	then
		echo "Backup succeeded!"
	else
		echo "Backup failed"
		exit 1
	fi
	
	```
	

- 引用其它脚本中的函数
	- `source filepath`或者`. filepath`




<h2 id="9">输入输出</h2>

###  输入输出重定向

- 输出重定向:  将输出重定向到文件中
	- `command > outputfile` //会覆盖原数据
	- `command >> outputfile` //会追加新数据到老数据后面

- 输入重定向: 将文件内容输入给命令
	- `command < outputfile` 

- 内联输入重定向

```
command << marker

data input

marker
```

例如：

```
>  Shell  result=`bc << EOF 
bquote> var1=3
bquote> var2=5
bquote> var3=var1+var2
bquote> print var3
bquote> EOF
bquote> `
> Shell  echo $result
8
```


###  文件描述符与错误重定向

每个进程都和三个系统文件 相关联：标准输入stdin，标准输出stdout、标准错误stderr，三个系统文件的文件描述符分别为0，1、2

文件描述符| 缩写| 描述|
---| -------|------|
0  | STDIN  | 标准输入|
1  | STDOUT | 标准输出|
2  | STDERR | 标准错误|

将错误保存到log中:

```
>cat iot 2 > errlog

```

- 临时重定向:将shell脚本中的错误输出到文件中:`command >& 文件描述符`

```
#! /bin/sh

echo "Test Error" >& 2
echo "Normal output"

```

执行脚本，错误会打印在命令行中，原因是console是默认的错误输出。如果要将shell中产生的错误输出到文件中，执行shell的时候要追加重定向：

```
>  ./iotest 2>> errlog

```

- 理解/dev/null

在类Unix系统中，`/dev/null`，或称空设备，是一个特殊的设备文件，它丢弃一切写入其中的数据，读取它则会立即得到一个`EOF`。 因此`/dev/null`也被称为黑洞。shell脚本中一种常见的写法是`command >/dev/null 2>&1`，它是一种当将command命令的输出或者错误重定向到黑洞里做法，所以这里2>&1 的意思就是将标准错误也输出到标准输出当中。举个例子：；

```
>cat test.sh
t
date

> ./test.sh > test1.log
> ./test.sh: line 1: t: command not found

> cat test1.log
Wed Jul 10 21:12:02 CST 2007
```

可以看到，date的执行结果被重定向到log文件中了，而t无法执行的错误则只打印在屏幕上。

```
> ./test.sh > test2.log 2>&1

> cat test2.log
./test.sh: line 1: t: command not found
Tue Oct 9 20:53:44 CST 2007

```

这次，stderr和stdout的内容都被重定向到log文件中了。


- 永久重定向: `exec 文件描述符>文件名`

```

#! /bin/sh

exec 1>output
exec 2>errlog

echo "Test Error" >& 2
echo "Normal output"

```

这样，输入输出都回重定向到文件中





###  管道

将一个命令的输出，重定向到另一个命令的输入:`command1 | command2`

- 将当前进程以翻页的形式查看:`ps -ef | less`
- 查找当前进程中的某个进程:`ps -ef | grep bash`

<h2 id="10">Shell运行控制</h2>

###  Linux中的信号

- 什么是信号
	- 软中断
	- 进程间异步通信
	- `man signal`

- 产生信号
	- 终止进程:`SIGINT`: CTRL+C
	- 暂停进程:`SIGSTP`:CTRL+Z 
	- `kill`/`kill all`
		- `kill -9`: `9`是`SIGKILL`，无条件终止进程
- 处理信号
	- 系统默认方式处理
	- 忽略信号
	- 自定义处理信号的方法
		- 使用`trap`捕获SIGINT：`trap "echo 'Signal Trapped SIGINT for ctrl+C'" SIGINT`
		- 移除添加的`trap`：`trap - SIGINT`
	 
###  后台运行的脚本

- 命令格式:	`SCRIPT &`。
	- 例如:`./bgtest &`：将脚本在后台运行
	- 查看所有运行中的作业:`jobs`
	- 在前台重启停止的作业:`fg 作业号`
		- 例如:`fg 1`
	- 在后台重启停止的作业:`fg 作业号`
		- 例如:`bg 1`

	- 作业优先级: -20(高) ~ +19（低） 
	- 设定优先级:`nice`
		- `nice -n 10 ./bgtest &` 
	- 重新设定优先级:`renice` 
		- `renice 10 -p pid` 

- 使用`nohub SCRIPT &`：当bash窗口退出时，不终止脚本执行

###  退出码

- 退出码:`$?`
	- 默认值为上一条命令的返回值 
	- 查看命令的返回值：`man command`
	- example: `> ls -al ; echo $?`

- return 返回值
	- 范围：0~255 
	- 0：成功
	- 非0：失败
	
```

HOST="google.com"
ping -c 1 $HOST
if[ "$?" -eq "0" ]
then
	echo "$HOST reachabe."
else
	echo "$HOST unreachable."
fi

```

- 自定义退出码

```

HOST="google.com"
ping -c 1 $ HOST
if[ "$?" -ne "0"]
then
	echo "$HOST unreachable"
	exit 1 #自定义退出码
fi
exit0

```

- 串联命令
	- 使用`;`，不管命令返回值，顺序执行
		- `> cp test.txt /tmp/bak/ ; cp test.txt /tmp`
	- 使用`&&`和`||`，后一条指令的执行依赖前一条指令的结果
		- `&&`：前一条执行成功后，后面才执行：`> mkdir /tmp/bak && cp test.txt /tmp/bak/`
		- `||`：前一条执行成功后，后面语句不执行: `> ping -c 1 "google.com" || echo "not reachable" ` 
	
<h2 id="11"> Wildcards </h2>

- A character or string used for pattern matching
	- Globbing expands the wildcard pattern into a list of files and directories
	- Wildcards can be used with most commands
	- ls,rm,cp

- Wildcards
	- `*` matches zero or more characters
		- *.txt //所有txt结尾的
		- a* //a开头的文件
		- a*.txt //a开头的txt文件
	- `?` matches exactly one character
		- ?.txt //匹配只有一个字母做文件名的txt
		- a? //
		- a?.txt 
	- `[]` A character class
		- 匹配括号里的字符，只匹配一个夫妇
		- ca[nt]* 匹配ca开头，第三个字母是n或者t的单词
			- can
			- cat
			- candy
			- catch
	- `[!]` 匹配所有不以括号内开头的单词，只匹配一个字符
		- [!aeiou]* 不以元音开头
			- basketball
			- cricket

	- `[-]` 匹配范围，通过中划线标记范围
		- [a-g]* 匹配所有以a-g开头的文件
		- [3-6]* 匹配所有3-6数字之间开头的文件
		- 系统支持的匹配格式：
			- [[:alpha:]] 匹配所有大小写字母 
			- [[:alnum:]] 匹配所有大小写字母+数字
			- [[:digit:]] 匹配数字
			- [[:lower:]] 匹配小写字母
			- [[:space:]] 匹配空格
			- [[:upper:]] 匹配大写字母

<h2 id="12"> Logs </h2>

###  Syslog

- The syslog standard uses facilities and serverities to categorize messages
	- Facilities: kern, user, mail, daemon, auth, local0, local7
	- Serverities: emerg, alert, crit, err, warning, notice, info,debug

- Log file locations are configurable
	- /var/log/messages
	- /var/log/syslog

###  Logging with logger
- The logger utility
- By default creates user.notice messages.

```

logger "Message"
logger -p local10.info "Message"
logger -t myscript -p local10.info "Message"
logger -i -t myscript "Message"

```  	

- example

```

logit(){
	local LOG_LEVEL=$1
	shift
	MSG=$@
	TIMESTAMP=$(date +"%Y-%m-%d %T")
	if[ $LOG_LEVEL = 'ERROR' ] || $VERBOSE 
	then
		echo "${TIMESTAMP} ${HOST} ${PROGRAM_NAME}[${PID}]: ${LOG_LEVEL} ${MSG}"
	fi
 }

> logit INFO "Processing data".

```

###  Debug

- Build in Debugging Help
	- `-x` Prints commands as they execute
	- After substitutions and expansions
	- Called an x-trace, tracing, or print debugging 
	- `#!/bin/bash -x`
	- `set -x`
		- `set +x` to stop debugging   
	
	- 使用 `-e`，当脚本出错时，自动exit
		- `#!/bin/bash -ex`
		- `#!/bin/bash -xe`
		- `#!/bin/bash -e -x`
		- `#!/bin/bash -x -e` 

	- 使用`-v` 打印出shell执行的script内容，可以和其它参数一起使用

- 脚本里使用 `-x`

```

#!/bin/bash -x

TEST_VAR="test"
echo "$TEST_VAR"

> ./debugging-01.sh #执行test脚本
+ TEST_VAR=test #调试信息
+ echo test
test

-------

#!/bin/bash

TEST_VAR="test"
set -x
echo $TEST_VAR
set +x
hostname

>  lessons ./debugging-02.sh   
+ echo test
test
+ set +x
JaysondeMBP.fios-router.home

```

- 脚本里使用`-e`

```

#!/bin/bash -ex

FILE_NAME="/not/here"
ls $FILE_NAME
echo $FILE_NAME

>  ./debugging-04.sh   
+ FILE_NAME=/not/here
+ ls /not/here
ls: /not/here: No such file or directory

```

- 脚本里使用`-v`

```

#!/bin/bash -vx
TEST_VAR="test"
echo "$TEST_VAR"

> ./debugging-06.sh   
#!/bin/bash -vx
TEST_VAR="test" #读入指令 -v
+ TEST_VAR=test #进行变量替换后指令 -x
echo "$TEST_VAR" #读入指令 -v
+ echo test #进行变量替换后指令 -x
test

```

###  Manaual Debugging

- You can create your own debugging code.
- Use a special variable like DEBUG
	- DEBUG=true
	- DEBUG=false

```
#!/bin/bash

DEBUG=true

if $DEBUG
then
  echo "Debug mode ON."
else
  echo "Debug mode OFF."
fi

```

- 几种使用DEBUG的方式

```
#!/bin/bash

DEBUG=true
$DEBUG || echo "Debug mode OFF." #log的方式

---

#!/bin/bash

DEBUG="echo"
$DEBUG ls #使用debug做echo

---

#!/bin/bash

#DEBUG="echo"
$DEBUG ls #使用debug做注释

``` 

- 自定义log函数

```

#!/bin/bash

debug() {
  echo "Executing: $@"
  $@
}

debug ls

```

<h2 id="13"> UNIX环境 </h2>


### Shell启动

- 对于Bash，系统从上到下按照以下顺序执行，先A，然后B（B1，B2，B3）,然后C

Option name         | Interactive login| Interactive non-login | Script |
--------------------|------------------|-----------------------|--------|
/etc/profile  		| A |   |  |
/etc/bash.bashrc    |   | A |  |
~/.bashrc   			|   | B |  |
~/.bash_profile     |B1 |   |  |
~/.bash_login       |B2 |   |  |
~/.profile          |B3 |   |  |
BASH_ENV            |   |   |A |
~/.bash_logout      | C |   |  |

更详细的流程图如下所示：

![](/assets/images/07/02/shell1.png)

对于Bash，最好把自己的配置文件写到`~/.bashrc`里，在`~/.bash_profile` 里面`source ~/.bashrc`。

- 对于Zsh(在`~/.zshrc`缺失的情况下会读取`~/.profile`)

Option name         | Interactive login| Interactive non-login | Script |
--------------------|------------------|-----------------------|--------|
/etc/zshenv 			| A |  A  | A  |
~/.zshenv     		| B |  B  | B  |
/etc/zprofile     	| C |     |    |
~/.zprofile 			| D |   |  |
/etc/zshrc			| E | C |  |
|~/.zshrc           | F | D |  |
/etc/zlogin         | G |   | |
~/.zlogin           | H |   |  |
~/.zlogout  	       | I |   |  |
/etc/zlogout 	       | J |   |  |

对于 Zsh，把自己的配置写在`~/.zshrc`中最保险

### UNIX中的环境变量

- 环境变量是用来存储有关shell回话的工作环境信息的变量
	- `env` 查看全部环境变量
	- `set` 查看当前进程中全部的环境变量
	- `unset XXX` 删除注册的环境变量XXX

- 创建环境变量
	- `export`将局部变量导出为当前进程的全局变量
	
	```
	>PATH_SELF=`pwd`
	>echo $PATH_SELF //局部变量
	>export PATH_SELF  //当前进程的全局变量
	```
- 修改环境变量
	- `$PATH`保存了所有shell命令的路径，以`:`分割
	- 向`PATH`中增加路径:`PATH=$PATH:待添加的路径`,例如:
	- 使用`source`重新加载配置文件，使配置生效
	
	```
	export PATH="$PATH:$HOME/.z/z.sh"
	//上面一句包含了三个操作：
	(1) "$PATH:$HOME/.z/z.sh" 在$PATH的追加上":$HOME/.z/z.sh"字符串
	(2) PATH = $PATH 更新PATH的值 
	(3) export PATH 将局部变量导出为当前进程的全局变量
	```
	
	
###  PS1,PS2,PS3,PS4



- PS4

PS4用来控制在`-x`的模式下，每行前面显示的内容，默认是`+`号，格式为

```
PS4='+ $BASH_SOURCE : $LINENO ' 
``` 

其中`$BASH_SOURCE`和`LINENO`为注册的环境的变量

```
#!/bin/bash -x
	
PS4='+ $BASH_SOURCE : $LINENO : '
	
TEST_VAR="test"
echo "$TEST_VAR"
	
> ./debugging-14.sh
+ PS4='+ $BASH_SOURCE : $LINENO : '
+ ./debugging-14.sh : 5 : TEST_VAR=test
+ ./debugging-14.sh : 6 : echo test
test	
```



###  DOS vs Linux File Types

- Windows的DOS和linux的shell之间可能有字符集不兼容的情况，可使用`file script.sh`命令查看字符编码:

```
> file debugging-15.sh
debugging-15.sh: Bourne-Again shell script text executable, ASCII text

```

- 或者使用插件:`dos2unix script.sh`，会自动将windows下一些CRLF去除

## Reference 

- [你可能不知道的Shell](http://coolshell.cn/articles/8619.html)
