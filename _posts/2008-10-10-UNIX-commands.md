---
layout: post
updated: "2018-08-03"
list_title: UNIX 常用命令 | UNIX Shell Commands
title: UNIX 常用命令
categories: [UNIX]
---

## 用户管理

- `su` 切换用户
	- `su - USERNAME` 使用 login shell 方式切换用户
- `sudo` 以其他用户身份执行命令
	- `visudo` 

## 文件操作

- `echo`：输出字符
	- `echo hello world > ~/text`: `>`为输出流标志，输出文本到text中
- `man`: 命令手册
- `ls` 命令:	
	- `ls -l`：将结果按照长列表形式显示
	- `ls -a`: 查看隐藏文件
	- `ls -r`: 将结果reverse排列
	- `ls -S`: 将结果按文件size进行排序
	- `ls -t`: 将结果按时间进行排序
	- `ls -R`: 将结果进行递归显示
- `touch`:创建空文件
- `cat/more/less`：查看文件
	- `cat`：查看短文件
	- `more/less`：查看长文件
	- `less`：
- `tail`
	- `tail -n 1 log.error`
- `cp`：copy文件
- `mv`：移动文件,重命名
- `rm`：删除文件
	- `rm`：删除文件
	- `rm -d`：删除空文件夹
	- `rm -rd`：强制递归删除文件夹中的文件
	- `rm -rf`
	- `rm -rf test1/test2 -v`：查看删除过程

### 统计文件个数

- 统计当前目录下文件的个数（不包括字目录： ` ls -l  | grep "^-" | wc -l`
- 统计当前目录下文件的个数（包括子目录）： ` ls -lR | grep "^-" | wc -l`
- 统计当前目录下文件夹个数（包括子目录）： ` ls -lR | grep "^d" | wc -l`

### 文件权限

以下面这两条记录为例：

```shell
drwxr-xr-x  6 root root 4096 Aug  8 15:58 nginx
-rw-r--r--  1 root root 2078 Aug  8 14:29 nginx.conf
```

1. `d`代表文件夹
2. `-`代表文件

每条权限有三个section，每个section有三个字母`rwx`表示，分别表示read，write，exe的权限，如果某个secion不足三个字母，缺失部分用`-`代替。以上面第二条为例分析

```shell
owner | group | everyone
rw-     r--      r--
```

1. Owner可以读写该文件，但不能执行
2. group组的用户和一般用户只能read

当创建一个新文件，通常要为其赋予权限码，其规则为,`r`代表`4`，`w`代表`2`，`x`代表1，空位为`0`，每个section的值为`r+w+x`，如下

```shell
Octal Permissions
---------------------------
r = 4 	w = 2 	x = 1
```

还是上面的例子，按照上述规则计算，则可推出，`nginx.conf`的`chmod`值为`644`

```shell
owner | group | everyone
rw-   | r--   |  r--
4+2+0 | 4+0+0 | 4+0+0
6       4        4
```

常用的几个权限码为

```shell
-rw-r--r--    644
-rwxrwxrwx    777
drwxr-xr-x    755
-rw-------    600
```

### 文件压缩

- **使用zip**

```shell
#To compress
zip -r archive_name.zip folder_to_compress
#avoid "_MACOSX” or “._Filename” and ".ds"
zip -r -X archive_name.zip folder_to_compress
# To extract
unzip archive_name.zip
```

- **使用tar**

```shell
#To compress
tar -zcvf archive_name.tar.gz folder_to_compress
#To extract
tar -zxvf archive_name.tar.gz
```

### 加密

- 查看RSA的FingerPrint

```shell
ssh-keygen -l -f rsa_key.pub
```

## 网络操作

- 查看端口进程

```shell
lsof -wni tcp:3000 #查看3000端口进程
kill -9 PID #杀掉进程
```

- 查看域名IP地址

```shell
host example.com
host -t a example.com
```

- **CURL post request**

For sending data with POST and PUT requests, these are common `curl` options:
- request type
	- `-X POST`
	- `-X PUT`
- content type header
	- `-H "Content-Type: application/x-www-form-urlencoded"`
	- `-H "Content-Type: application/json"`
- data
	- form urlencoded: `-d "param1=value1&param2=value2"` or `-d @data.txt`
	- json: `-d '{"key1":"value1", "key2":"value2"}'` or `-d @data.json`


## Resouces

- [Linux/Mac OS下文件夹的含义](http://en.wikipedia.org/wiki/Filesystem_Hierarchy_Standard)