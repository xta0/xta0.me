---
layout: post
title: Git简明操作（二）
list_title: Git简明操作 | Git Commands Quick Reference
categories: [Git,Cheatsheet]
---

### `.git`目录

`.git`目录下包含很多配置信息，我们先从`HEAD`文件入手，`HEAD`实际上是一个文本文件，我们可以用过`cat`命令查看其内容

```shell
➜  .git git:(master) cat HEAD
ref: refs/heads/master
```
`HEAD`文件中的内容表示我们当前工作在哪个分支上，我们可以用`git branch -av`命令来验证。从输出的信息上看，`HEAD`实际的指向是`.git`下的`refs/head/master`。对于`refs`目录，它包含下面几个文件夹

```shell
➜  refs git:(master) ls
heads/   remotes/ tags/
```
其中`heads`包含了所有的本地分支，`remotes`中包含了远端分支，`tags`是所有tag节点。因此`HEAD`实际上指向的是一个本地分支的名称。假如我们现在本地分支在`master`上，我们可以在查看`heads/master`中的内容

```shell
➜  heads git:(master) cat master
80ad70abdfe2b44364c9ac0b412a244a700269e7
➜  heads git:(master) git cat-file -t 80ad70abd #检查80ad70abd类型
commit
```
我们发现`master`文件中的值存放的是一个commit的哈希值，这个hash值的内容为

```shell
➜  heads git:(master) git cat-file -p master
tree 80ad70abdfe2b44364c9ac0b412a244a700269e7 #检查80ad70abd的内容
parent 02c74583828e7e8d93c741fae34b07d65426f643
author user_name <user_email> 1546623228 -0800
committer user_name <user_email> 1546623228 -0800

Update(auto commit)
```
`.git`中另一个有用的信息是`objects`文件夹，里面存放了本地提交的所有记录：

```shell
➜  objects git:(master) ls
00/   0b/   16/   21/   2c/   37/   42/   4d/   58/   
63/   6e/   79/   84/   8f/   9a/   a5/   b0/   bb/   
c6/   d1/   dc/   e7/   f2/   fd/ info/ pack/
```
如果`00-ff`均被使用，则git会对commit内容进行压缩，存放到`pack`目录下，在每个提交目录中可通过`git cat-file -p`查看提交的内容：

```
#哈希值前要加上目录值
➜  00 git:(master) git cat-file -p 00262fc2da0013eb2c913ac8b7b85e59b5378be9 
---
layout: post
title: Computer Problems
---
```
