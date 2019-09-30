---
updated: '2019-01-09'
layout: post
title: Git的内部工作原理
list_title: Git的内部工作原理 | Git Internal
categories: [Git]
---

### 工作区

所谓工作区就是git clone下来后，本地的repo。所谓暂存区是指当执行`git add <filename>`命令之后，文件会被暂存管理的一个区域

### `.git`目录

`.git`目录下包含很多配置信息，我们先从`HEAD`文件入手，`HEAD`是一个文本文件，我们可以用过`cat`命令查看其内容

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
我们发现`master`文件中的值存放的是一个commit的哈希值，这个hash值的内容为此次提交的记录

```shell
#检查80ad70abd的内容
➜  heads git:(master) git cat-file -p master
tree 80ad70abdfe2b44364c9ac0b412a244a700269e7 
parent 02c74583828e7e8d93c741fae34b07d65426f643
author user_name <user_email> 1546623228 -0800
committer user_name <user_email> 1546623228 -0800

Update(auto commit)
```
如果查看某次提交的具体内容，`.git`中`objects`文件夹内存放了本地提交的所有记录：

```shell
➜  objects git:(master) ls
00/   0b/   16/   21/   2c/   37/   42/   4d/   58/   
63/   6e/   79/   84/   8f/   9a/   a5/   b0/   bb/   
c6/   d1/   dc/   e7/   f2/   fd/ info/ pack/
```
如果`00-ff`均被使用，则git会对commit内容进行压缩，存放到`pack`目录下，在每个提交目录中可通过`git cat-file -p`查看提交的内容：

```shell
#哈希值前要加上目录值
➜  00 git:(master) git cat-file -p 00262fc2da0013eb2c913ac8b7b85e59b5378be9 
---
layout: post
title: Computer Problems
---
```

### Git对象之间的关系

Git中有三种对象，分别是`Commit`，`Tree`和`Blob`。每种对象均有唯一的哈希值表示，每一个commit对应一个tree结构，每个tree中又包含tree和blob，如此递归嵌套，其中tree表示文件夹，blob表示文件，三者关系可如下图所示

<img src="{{site.baseurl}}/assets/images/2011/02/git-objects.png" class="md-img-center">

举一个实际的例子，假如我们在一个git仓库中创建了一个文件夹`doc`，在该目录下新创建了一个`readme`的文件

```shell
└── doc
    └── readme.md
```
在执行`add`命令前，`.git/objects`的目录是空的，说明只创建文件git是不会将其加入到暂存区的。此时我们执行`add`操作，观察objects目录的变化

```shell
> find .git/objects -type f
.git/objects/8d/0e41234f24b6da002d962a26c2495ea16a425f
> git cat-file -t 8d0e
blob
> git cat-file -p  8d0e
hello git
```
可知objects下生成了一个blob, 也就是我们的readme文件，内容为`hello git`。然后我们进行commit操作，objects目录下会生成四个对象，

```shell
> find .git/objects -type f
.git/objects/da/e6876e15d84cdec59319806e39272c5d58e7c9
.git/objects/88/d0860df2c42bd9d9c2ed893eb43460a1871bbe
.git/objects/5d/6d9f8679f21ea8a190a9222aa8f4d3c38e6344
.git/objects/8d/0e41234f24b6da002d962a26c2495ea16a425f
```
我们逐个观察其类型

```shell
> git cat-file -t dae6
commit
> git cat-file -t 88d0
tree
> git cat-file -t 5d6d
tree
> git cat-file -t 8d0e
blob
```
他们之间的关系如下图所示

<img src="{{site.baseurl}}/assets/images/2011/02/git-objects-2.png" class="md-img-center">

### HEAD

前面可知Head指向的是当前工作区的分支的最新commit，但是也可以指向仓库中的任意一个commit

- Detach Head

当我们checkout某个commit的时候，我们可以能已经脱离了某个分支，此时Git会提醒我们处于Detach HEAD的状态。如果此时在该commit上进行了一些修改，则当我们切回某个分支时，Git会提示

```shell
If you want to keep it by creating a new branch, this may be a good time to do so with:
    git branch <new-branch-name> <commit-id>
```
这说明当前在该commit上的修改并不会被自动保留或者合并到当前分支上，很可能会被Git当做垃圾处理掉。如果想要保留，需要单独建一个分支保留

- HEAD 

由于HEAD可以指代commit，对所有commit相关的git命令，均可以用HEAD进行替换，比如

```shell
# git diff <commit-1> <commit-2>
git diff HEAD HEAD~1 
#比对HEAD指向的commit和HEAD前一条commit之间的差别
```

## Resource

- [玩转Git](https://git201901.github.io/github_pages_learning/docs/%E8%8B%8F%E7%8E%B2%E3%80%8A%E7%8E%A9%E8%BD%ACGit%E4%B8%89%E5%89%91%E5%AE%A2%E3%80%8B-%E6%9E%81%E5%AE%A2%E6%97%B6%E9%97%B4.pdf)