---
updated: '2019-01-09'
layout: post
title: 理解Git (一)
list_title: 理解Git (一) | Understand Git Concept Part 1
categories: [Git]
---

### 工作区

所谓工作区就是git clone下来后，本地的repo。所谓暂存区是指当执行`git add <filename>`命令之后，文件会被暂存管理的一个区域(staging area)。当文件进入暂存区后，它将有资格被Commit

### Commit

每一个条Git的Commit都可以用一串md5的hash值表示，都有自己的parent，即指向当前分支的前一条commit，因此所有的commit会组成一棵树。

### HEAD

HEAD用来指向当前工作区分支的最新commit，但是也可以指向仓库中的任意一个commit。可以认为HEAD就是某个commit的指针，对所有commit相关的git命令，均可以用HEAD进行替换，比如

```shell
# git diff <commit-1> <commit-2>
> git diff HEAD HEAD~1 
#比对HEAD指向的commit和HEAD前一条commit之间的差别
```

- HEAD与分支

假设我们现在在某个repo的master上，由于HEAD默认指代当前分支的最新commit，我们可以用下面命令查看HEAD

```shell
> git show HEAD
commit dae6876e15d84cdec59319806e39272c5d58e7c9 (HEAD -> master)
```
可以看到它指向上面一条commit，这时候我们创建一个branch并切到最新的branch上，观察HEAD的变化

```shell
> git branch b1
> git checkout b1
> git show HEAD
commit dae6876e15d84cdec59319806e39272c5d58e7c9 (HEAD -> master, b1)
```
可以看到它依旧指向同一个commit，这是因为`b1`是基于master创建。接着我们修改Readme文件

```shell
> echo blahblah >> readme.md
> git add .
> git commit -m "update readme.md"
> git show HEAD 
commit 6701c21717b895bbe7cb745fa5a7ac37d32490f9 (HEAD -> b1)
```
由于我们生成了一个新的commit，可以看到此时HEAD指向了它，即当前分支最新的commit。

- Detach HEAD

HEAD也可以不和分支挂钩，当我们checkout某个commit的时候，我们可以能已经脱离了某个分支，此时Git会提醒我们处于Detach HEAD的状态。如果此时在该commit上进行了一些修改，则当我们切回某个分支时，Git会提示

```shell
If you want to keep it by creating a new branch, this may be a good time to do so with:
    git branch <new-branch-name> <commit-id>
```
这说明当前在该commit上的修改并不会被自动保留或者合并到当前分支上，很可能会被Git当做垃圾处理掉。如果想要保留，需要单独建一个分支保留

### Relative Refs

Git中的commit的值用一串很长hash字符串表示，不好记，因此Git中还提供几种使用Relative commit的方式，包括下面几种

- `^`表示当前commit的前若一条commit

```shell
> git show HEAD^    #HEAD的前一条commit
> git show master^  #master分支当前commit的前一条commit
> git show bugFix^^ #bugfix分支当前commit的前两条commit
```

- `~<num>`表示当前commit的前`num`条commit

```shell
> git checkout HEAD~2 #checkout HEAD之前两条的commit，此时
```
此时HEAD会处于detach的状态并指向某条commit，我们可以让分支指针强行指向HEAD（谨慎操作，有风险）

```shell
#假设HEAD指向master
> git checkout HEAD~5 #移动HEAD到master后5个commit
> git branch -f master HEAD #将master强行移到HEAD，则master前5个commit会丢失
```

### Merge

理论上每个Commit都只有一个Parent，但实际上，有些情况某个Commit会有两个parent，比如当我们合并分支的时候，Git会自动生成一个指向两个parent的commit，如下图所示

<img src="{{site.baseurl}}/assets/images/2011/02/git-commits-merge.png" class="md-img-center">

上图中我们执行了merge命令，将`bugFix`分支合并到`master`上

```shell
> git merge bugFix
```
我们看到Git为我们新生成了一个commit - `c4`。这时候我们查看`git log`，可以发现master已经包含了`bugFix`分支的提交记录，说明master已经拥有了包括`bugFix`在内的全部的commit。此时我们先记录下HEAD

```shell
commit 70726c576c87a55c333c7c6050c5f37a574d3e1c (HEAD -> master)
```
接下来我们将分支切到`bugFix`，并执行`git log`，发现`bugFix`分支并没有master的信息，于是我们可以执行

```shell
> git merge master
```

此时由于master分支已经包含了包括`bugFix`在内的所有commit，因此Git只需要将HEAD指针指向master即可，如下图所示

<img src="{{site.baseurl}}/assets/images/2011/02/git-commits-merge-2.png" class="md-img-center">

此时`bugFix`分支也包含了`master`的所有信息，我们可以通过`git show HEAD`来验证

```shell
> git show HEAD
commit 70726c576c87a55c333c7c6050c5f37a574d3e1c (HEAD -> bugFix, master)
```
使用Merge方式的一个问题在于对于多人合作的项目会产生多个无效的Merge节点，阅读体验不是很友好

<img src="{{site.baseurl}}/assets/images/2011/02/git-commits-merge-3.png" class="md-img-center">

比如上图中当有两个人同时commit代码时，log会产生出多个分支和合并节点，试想当有10个人同时协作时，将会有10条平行线，对于问题排查或者追溯历史非常不友好

### Rebase

虽然Git默认的模式是Merge，但是也支持使用Rebase模式。所谓Rebase是指当有新的改动时，我们为其生成一个新的commit，通过改变commit在目标分支中的位置，从而将其纳入到目标分支中。Rebase的优点在于所有commit是线性排列的，log上看没有多条分支合并的情况；同时Rebase也不会产生Merge模式的commit节点，即每个commit都只有一个或者零个parent

还是上面的例子，加入我们现在在bugFix上生成一个commit `c3`，现在我们想让它rebase到master上，我们可以执行

```shell
> git rebase master
```
如下图所示

<img src="{{site.baseurl}}/assets/images/2011/02/git-commits-rebase.png" class="md-img-center">

我们看到c3被rebase到了master，但是这个c3并不是当前bugFix分支上的c3，而是它的一个copy。

而此时bugFix也变成了一个线性的，里面含有master的commits。如果master上的代码和bugFix有冲突，则此时需要解决合并的冲突。

此时我们将分支切回master，并执行

```shell
> git rebase bugFix
```

如下图所示，来自bugFix分支的c3成了当前master的最新节点，被rebase到了最前面

<img src="{{site.baseurl}}/assets/images/2011/02/git-commits-rebase-2.png" class="md-img-center">

### Remote

当我们把一个Git仓库Clone到本地后，Git会为我们生成一个Remote Branch，格式为`<remote_name>/<branch_name>`。大多数情况下这个branch的名字为`origin/master`


## Resource

- [Learn Git](https://learngitbranching.js.org/)
- [玩转Git](https://git201901.github.io/github_pages_learning/docs/%E8%8B%8F%E7%8E%B2%E3%80%8A%E7%8E%A9%E8%BD%ACGit%E4%B8%89%E5%89%91%E5%AE%A2%E3%80%8B-%E6%9E%81%E5%AE%A2%E6%97%B6%E9%97%B4.pdf)