---
updated: '2019-01-09'
layout: post
title: Git | Git中的一些重要概念
list_title: Git中的一些重要概念 | Understand Git Concept
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

HEAD也可以不和分支挂钩，当我们checkout某个commit的时候，我们可以能已经脱离了某个分支，此时Git会提醒我们处于Detach HEAD的状态，即HEAD指向了某一条commit，但是不属于某个分支。如果此时在该commit上进行了一些修改，则当我们切回某个分支时，Git会提示

```shell
If you want to keep it by creating a new branch, this may be a good time to do so with:
    git branch <new-branch-name> <commit-id>
```
这说明当前在该commit上的修改并不会被自动保留或者合并到该分支上，很可能会被Git当做垃圾处理掉。如果想要保留，需要单独建一个分支保留。因此使用Detached Head的一种场景是基于某个commit节点来拉一个新的分支，比如master上有10个commit节点，我们checkout第5个，此时HEAD指向第五个commit，并处于detached状态，此时我们执行`git branch test` 则会基于该commit拉一条新的分支`test`出来

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

### Remote Branch

当我们把一个Git仓库Clone到本地后，Git会为我们生成一个Remote Branch，格式为`<remote_name>/<branch_name>`。大多数情况下这个branch的名字为`origin/master`。注意这个`origin/master`和本地的`master`并不是同一个分支，我们可以用`git branch -av`来查看

```shell
> git branch -av
* master                   542d9b1 Update(auto commit)
  remotes/origin/HEAD      -> origin/master
  remotes/origin/master    542d9b1 Update(auto commit)
```

上面我们看到远端的`remotes/origin/master`和本地的`master`指向同一个commit，当我们产生一个本地的commit时，log会显示本地的master领先于远端的master一个commit，说明这两个分支并不会自动同步

```shell
* master                   a212c4a [ahead 1] update
  remotes/origin/HEAD      -> origin/master
  remotes/origin/master    542d9b1 Update(auto commit)
```

此时我们会有两个疑问

1. 本地的master分支是怎么创建的呢？
2. 为什么不直接在`remote/origin/master`上开发呢？

当我们在本地checkout `remote/origin/master`这个分支时，我们会发现HEAD处于了一个Detached的状态，这说明Git限制了该分支不能用来做本地开发，因此Git会为我们创建一个本地分支用来track远端的master，其它分支同理。那`remote/origin/master`什么时候更新呢？当我们执行push操作的时候本地的`master`会同步到远端的master，进而更新`remote/origin/master`

当然我们也可以不使用Git为我们创建的本地master分支，我们也可以自己分支来跟踪远端的master，具体命令为

```shell
> git checkout -b <branch_name> remote/origin/master
```

### Fetch & Pull

`git fetch`的命令会做两件事 

1. 将远端所有本地repo中没有的commits下载下来
2. 更新本地的`remotes/`分支和远端保持同步，例如远端新建了一个`bootstrap`分支，那么fetch后本地将会生成一个对应的`remotes/origin/bootstrap`的分支

但是`git fetch`并不会改变本地代码的任何状态，因此可以将fetch简单的理解为download。想要改变本地的状态，需要使用`git pull`

`git pull`可能是我们最熟悉的命令，但它实际上是包含了`fetch`的过程，当执行`git pull`时，Git会先fetch远端的commits到本地，然后再执行`merge`或者`rebase`将新的commits与本地的commits进行合并或者rebase

- merge: `git pull <remote_branch>`
- rebase: `git pull <remote_branch> --rebase`

### Push

push命令的具体格式为 `git push <remote> <place>`。一个常见的例子是`git push origin master`，其含义为去本地master拿到所有的commit，然后找到remote/origin/下的master，将缺失的commits push上去。

另外一种情况是当你处于某个分支，而却想将commit push到其它分支，这时可以使用`git push <source>:<destination>`，例如

```shell
> git push origin master^:foo
```
上述命令的含义是将master的前一个commit push到`foo`的分支上。如果`foo`不存在，则Git为其在远端创建一个`foo`分支

## Resource

- [Learn Git](https://learngitbranching.js.org/)