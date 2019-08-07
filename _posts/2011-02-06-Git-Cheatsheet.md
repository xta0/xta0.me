---
layout: post
title: Git简明操作
list_title: Git简明操作 | Git Commands Quick Reference
categories: [Git,Cheatsheet]
---

### Config

- Global Configuration
    - `git config --global user.name "your_name"`
    - `git config --global user.email "your_email@domain.com"`

- Configuration for project
    - `git config --local user.name "your_name"`
    - `git config --local user.email  "your_email@domain.com"`
    - 该命令实际上是修改`.git`下的config文件中的`[user]`字段

- Show Configuration
    - `git config --list --local`
    - `git config --list --global`


### Log

- 查看前5条log：`git long -n5 --graph`

### diff

- 查看commit之间的差异 `git diff <commit_1> <commit_2>`
- 查看某次commit和HEAD之间的差异 `git diff HEAD <commit_2>`
- 查看工作区和暂存区的差异 `git diff -- <filename1> <filename2>`
- 查看暂存区和HEAD之间的差异 `git diff --cached`
- 查看不同分支之间某个文件的差异 `git diff <branch1> <branch2> -- <filename>`

### File Operations

-Add
    - `git add -u`, 将工作空间新增和被修改的文件添加的暂存区
    - `git add .`, 将工作空间被修改和被删除的文件添加到暂存区(不包含没有纳入Git管理的新增文件)
    - `git add -A`, stash所有修改
- Remove: 
    - `git rm file_name`
    - 同时删除工作区和暂存区的文件
- Rename: `git mv file_name_1 file_name_2`, 重命名文件

### Revert 

- revert更工作区的提交: `git checkout -- <filename>`
- revert暂存区的提交: `git reset HEAD -- <filename>` 
- revert到某个commit `git reset --hard <commit_id>`

### Stash

- 将工作区变更存放到Stash区域: `git stash`
- 查看Stash内容：`git stash list`
- 取回Stash中的变更
    - `git stash apply`
    - `git stash pop` 会丢掉stash区域里的信息

### Branch

- 查看分支
    - 查看本地分支,`git branch -av`
- 创建新分支
    - 创建本地分支, `git branch <branchName>`
    - 创建远端分支, `git push origin <branchName>`
- 切换分支
    - `git checkout <branchName>`
    - 创建新分支并且换:`git checkout -b <branchName>`
- 删除分支
    - 删除本地分支：`git branch -d <branchName>`
    - 删除远程分支：`git push origin --delete <branchName>`

### Commit

- 修改本地当前的commit的message：`git commit --amend`
- 修改本地当前的commit中的个人信息: `git commit --amend --author="name <email>"`
- 修改本地之前的commit的message：`git rebase -i` 后选择`r`
- 合并commit
    - `git rebase -i `
    - pick一个commit作为最终合并后的commit，其它的commit会合并到它上面
- revert当前本地的commit: `git reset --hard`
- 参数
    - `-m`, commit信息
    
### Repo

- Check remote repo: `git remote -v `
- Change repo's origin `git remote set-url origin https://xxxx.git`
- Add remote origin
    - `git remote add upstream xxx`
    - `git fetch upstream`
    - `git merge upstream/master`

### 解决合并冲突

- `grep -lr '<<<<<<<' . | xargs git checkout --ours`
- `grep -lr '<<<<<<<' . | xargs git checkout --theirs`


### 其它

- `gitk`， 图形化界面