---
layout: post
title: Git | Git简明操作
list_title: Git简明操作 | Git Quick Reference
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

- Add
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

### Delete

- 删除工作区的未追踪文件: `git clean -fxd`
- 删除暂存区中的文件: `git cache rm <file>`
- 删除Git缓存: `git gc`

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

- 修改本地当前的commit：`git commit --amend`
- 修改本地当前的commit的提交信息: `git commit --amend --author="name <email>"`
- revert本地的commit: `git reset --hard`
- 修改之前commit的内容
    - `git rebase -i <commit>^`
    - 标记要修改的commit为edit
    - 修改完成后执行`git add`修改添加到暂存区
    - `git commit --amend`
    - `git rebase --continue`
- 合并commit
    - `git rebase -i `
    - pick一个commit作为最终合并后的commit，其它的commit会合并到它上面

### Cherry-pick

- `git cherry-pick <commit1> <commit2>,...`

### tag

- 标记某个commit为tag： `git tag <tag_name> <commit>`
- 查看某个commit和最近tag的关系 `git describe <commit>`，结果的格式为
    - `<tag>_<numCommits>_g<hash>`表示当前commit - `g<hash>`距离`<tag>`有`<numCommits>`个commit

### Repo

- Check remote repo: `git remote -v `
- Change repo's origin `git remote set-url origin https://xxxx.git`
- Add remote origin
    - `git remote add upstream xxx`
    - `git fetch upstream`
    - `git merge upstream/master`

### 其它

- `gitk`， 图形化界面