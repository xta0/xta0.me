---
layout: post
title: Git简明操作（一）
list_title: Git简明操作（一） | Git Commands Quick Reference
categories: [Git,Cheatsheet]
---

### Config

- Global Configuration
    
    ```shell
    git config --global user.name "your_name"
    git config --global user.email "your_email@domain.com"
    ```

- Configuration for project

    ```shell
    git config --local user.name "your_name"
    git config --local user.email  "your_email@domain.com"
    ```

    该命令实际上是修改`.git`下的config文件中的`[user]`字段

- Show Configuration

    ```shell
    git config --list --local
    git config --list --global
    ```

### Add/Remove Files


- Remove
    - `git rm file_name`
- Rename
    - `git mv file_name_1 file_name_2`, 重命名文件

### Stash

- Add
    - `git add -u`, 将工作空间新增和被修改的文件添加的暂存区
    - `git add .`, 将工作空间被修改和被删除的文件添加到暂存区(不包含没有纳入Git管理的新增文件)
    - `git add -A`, stash所有修改

- 从stash文件中删除一个
```
git reset path_to_the_file
```

### Commit



- 参数
    - `-m`, commit信息

- 恢复到上一次commit的状态
    ```
    git checkout -- .
    ```
- 修改commit中的个人信息

    ```
    git commit --amend --author="Author Name <email@address.com>"`
    ```

    对于某次Commit，其auther和committer可能不同，

    
### Repo

- Check remote repo 

    ```
    git remote -v 
    ```
- Change repo origin 

    ```
    git remote set-url origin https://xxxx.git
    ```

### Branch

- 查看分支
    - 查看本地分支,`git branch -av`

- 创建新分支
    - 创建本地分支, `git branch <branchName>`
    - 创建远端分支,`git push origin <branchName>`
- 拉取分支

    ```shell
    git fetch
    git checkout -b <branchName>
    ```

- 删除分支

```
//删除本地分支
git branch -d <branchName>
//删除远程分支
git push origin --delete <branchName>
```

- 合并

```
git merge branch_to_merge
```

- 解决合并冲突

- `grep -lr '<<<<<<<' . | xargs git checkout --ours`
- `grep -lr '<<<<<<<' . | xargs git checkout --theirs`

How this works: `grep` will search through every file in the current directory (the `.`) and subdirectories recursively (the `-r` flag) looking for conflict markers (the string '<<<<<<<')

the `-l` or `--files-with-matches` flag causes grep to output only the filename where the string was found. Scanning stops after first match, so each matched file is only output once.

The matched file names are then piped to `xargs`, a utility that breaks up the piped input stream into individual arguments for `git checkout --ours` or `--theirs`


### 其它

- `gitk`， 图形化界面