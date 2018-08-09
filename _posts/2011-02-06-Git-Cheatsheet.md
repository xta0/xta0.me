---
layout: post
title: Git简明操作
list_title: Git简明操作 | Git Commands Quick Reference
---

### Config

- Global Configuration
    ```
    git config --global user.name = ""
    git config --global user.emal = ""
    ```

- Configuration for project
    ```
    git config user.name ""
    git config user.email ""
    ```

### Stash

- stash所有修改
```
git add -A
```
- 从stash文件中删除一个
```
git reset path_to_the_file
```

### Commit

- 恢复到上一次commit的状态
    ```
    git checkout -- .
    ```
- 修改commit中的个人信息
    ```
    git commit --amend --author="Author Name <email@address.com>"`
    ```
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

- 创建新分支

```
//本地创建新分支
git branch some_branch
//push
git push origin some_branch
```

- 切分支

```
git branch branch_name
```

- 删除分支

```
git branch -d branch_name
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

