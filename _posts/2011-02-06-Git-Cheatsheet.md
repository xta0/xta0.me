---
layout: default
title: Git简明操作
list_title: Git简明操作 | Git Commands
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

### Conflict

- `grep -lr '<<<<<<<' . | xargs git checkout --ours`
- `grep -lr '<<<<<<<' . | xargs git checkout --theirs`

How this works: `grep` will search through every file in the current directory (the `.`) and subdirectories recursively (the `-r` flag) looking for conflict markers (the string '<<<<<<<')

the `-l` or `--files-with-matches` flag causes grep to output only the filename where the string was found. Scanning stops after first match, so each matched file is only output once.

The matched file names are then piped to `xargs`, a utility that breaks up the piped input stream into individual arguments for `git checkout --ours` or `--theirs`

