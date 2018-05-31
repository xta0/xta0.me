---
list_title: Package Manager Part 2
layout: post
---

## Dotfile

> "工欲善其事必先利其器"

上一篇整理常用的包管理系统，但是如果每次配环境都要重复一次实在是痛苦，而且除了包管理之外其他的配置更是繁琐复杂。于是有人就想到了整理系统所有的配置文件到云端，这样下次装新机器或者配环境的时候就可以自动还原系统的所有配置，于是就有了所谓的Dotfile。

我个人使用Dotfile主要有两个原因：一是用于同时工作于三台机器上，希望每台机器的环境都能和其它两台及时同；二是一些个人偏好的配置，程序员都有一些强迫症，自己看着舒服的界面工作效率才高。因此有必要将通用配置抽取出来放到云端，如果有配置更新就同步云端，然后更新到其它机器上。


### Dotfile结构

Dotfile的结构因人而异，没有什么固定的套路，本质上就是一些配置文件按目录存放，clone下来后，自己可以解析并还原配置就可以了。也有人收集了很多其它人的[Dotfiles](https://dotfiles.github.io/)供人参考，如果想搞自己的Dotfile可以去这里挖一挖，里面经常有一些好用的脚本或者一些不错的配置。我也个人的Dotfile结构如下：

```
➜  .dotfiles git:(master) tree -L 1
.
├── LICENSE
├── README.md
├── bin
├── dotfiles
├── etc
├── install
├── install.sh
├── packages
├── system
└── uninstall.sh
```





<img src="/assets/images/2014/05/dotfile-1.png"  width="60%" />


### Resoures

- [Dotfiles](https://dotfiles.github.io/)