---
layout: post
title: 理解CoreAnimation和GPU
tag: iOS
categories: iOS

---

<em>所有文章均为作者原创，转载请注明出处</em> 


## 理解CoreAnimation

<em>更新于2015年8月</em>

之前写了一篇关于UIView渲染的文章，很浅显，很多关于CoreAnimation的细节还不够清楚，这几年为了理解这个问题，又陆陆续续收集了一些资料，但一直处于零敲碎打的阶段，很难把这些零碎的知识点拼到一起。直到今年研究AsyncDisplayKit时又好好看了一遍WWDC关于CoreAnimation的session，理解又加深了一些。

### 跨进程

很重要的一点是，CoreAnimation是夸进程的，无论是Implict Transition还是动画，

### 相关的资料

- WWDC:
	- 2011-session 318:
	- 2012-session
	- 2014-session 419
	- 2015-session

 