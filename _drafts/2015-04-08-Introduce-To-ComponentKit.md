---
layout: post
title: Introduce to ComponentKit
tag: iOS

---

##ComponetKit

### Overview

ComponentKit(下文简称CK)最初被设计用来优化Facebook App的News Feed模块，整理开源后适用于优化List/Collection的场景，它有如下几个feature：

- **Declarative**: 布局无需复杂的数学计算，指定view间的相对位置即可
- **Functional**: 当状态变化时，重新生成新的数据结构，而不是更改原来的状态。
- **Performance**: View的Layout在后台线程中执行

Composable: Here FooterComponent is used in an article, but it could be reused for other UI with a similar footer. Reusing it is a one-liner. CKStackLayoutComponent is inspired by the flexbox model of the web and can easily be used to implement many layouts.

> - Scroll Performance: All layout is performed on a background thread, ensuring the main thread isn't tied up measuring text. 60FPS is a breeze even for deep, complex layouts like Facebook's News Feed.

CK的layout是异步执行的，部分UIView的渲染（例如text）使用了AsyncDisplayKit，是异步绘制的。

- 但是它目前也有一些局限性：

> - Interfaces that aren't lists or tables aren't ideally suited to ComponentKit since it is optimized to work well with a UICollectionView.

目前只适用于list/collection的场景

### 初识CompnentKit





这两条正好可以用来帮我优化VZListViewController :)

### SourceCode

看了源码和关键类的API设计，非常困惑，API命名看不懂，使用C++的语法除了在构造函数上支持了所谓的"Declarative"，在其它地方简直是灾难，Debug起来非常麻烦。

### Demo

