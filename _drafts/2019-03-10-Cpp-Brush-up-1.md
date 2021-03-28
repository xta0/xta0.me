---
list_title: C++拾遗 
title: C++ Brush up Part 1
layout: post
categories: ["C++"]
---

### `noexcept`

当 push_back、insert、reserve、resize 等函数导致内存重分配时，或当 insert、erase 导致元素位置移动时，vector 会试图把元素move到新的内存区域。vector通常保证强异常安全性，如果元素类型没有提供一个`noexcept`的移动构造函数，**vector 通常会使用拷贝构造函数**。因此，对于拷贝代价较高的自定义元素类型，我们应当定义移动构造函数，并标其为 `noexcept`，或只在容器中放置对象的智能指针。



