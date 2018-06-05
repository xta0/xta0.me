---
layout: post
list_title: C++ Part 4| STL Containers | STL容器
title: STL 关联容器
---

### 关联容器概述

- 元素是排序的，插入任何元素，都按相应的排序规则来确定其位置
- 在查找时具有很好的性能
- 通常以平衡二叉树方式实现，插入和检索的时间都是`O(log(N))`
- 常用的关联容器：
	- `set/multiset`
		- 头文件`<set>`，
		- 集合,`set`不允许有相同的元素，`multiset`中允许存在相同的元素 
	- `map/multimap` 
		- 头文件`<map>`，
		- `map`与`set`不同在于map中的元素有且皆有两个成员变量，一个名为`first`，一个名为`second`
		- `map`根据`first`值进行大小排序，并可以快速的根据first来检索。
		- `map`同`multimap`的区别在于是否允许相同的`first`值。
- 关联容器API
	- `find`：查找等于某个值的元素(x小于y和y小于x同时不成立即为相等),返回一个迭代器类型对象
	- `lower_bound`：查找某个下界
	- `upper_bound`：查找某个上界
	- `equal_range`：同时查找上界和下界

### 