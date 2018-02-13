---
layout: post
title: C++ STL Part 2
categories: PL
tag: C++
mathml: true
---

## STL算法

- STL中的算法大致可以分为以下七类:
    - 不变序列算法
    - 变值算法
    - 删除算法
    - 变序算法
    - 排序算法
    - 有序区间算法
    - 数值算法

- 大多重载的算法都是有两个版本的
    1. 用 `==` 判断元素是否相等, 或用 `<` 来比较大小
    2. 多出一个类型参数 `Pred` 和函数形参 `Pred op`: 
        - 通过表达式 `op(x,y)` 的返回值: `ture/false`判断`x`是否 “等于” `y`，或者x是否 “小于” y

    ```cpp
    iterator min_element(iterator first, iterator last);
    iterator min_element(iterator first, iterator last, Pred op);
    ```

###  不变序列算法

- 该类算法不会修改算法所作用的容器或对象
- 适用于顺序容器和关联容器
- 时间复杂度都是O(n)


算法  | 功能
------------- | -------------
find  | 求两个对象中较小的(可自定义比较器)
min | 求两个对象中较小的(可自定义比较器)
max | 求两个对象中较大的(可自定义比较器)
min_element | 求区间中的最小值(可自定义比较器)
max_element | 求区间中的最大值(可自定义比较器)
for_each | 对区间中的每个元素都做某种操作
count | 计算区间中等于某值的元素个数
count_if |  计算区间中符合某种条件的元素个数
find | 在区间中查找等于某值的元素
find_if | 在区间中查找符合某条件的元素
find_end | 在区间中查找另一个区间最后一次出现的位置(可自定义比较器)
find_first_of | 在区间中查找第一个出现在另一个区间中的元素 (可自定义比较器)
adjacent_find | 在区间中寻找第一次出现连续两个相等元素的位置(可自定义比较器)
search | 在区间中查找另一个区间第一次出现的位置(可自定义比较器)
search_n | 在区间中查找第一次出现等于某值的连续n个元素(可自定义比较器)
equal | 判断两区间是否相等(可自定义比较器)
mismatch | 逐个比较两个区间的元素，返回第一次发生不相等的两个元素的位置(可自定义比较器)
lexicographical_compare | 按字典序比较两个区间的大小(可自定义比较器)

- **find**

```cpp
template<class InIt, class T>
InIt find(InIt first, InIt last, const T& val);
```
返回区间 `[first,last)` 中的迭代器 `i` ,使得 `* i == val`

- **find_if**:

```cpp
template<class InIt, class Pred>
InIt find_if(InIt first, InIt last, Pred pr);
```
返回区间 `[first,last)` 中的迭代器 `i`, 使得 `pr(*i) == true`