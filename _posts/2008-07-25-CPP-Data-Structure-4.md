---
layout: post
title: Data Structure Part 4
---

## 字符串

- 字符串，特殊的<mark>线性表</mark>，即元素为字符的线性表
- `n(≥0)` 个字符的有限序列`n≥1`时，一般记作`S: "c0c1c2...cn-1" `
    – S 是串名字
    – `"c0c1c2...cn-1"`是串值
    – `ci`是串中的字符
    – `N` 是串长（串的长度）：一个字符串所包含的字符个数
        -  空串：长度为零的串，它不包含任何字符内容
- String类
    - 内部封装了C字符串API

## 模式匹配

- 模式匹配（pattern matching）
    - 目标对象T（字符串）
    - 模式P（字符串）

- 给定模式P，在目标字符串T中搜索与P模式匹配的子串，并返回第一个匹配串首字符的位置

```
T t(0) t(1) ... t(i) t(i+1)... t(i+m-2) t(i+m-1) ... t(n-1)
                 ||  ||            ||     ||
P               p(0) p(1) ...... p(m-2)  p(m-1)
```
为使模式`P`与目标`T`匹配，必须满足: `p(0)p(1)p(2)...p(m-1) = t(i)t(i+1)t(i+2)...t(i+m-1)`

### 朴素算法

- 以`T`的每个字符为起点，先后遍历，看是否与`P`匹配

```cpp
int FintPat(string S, string P, int startIndex){
    int lastIndex = S.length() - P.length();
    if(startIndex < lastIndex){
        return (-1);
    }
    for(int g = startIndex; g<=LastIndex; g++){
        if(P == S.substr(g,P.length())){
            return g;
        }
    }
    return -1;
}
```
- 假定目标`T`的长度为`n`，模式`P`，长度为`m(m<=n)`
- 最坏的情况
    - 每一次循环都不成功，则一共要进行比较`(n-m+1)`次
    – 每一次“相同匹配”比较所耗费的时间，是`P`和`T`逐个字符比较的时间，最坏情况下共`m`次
    – 整个算法的最坏时间开销估计为`O(m*n)`
- 最好情况
    - 在目标的前`m`个位置上找到模式
        - 总比较次数：`m`
        - 时间复杂度：`O(m)`

### KMP算法

- 无回溯遍历
- 上面朴素法匹配的过程中，如果出现：
    - `P`的前`j-1`位和`T`的`(i-j+1)`位相等，即`P.substr(1,j-1) == T.substr(i-j+1,j-1)`
    - 但是最后一位不相等，即`P(j) != T(i)`
- 此时该用`P`中哪个字符继续和`T(i)`比较呢？
    - 
- 





