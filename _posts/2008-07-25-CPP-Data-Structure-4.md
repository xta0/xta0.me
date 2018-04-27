---
layout: post
title: Data Structure Part 4
mathjax: true
---

## 字符串

- 字符串，特殊的<mark>线性表</mark>，即元素为字符的线性表
- `n(≥0)` 个字符的有限序列`n≥1`时，一般记作`S: "c0c1c2...cn-1" `
    – S 是串名字
    – `"c0c1c2...cn-1"`是串值
    – `ci`是串中的字符
    – `N` 是串长（串的长度）：一个字符串所包含的字符个数
        -  空串：长度为零的串，它不包含任何字符内容
- 子串
    - 假设`s1,s2` 是两个串：`s1 = a0a1a2…an-1`,`s2 = b0b1b2…bm-1`,其中 `0 ≤ m ≤ n`, 若存整数`i (0 ≤ i ≤n-m)`，使得 `b(j) = a(i+j), j =0,1,…,m-1`同时成立，则称串`s2`是串`s1`的子串,`s1`为串`s2`的主串，或称`s1`包含串`s2`
    - 特殊子串
        – 空串是任意串的子串
        – 任意串`S`都是`S`本身的子串
        – 真子串：非空且不为自身的子串
- String类
    - 内部封装了C字符串API

### 字符串的特征向量

设模式`p`由`m`个字符组成，记为 `p=p(0)p(1)...p(m-1)`，令特征向量`N`用来表示模式`P`的字符分布特征，简称`N`向量由`m`个特征数`n(0)...n(m-1)`整数组成，记为 `N=n(0)n(1)...n(m-1)`,`N`也称为next数组，每个`n(j)`对应next数组中的元素`next[j]`

- 字符串的特征向量的构造方法

设`P`第`j`个位置的特征数`n(j)`，首尾串最长的`k`。 则首串为： `p(0)p(1)...p(k-2)p(k-1)`；尾串为： `p(j-k)p(j-k+1)...p(j-2)p(j-1)`。特征向量`next[j]`为

$$
next[j]=
\begin{cases}\
-1, & & j=0\\
max \{ P[0..k-1]=P[j-k..j-1] \} & & k:0<k<j\\
0, & & other \\
\end{cases}
$$

即`next[j]`为长度为`j`的子串的<mark>前(j-1)个字符中最长的首</mark>尾真子串长度`k`。假设模式串`P`为`aaaabaaaac`，则特征向量`N(next[j])`的值为：

```
P = a  a  a  a  b  a  a  a  a  c
N = -1 0  1  2  x  0  1  2  3  4
              (x=3)
```

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
- 此时该用`P`中哪个字符(`P(k)`)继续和`T(i)`比较呢？
- Knuth-Morrit-Pratt(KMP)
    - `k`值仅仅依赖于模式`P`本身，与目标对象`T`无关

- 算法思想

```
T t(0) t(1) ... t(i) t(i+1) ... t(i+j-2) t(i+j-1) t(i+j)... t(n-1)
                 |   |          |        |        X
P               p(0) p(1)   ... p(j-2)   p(j-1)   p(j)
```
假如有目标`T`和模式`P`，他们具有上面的，可知`t(i)t(i+1)...t(i+j-1) = p(0)p(1)...p(j-1)`，但是`p(j) != t(i+j)`，如果按照上面朴素算法，下一步比较应该是将`p(0)`向右移动一位，使其指向`t(i+1)`，即：

```
T t(0) t(1) ... t(i) t(i+1) ... t(i+j-2) t(i+j-1) t(i+j)... t(n-1)
                |    |          |        |        X
P               p(0) p(1)   ... p(j-2)   p(j-1)   p(j)
朴素匹配下一步          p(0) p(1) ... p(j-3)  p(j-2) p(j-1) p(j)
```
显然这种情况下，`p(0)`到`p(j-1)`的比较是多余的。类似的，对于朴素的匹配算法有两个冗余的问题：

- 每次目标串`T`需要回溯
- 每次模式串`P`需要重头遍历

解决第一个问题需要在`T`不移动的前提下，模式串`P`尽可能的右移。那么该右移多少位呢？先看一个具体例子，假设有串`P`和目标串`T`如下，其中`N`为模式串`P`的特征向量，如下：

```
P =  a b a b a b b
N = -1 0 0 1 2 3 4

T = a b a b a b a b a b a b a b b
    | | | | | | x
P = a b a b a b b (i=6,j=6,N[j]=4)
```
可以看到当比较到`i=6, j=6`时，`T`和`P`不相等，这时候`P`该向右移动多少位呢？此时`P[6]`的特征向量，`N[6]=4`，说明首串最大长度为`4`，因此可以将`P`右移`j-k`位,即`6-4=2`位:

```
T = a b a b a b a b a b a b a b b
        | | | | | | x
P =     a b a b a b b (i=8,j=6,N[j]=4)
```
重复上面步骤，计算`j-k`，即`6-4=2`位，继续移动:

```
T = a b a b a b a b a b a b a b b
            | | | | | | x
P =         a b a b a b b (i=10,j=6,N[j]=4)
```

不断重复上述步骤，直到:

```
T = a b a b a b a b a b a b a b b
                    | | | | | | |
P =                 a b a b a b b 
```

总结一下，如果`P`首尾的真子串不相等，即：`p(0)p(1)...p(j-2) != p(1)p(2)...p(j-1)`，则能得出朴素匹配的下一步一定是不匹配的。同样的，如果`p(0)p(1)...p(j-3) != p(2)(p3)...p(j-1)`，则再下一趟也一定不匹配。直到对于某个`k`值（首尾串的长度），使得第`k`个首尾子串不等：`p(0)p(1)...p(k) != p(j-k-1)p(j-k)...p(j-1)` 但是第`k-1`个首尾子串相等`p(0)p(1)...p(k-1) = p(j-k)p(j-k)...p(j-1)` 即

```
t(i+k-1) t(i+k) ...   t(i+j-1) t(i+j)
|        |            |        X
p(j-k)   p(j-k+1) ... p(j-1)   p(j)
|        |            |        ?
p(0)     p(1)   ...   p(k-1)   p(k)
```
此时相当于将`P`右滑`j-k`位，另`p(k)`再和目标串字符`t(i+j)`继续进行比较

- KMP的算法实现

```cpp
int KMPStrMatching(String T, string P, int *N, int start){
    int tLen = T.length();
    int pLen = P.length();
    if(tLen - start < pLen){
        return -1;
    }
    int i=start;
    int j=0;
    while(i<tLen && j<pLen){
        if(T[i] == P[j] || j == -1){
            i++, j++;
        }else{
            j = N[j];
        }
        if(j >= pLen){
            return i-pLen;
        }else{
            return -1;
        }
    }
}
```

- 特征值数组

上述代码假设已经有特征向量数组`N`，接下来的问题是如何计算`next[]`数组，假设有子串`P`如下，我们先观察一下`next[k]`数组的一般规律。

```
A B Y ... A B X P(j)
    |         | 
```

假设现在要求`j`位置的`next[j]`的值，那么，如果`X`和`Y`相同，那么`next[j] = next[j-1] + 1`，如果不相同则考虑`X`是否和`P[0]`相同，即`X`是否等于`A`，如果相同，则`P[j]=1`，不相同则`P[j]=0`。

接下来的问题，便是怎么用下标找出`X`和`Y`。不难看出`X=P(j-1)`，由于`X`前面的尾串`AB`和首串`AB`相同，因此可以得到`next[j-1] = 2`，那么可以推理得到`Y=P(next[j-1])`。完整算法可以实现如下：

```cpp
vector<int> findNext(string P){
    int m = P.length();
    vector<int> next(m);
    next[0] = -1;
    next[1] = 2;
    int j = 2;
    while(j < m ){
        int index = next[j-1];
        if(index != 0 && P[index] == P[j-1]){
            next[j] = next[j-1] +1;
        }else{
            if(P[j-1] == P[0] ){
                next[j] = 1;
            }else{
                next[j]=0;
            }
        }
        j++;
    }
    return next;
}
```
- **算法复杂度分析**
    - 循环体中`j = N[j];` 语句的执行次数不能超过`n`次。否则
        - 由于`j = N[];` 每一行必然使得`j`减少
        - 而使得`j`增加的操作只有`j++`
        - 那么，如果`j = N[j];`的执行次数超过`n`次，最终的结果必然使得`j`为比`-1`小很多的负数。这是不可能的(`j`有时为`-1`,但是很快`+1`回到`0`)。
    - 同理可以分析出求N数组的时间为`O(m)`，<mark>KMP算法的时间为Ｏ(n+m)</mark>

### Resources

- [Pattern Matching Pointer](http://www.cs.ucr.edu/~stelo/pattern.html)
- [字符串匹配算法的描述、复杂度分析和C源代码](http://www-igm.univ-mlv.fr/~lecroq/string/)（）


