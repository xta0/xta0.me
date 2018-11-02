---
layout: post
list_title: 数据结构基础 | Data Structure | 字符串 | String
title: 字符串
sub_title: String Algorithms
mathjax: true
categories: [DataStructure]
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

所谓KMP算法，是指在不回溯的前提下，通过某种算法来减少不必要的比较，从而最大限度的减少比较次数。在上面朴素法匹配的过程中，不难发现有很多的重复比对，比如下图所示，有目标`T`和模式`P`两个串，他们在某`j`个长度上匹配成功，即`t(i)t(i+1)...t(i+j-1) = p(0)p(1)...p(j-1)`，

```
T t(0) t(1) ... t(i) t(i+1) ... t(i+j-2) t(i+j-1) t(i+j)... t(n-1)
                 |   |          |        |        X
P               p(0) p(1)   ... p(j-2)   p(j-1)   p(j)
```
但是第`j+1`个位置出现了不匹配，即`p(j) != t(i+j)`，如果按照上面朴素算法，下一步比较应该是将`p(0)`向右移动一位，使其指向`t(i+1)`，即：

```
T t(0) t(1) ... t(i) t(i+1) ... t(i+j-2) t(i+j-1) t(i+j)... t(n-1)
                |    |          |        |        X
P               p(0) p(1)   ... p(j-2)   p(j-1)   p(j)
朴素匹配下一步          p(0) p(1) ... p(j-3)  p(j-2) p(j-1) p(j)
```

这是我们可以分析一下这一步是否有必要，假如：

1. `p(0)`和`p(1)`不相等，由于`p(1)=t(i+1)`，则有`p(0) != t(i+1)`，显然这一步匹配的结果是可以预先知道的。
2. `p(0)`和`p(1)`相等，由于`p(1)=t(i+1)`，则有`p(0) = t(i+1)`，同样，这一步匹配的结果也是可以预先知道的。

因此，无论`p(0)`和`p(1)`的关系如何，这步比较都是多余的，我们只需要通过某种方式就可以提前确定结果，因此朴素的匹配算法有是存在一定冗余的，这个冗余可以理解为，每次目标串`T`都需要进行回溯（已经比较到`t(i+j)`了，发现不等，需要回到`t(i+1)`的位置），因此我们该怎么移动模式串`P`来减少冗余的比较呢？仔细思考不难发现，这部分冗余产生的原因是因为朴素的匹配算法没有把模式串中已经比配成功的部分记录下来，这部分信息浪费掉了，导致每次都需要重新来过，我们如果可以把这部分信息以某种方式加以利用，当出现`p(0) != t(i+1)`时，根据这部分信息来做出移动位置的计算，使模式串`P`大幅度的向后滑动，则可以大大提高匹配效率。

接下来我们就研究一下，这些没有被利用的信息到底是什么。我们还是以上面这种情况作为一般情况进行分析，假设我们有下面两个串：

```
T |.... R   E   G   R   E   T....
        |   |   |   |   X               
P |     R   E   G   R   O   
                    |
P |                 R   E   G   R   O 
```

当`T`和`P`串在`E`和`O`位置出现了不等，此时我们可以将`P`串直接向后移动到`R`的位置，为什么是`R`的位置，这个规律是什么呢？ 仔细观察可发现，<mark>对模式串`P`来说，如果在出现不等位置有尾串和首串相等，则可以直接将`P`向后移动到尾串的起始位置。</mark>接下来的问题就是，我们怎么知道在出现不匹配字符时，它前面的子串是否存在首尾相同的子串，以及这个子串的长度是多少呢？

这就要用到前面一节提到的字符串的`next[]`数组，当模式串`P(j)`出现失配后，检查`next[j]`的值，若`next[j]`的值不为`-1`或`0`，说明则前面子串存在相同的首尾字符串，可以将新的对其位置更新为`P(j)`。而KMP算法的核心就在于得到`next[]`数组，到这里我们可以大致写出KMP算法的雏形：

```cpp
int kmp(String T, string P, int start){
    vector<int> next = buildNext(P); 
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
            j = N[j]; //注意，i不增加
        }
        if(j >= pLen){
            return i-pLen;
        }else{
            return -1;
        }
    }
}
```


接下来我们看一个具体例子，假设有串`P`和目标串`T`如下，其中`N`为模式串`P`的特征向量，如下：

```
P =  a b a b a b b
N = -1 0 0 1 2 3 4

T = a b a b a b a b a b a b a b b
    | | | | | | x
P = a b a b a b b (i=6,j=6,N[j]=4)
```
可以看到当比较到`i=6, j=6`时，`T`和`P`不相等，此时`P[6]`的特征向量，`N[6]=4`，因此下一轮需要用`P[4]`继续和`T[6]`比较，说明首串最大长度为`4`，相当于将`P`右移`j-k`位,即`6-4=2`位，得到：

```
T = a b a b a b a b a b a b a b b
        | | | | | | x
P =     a b a b a b b (i=8,j=6,N[j]=4)
```
重复上面步骤，发现在第`i=8,j=6`时，出现了失配，继续查表得到`next[6]=4`，令`P[4]`继续和`T[8]`进行比较，因此需要将`P移动`j-k`位，即`6-4=2`位

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

上述代码并没有特征向量`next[]`是怎么来的，因此接下来的问题是如何得到`next[]`数组，假设有子串`P`如下，我们先观察一下`next[k]`数组的一般规律。

```
A B Y ... A B X P(j)
    |         | 
```

假设现在要求`j`位置的`next[j]`的值，那么，如果`X`和`Y`相同，那么`next[j] = next[j-1] + 1`，如果不相同则考虑`X`是否和`P[0]`相同，即`X`是否等于`A`，如果相同，则`P[j]=1`，不相同则`P[j]=0`。

接下来的问题，便是怎么用下标找出`X`和`Y`。不难看出`X=P(j-1)`，由于`X`前面的尾串`AB`和首串`AB`相同，因此可以得到`next[j-1] = 2`，那么可以推理得到`Y=P(next[j-1])`。完整算法可以实现如下：

```cpp
vector<int> buildNext(string P){
    int m = P.length();
    vector<int> next(m);
    next[0] = -1;
    next[1] = 0;
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
KMP算法复杂度分析：

1. 循环体中`j = N[j];` 语句的执行次数不能超过`n`次。否则
    - 由于`j = N[];` 每一行必然使得`j`减少
    - 而使得`j`增加的操作只有`j++`
    - 那么，如果`j = N[j];`的执行次数超过`n`次，最终的结果必然使得`j`为比`-1`小很多的负数。这是不可能的(`j`有时为`-1`,但是很快`+1`回到`0`)。
2. 同理可以分析出求N数组的时间为`O(m)`，<mark>KMP算法的时间为Ｏ(n+m)</mark>


## 字符串常见问题

### 字符串去重问题

字符串去重问题是一个很常见的问题，解法也有很多种，有些语言的库函数可以直接提供high-level的API进行去重。这里提供一个非常巧妙的解法，利用双指针+`set`进行一遍扫描即可。题目如下:

> Given a string that contains duplicate occurrences of characters, remove these duplicate occurrences. For example, if the input string is "abbabcddbabcdeedebc", after removing duplicates it should become "abcde".

算法思路如下：

1. 用一个set保存所有不重复的字母
2. 定义两个游标，一个read负责读字符，一个write负责写字符
3. 从字符串头部开始移动read
    - 如果read到集合里已经有的字符，则read++，继续向后搜索
    - 如果read到set里没有的字符，则向set中添加字符，同时write在当前位置写入该字符, write++, read++
4. 算法伪码如下：

    ```python
    read_pos = 0
    write_post = 0;
    while(read_pos < str.len){
        c = str[read_pos]
        if(c not in set){
            set.add(c)
            str[write_pos] = str[read_pos]
            write_pos += 1;
        }
        read_pos += 1
    }
    ```
6. 该算法的时间复杂度为`O(n)`，空间复杂度也为`O(n)`

如果我们要求不使用额外的空间，这道题该怎么解呢？显然我们需要找到另一种方式可以代替`set`对字符进行判重，思考前面的算法可以发现，`write`游标左边的字符肯定是没有重复的，因此每当`read`读取一个新字符时，我们都需要在`[0,write)`这个范围内查找一下，看看是否存在，如果存在，说明是重复字符，`read`继续向后搜索，`write`不动。如果不存在，则按照原来的逻辑`write`进行写入，同时更新`write`和`read`的位置。此时算法的时间复杂度变为了`O(n^2)`，空间复杂度为`O(1)`。

```python
read_pos = 0
write_post = 0;
while(read_pos < str.len){
    c = str[read_pos]
    //修改判重方法
    if(c not in substr(0,write)){
        str[write_pos] = str[read_pos]
        write_pos += 1;
    }
    read_pos += 1
}
```

### 分割word问题

另一个字符串常见的问题是给定一个字符串和一个词典，判断是否可以将该字符串切割成一个或多个字典中的单词。 如下图所示

<img src="{{site.base}}/assets/images/2010/07/word-break.png" width="80%">


### 回文串问题

- [5. Longest Palindromic Substring](https://leetcode.com/problems/longest-palindromic-substring/description/)
- [9. Palindrome Number](https://leetcode.com/problems/palindrome-number/description/)

### 滑动窗口问题

- [3. Longest Substring Without Repeating Characters](https://leetcode.com/problems/longest-substring-without-repeating-characters/)

### 排列问题

- [49. Group Anagrams](https://leetcode.com/problems/group-anagrams/description/)
- [438. Find All Anagrams in a String](https://leetcode.com/problems/find-all-anagrams-in-a-string/description/)

### 回文问题

- 寻找回文串，两种解法
    - 中心扩散
    - 动态规划




## Resources

- [Pattern Matching Pointer](http://www.cs.ucr.edu/~stelo/pattern.html)
- [字符串匹配算法的描述、复杂度分析和C源代码](http://www-igm.univ-mlv.fr/~lecroq/string/)
- [CS106B-Stanford-YouTube](https://www.youtube.com/watch?v=NcZ2cu7gc-A&list=PLnfg8b9vdpLn9exZweTJx44CII1bYczuk)
- [Algorithms-Stanford-Cousera](https://www.coursera.org/learn/algorithms-divide-conquer/home/welcome)
- [算法与数据结构-1-北大-Cousera](https://www.coursera.org/learn/shuju-jiegou-suanfa/home/welcome)
- [算法与数据结构-2-北大-Cousera](https://www.coursera.org/learn/gaoji-shuju-jiegou/home/welcome)
- [算法与数据结构-1-清华-EDX](https://courses.edx.org/courses/course-v1:TsinghuaX+30240184.1x+3T2017/course/)
- [算法与数据结构-2-清华-EDX](https://courses.edx.org/courses/course-v1:PekingX+04833050X+1T2016/course/)
- [算法设计与分析-1-北大-Cousera](https://www.coursera.org/learn/algorithms/home/welcome)
- [算法设计与分析-2-北大-EDX](https://courses.edx.org/courses/course-v1:PekingX+04833050X+1T2016/course/)


