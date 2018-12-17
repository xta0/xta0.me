---
layout: post
list_title: 数据结构基础 | Data Structre | 内排序-2 | Sort Algorithm Part 2
title: 内排序算法（二）
# sub_title: In-place Sort Algorithm
mathjax: true
categories: [DataStructure]
---

前面介绍的几种排序方式主要是以元素比较作为为基本的运算单元，这种排序方式所能达到的上限是$O(N\log{N}$。本文会介绍三种时间复杂度为$O(N)$的排序算法，分别是桶排序，计数排序与基数排序。因为这些排序算法的时间复杂度为线性的，因此也叫做线性排序（Linear Sort）。之所以能做到线性复杂度原因在于这些排序算法并不是基于元素之间的比较和交换，而是采用另外一种时间换空间的思路来提高排序效率。

### 桶排序

对于桶排序有很多种理解和实现，我们先看一种最简单的，这种方式是借助一组向量做类似`key-value`形式的存储，其中`key`为待排数组的值，`value`为其出现的次数。排序时将待排数组按照`key-value`存放到桶中，然后再对桶按照`index`从小到大输出`value`即可。

```
待排数组:        7 3 8 9 6 1 8 1 2 

空桶：           0 0 0 0 0 0 0 0 0

建桶(count)：    0 2 1 1 0 0 1 1 2 1
桶index：        0 1 2 3 4 5 6 7 8 9
```

可以看到，我们只需要将桶按照index输出其对应的value个key即可完成排序

```cpp
void bucketSort(vector<int>& v){
    vector<int> bucket(v.size(),0);
    for(auto x:v){
        bucket[x] ++;        
    }
    int index = 0;
    for(int i=0;i<bucket.size();i++){
        for(int j=0; j<bucket[i];j++){
            v[index++] = i; 
        }
    }
}
```

上述代码的时间复杂度为$O(n*m)$，原因是我们再第二步进行桶排输出的时候使用了嵌套循环，这个嵌套循环的目的是为了解决排序后的index问题。这种桶排序方式的优点是对待排数据没有要求，如果待排数据足够随机，则时间复杂度可达到$O(N)$。这种方式的弊端在于桶的数量和待排数据的数量相同，如果待排数据有重复，则会导致有空桶，造成空间的浪费。在极端情况下（所有数据都相同，而且都位于最后一个桶中）的时间复杂度为$O(N^2)$。

另一种桶排序的方式为使用有限个桶，将待排数据分散到几个有序的桶中，每个桶中的数据再进行单独排序。桶内排序完成后，再把每个桶中的数据按照顺序依次取出，组成的序列就是有序的了。

```
待排数组: 
26,7,30,5,8,22,11,38,29,35,10

桶状态:
|5,7,8|     | 10,11 |   | 22,26,29|     | 30,35,38 |
|-----|     |-------|   |---------|     | ---------|      
| 0-9 |     | 10-19 |   | 20-29   |     | 30-39    |
```

我们来分析一下这种桶排序方式的时间复杂度，如果待排序数据有`n`个，我们将它们划分到`m`个桶内，则每个桶内有`k=n/m`个元素


### 计数排序(Counting Sort)

所谓计数排序是指在桶排序的基础上，增加一个数组保存每个元素的累计值，还是桶排序的第一个例子

```
待排数组:        7 3 8 9 6 1 8 1 2 

index：         0 1 2 3 4 5 6 7 8 9
count：         0 2 1 1 0 0 1 1 2 1

前若干个桶       0 1 2 3 4 5 6 7 8 9
累计的count：    0 2 3 4 4 4 5 6 8 9
```
改进后的桶内存的元素表示原数组元素对应排序后的index，例如，待排数组中的最后一个元素为2，通过查`total`数组，可知`total[2] = 3`，即`2`的前面有2个元素，2是新数组的第3个元素，因此将2放入到数组`2-1=1`的位置，同时令`count-1`。

```cpp
void bucketSort(vector<int>& v){
    //copy original array
    vector<int> tmpArray(v);
    //建桶
    vector<int> bucket(v.size(),0);
    for(auto x:v){
        bucket[x] ++;        
    }
    //更新桶内count值
    for(int i=1; i<v.size();i++){
        bucket[i] = bucket[i]+bucket[i-1];
    }
    //产生排序数组
    for(int i=0;i<v.size();i++){
        int x = tmpArray[i];
        //x在bucket[x]-1的位置
        int index = bucket[x]-1;
        //将x放到排序后的位置
        v[index] = x;
    }
}
```

- 算法分析

1. 数组长度为`n`, 所有记录区间`[0, m)`上
2. 时间代价
    - 总的时间代价为 $Θ(m+n)$
    - <mark>适用于$m$相对于$n$很小的情况，即待排数组比较紧凑的情况</mark>
3. 空间代价：
    - $m$个计数器，长度为$n$的临时数组，$Θ(m+n)$





## Resources 

- [CS106B-Stanford-YouTube](https://www.youtube.com/watch?v=NcZ2cu7gc-A&list=PLnfg8b9vdpLn9exZweTJx44CII1bYczuk)
- [Algorithms-Stanford-Cousera](https://www.coursera.org/learn/algorithms-divide-conquer/home/welcome)
- [算法与数据结构-1-北大-Cousera](https://www.coursera.org/learn/shuju-jiegou-suanfa/home/welcome)
- [算法与数据结构-2-北大-Cousera](https://www.coursera.org/learn/gaoji-shuju-jiegou/home/welcome)
- [算法与数据结构-1-清华-EDX](https://courses.edx.org/courses/course-v1:TsinghuaX+30240184.1x+3T2017/course/)
- [算法与数据结构-2-清华-EDX](https://courses.edx.org/courses/course-v1:PekingX+04833050X+1T2016/course/)
- [算法设计与分析-1-北大-Cousera](https://www.coursera.org/learn/algorithms/home/welcome)
- [算法设计与分析-2-北大-EDX](https://courses.edx.org/courses/course-v1:PekingX+04833050X+1T2016/course/)

