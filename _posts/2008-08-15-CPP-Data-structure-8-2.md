---
layout: post
list_title: Data Structre Part 8 | In-place Sort Algorithm-2 | 内排序-2
title: 内排序算法（二）
# sub_title: In-place Sort Algorithm
mathjax: true
---

前面介绍的内排序方式主要是以元素比较作为为基本的运算单元

## 分配排序（非比较排序）

非比较排序是是一种用空间换时间的策略

### 桶排序

桶排序是借助一组向量做类似`key-value`形式的存储，其中`key`为待排数组的值，`value`为其出现的次数。排序时将待排数组按照`key-value`存放到桶中，然后再对桶按照`index`从小到大输出`value`即可。

```
待排数组:        7 3 8 9 6 1 8 1 2 

空桶：           0 0 0 0 0 0 0 0 0

建桶(count)：    0 2 1 1 0 0 1 1 2 1
index：         0 1 2 3 4 5 6 7 8 9
```

可以看到，我们只需要将桶按照index输出其对应的value个key即可完成排序

```
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

上述代码的时间复杂度为$O(n^2)$，原因是我们再第二步进行桶排输出的时候使用了嵌套循环，这个嵌套循环的目的是为了解决排序后的index问题。为了解决这个问题，我们可以将桶内存储的内容改进一下

```
待排数组:        7 3 8 9 6 1 8 1 2 

index：         0 1 2 3 4 5 6 7 8 9
count：         0 2 1 1 0 0 1 1 2 1

前若干个桶       0 1 2 3 4 5 6 7 8 9
累计的count：    0 2 3 4 4 4 5 6 8 9
```
改进后的桶内存的元素表示原数组元素对应排序后的index，例如，待排数组中的最后一个元素为2，通过查`total`数组，可知`total[2] = 3`，即`2`的前面有2个元素，2是新数组的第3个元素，因此将2放入到数组`2-1=1`的位置，同时令`count-1`。

```
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
    - 总的时间代价为 Θ(m+n)
    - <mark>适用于 m 相对于 n 很小的情况，即待排数组比较紧凑的情况</mark>
3. 空间代价：
    - m 个计数器，长度为 n 的临时数组，$Θ(m+n)$

### 基数排序



