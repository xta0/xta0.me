---
layout: post
list_title: Data Structre Part 8 | In-place Sort Algorithm | 内排序
title: 内排序算法
sub_title: In-place Sort Algorithm
mathjax: true
---

本章主要讨论一些常见的内排序算法，所谓内排序是指整个排序过程是在内存中完成的。排序的算法有很多种，不同的排序方法应用场景不同，因此没有所谓的“最好”的排序方法。常用的排序算法有

| -- | -- | -- |
| bubble sort | swap adjacent pairs that are out of order | $O(n^2)$ | 
| selection sort | look for the smallest element, move to the front | $O(n^2)$ | 
| insertion sort | build an increasingly large sorted front portion | $O(n^2)$ | 
| merge sort | recursively divide the data in half and sort it | $O(nlog{n})$ | 
| heap sort | place the values into a sorted tree structure | $O(nlog{n})$ | 
| quick sort |recursively "partition" data based on a middle value | $O(nlog{n})$ |


另外还有一些其他的排序方法，比如桶排序(bucket sort)，(radix sort)等。对于排序方法，如何来衡量其性能，

## 选择排序

### 直接选择排序

所谓直接选择排序，是指在每次排序的过程中，依次选出剩下的未排序记录中的最小记录

- 算法思想

1. 遍历数组中最小的数
2. 和第0个元素交换
3. 从1开始遍历数组找最小的数
4. 和第1个元素交换
5. 重复此过程直到排序完成

```cpp
void selectionSort(vector<int>& vec){
    for(int i=0;i<vec.size();++i>){
        int minIndex = i;
        for(int j=i+1; j<vec.size(); ++j){
            if(v[j] < v[minIndex]){
                minIndex = j;
            }
        }
        swap(vec[i],vec[minIndex]);
    }
}
```
- 算法分析
1. 空间代价 $O(1)$
2. 时间代价
    - 比较次数：$\Theta(n^2)$
    - 交换次数：$n-1$
    - 总时间代价：$\Theta(n^2)$

### 堆排序

上面介绍的直接排序是直接从剩余记录中线性查找最大记录，而所谓的堆排序是指利用堆的性质，方便的依次找出最大数/最小数。

> 关于如何建堆，可参考[之前文章]()

```cpp
template <class Record>
void sort(Record Array[], int n){
    int i;
    // 建堆
    MaxHeap<Record> max_heap = MaxHeap<Record>(Array,n,n);
    // 算法操作n-1次，最小元素不需要出堆
    for (i = 0; i < n-1; i++){
        // 依次找出剩余记录中的最大记录，即堆顶
        max_heap. RemoveMax();
    }
}
```

- 算法分析

1. 建堆：$Θ(n)$
2. 删除堆顶: $Θ(log{n})$
3. 一次建堆，n 次删除堆顶
4. <mark>总时间代价为$Θ(nlog{n})$</mark>
5. 空间代价为$Θ(1)$

## 插入排序

### 直接插入排序

插入排序类似我们队扑克牌进行排序。我们可以首先将数组分为两部分，第一部分是有序的，第二部分是无序的。假设数组的前`i`个数是有序的，当插入元素`e`时，我们需要将该元素与前`i`个元素一次比较来找到插入位置，整个数组的有序部分会依次增长，最终达到整体有序。上述算法中，我们要做的就是每次将注意力都放到无序部分的首元素上，即`e`的值。

```
Sorted     Unsorted
[.....]  e  [......]
L[0, i)     L[i+1, n)
```

```cpp
void insertSort(vector<int>& v){
    for(int i=1;i<v.size();++i){ //无序部分从1开始减少
        int e = v[i]; //无序部分的首元素，待插入数字
        int j = i-1;
        for(;j>=0 ; j--){ //有序部分从0开始增加
            if(e < v[j]){
                //将大于等于e的记录向后移
                v[j+1] = v[j];
            }else{
                //得到插入位置j+1
                break;
            }            
        }
        v[j+1] = e;        
    }
}
```

- 算法分析

1. 最佳情况：n-1次比较，2(n-1)次移动，$\Theta(n)$
2. 最坏情况：$\Theta(n^2)$
    - 比较次数：$\sum{i=1}{n-1}i=n(n-1)/2$ = $\Theta(n^2)$

直接插入排序的两个性质：

1. 在最好情况（序列本身已是有序的）下时间代价为$Θ(n)$
2. 对于短序列，直接插入排序比较有效

## 交换排序

### 冒泡排序

- 算法思想

冒泡排序的主要思想为，不停地比较相邻的记录，如果不满足排序要求，就交换相邻记录，直到所有的记录都已经排好序

```cpp
void bubbleSort(vector<int>& vec){
    int sz = vec.size();
    //每次外层循环归为一个数
    for(int i=0;i<sz-1;i++){
        //j只需要循环sz-i-1次，因为已经有i个数被归位
        for(int j = 0; j<sz-i-1; j++){
            if(vec[j] > vec[j+1]){
                swap(vec,j,j+1);
            }
        }
    }
}
```

- 算法分析

1. 空间代价：$Θ(1)$
2. 时间代价分析
    - 最少：$Θ(n)$
    - 最多：交换次数最多为 $Θ(n^2)$，最少为$0$，平均为$Θ(n^2)$

### 快速排序

快速排序是20世纪十大算法之一，由Tony Hoare在1962年提出，是一种基于分治策略的排序算法，类似的还有更早提出的归并排序。

> 关于分治法，参考[这里]()

- 算法思想

1. 选择轴值 (pivot)
2. 将序列划分为两个子序列 L 和 R，使得 L 中所有记录都小于或等于轴值，R 中记录都大于轴值
3. 对子序列 L 和 R 递归进行快速排序

关于pivot值的选择，其原则是尽可能使L，R长度相等，常用的策略有

1. 选择最左边的记录
2. 随机选择
3. 选择平均值

```cpp
void quickSort(vector<int>& arr, int left, int right){
    if(left >= right){
        return ;
    }
    //选择轴值为最左边数
    int pivot = arr[left];
    int l=left,r=right;
    while(l != r){
        //skip掉右边界大于pivot的值
        while(arr[r] >= pivot && r>l){
            r--;
        }
        //skip掉左边界小于pivot的值
        while(arr[l]<=pivot && r>l){
            l++;
        }
        //如果走到这里，说明有逆序对交换
        if(r > l){
            swap(arr[r],arr[l]);
        }
    }
    //归位轴值
    swap(arr[left],arr[l]);
    //两段递归分治
    quickSort(arr,left,l-1);
    quickSort(arr,l+1, right);
}
```

- 算法分析
1. 最差情况：
    - 时间代价：$Θ(n2)$
    - 空间代价：$Θ(n)$
2. 最佳情况：
    - 时间代价：$Θ(nlog{n})$
    - 空间代价：$Θ(log{n})$
3. 平均情况：
    - 时间代价：$Θ(nlog{n})$
    - 空间代价：$Θ(log{n})$


### 归并排序

归并排序是1945年由冯诺依曼提出的，是典型的分治法，使用递归实现，

- 算法思想

1. 将列表分成两个相等的部分
2. 左边排序
3. 右边排序
4. 合并两边排序的结果

归并排序的思路很简单，有一个问题是如何merge两个有序数组，merge的规则为：

1. 比较左边数组和右边数组的第一个元素(index = 0)，假设左边的元素小（如果右边元素小，方法相同），则将左边的元素放入merge后的数组，左边数组index+1=1，右边数组index=0不变。
2. 比较左边index = 1 和右边 index = 0，那边大那边的index+1
3. 重复步骤2

merge过程消耗的时间为$O(n)$

```cpp
void mergeSort(vector<int>& v){
    if(v.szie() <= 1){
        return;
    }
    //recursive case
    //1. split vector half
    vector<int> left = vector<int>(v.begin(), v.begin()+v.size()/2);
    vector<int> right = vector<int>(v.begin()+v.size()/2, v.end());
    //2. sort halves
    mergeSort(left);
    mergeSort(right);
    //3. merge halves
    int i1 = 0; //index into left
    int i2 = 0; //index into right
    for(int i =0; i<v.size(); i++){
        //take from left IF:
        //      1. nothing remaining on the right
        //      2. thing on the left is maller
        if(i2>=right.size() || (i1<left.size() && left[i]<right[i2])){
            v[i] = left[i1];
            il++;
        }else{
            v[i] = right[i2];
            i2++;
        }
    }
}
```

