---
layout: post
list_title: Data Structre Part 8 | Introsort | 内排序
title: 内排序算法
sub_title: Introsort Algorithm
mathjax: true
---

本章主要讨论一些常见的内排序算法，所谓内排序是指整个排序过程是在内存中完成的。排序的算法有很多种，不同的排序方法应用场景不同，因此没有所谓的“最好”的排序方法。常用的排序算法有

1. bubble sort: swap adjacent pairs that are out of order
2. selection sort: look for the smallest element, move to the front
3. insertion sort: build na increasingly large sorted front portion
4. merget sort: recursively divide the data in half and sort it
5. heap sort: place the values into a sorted tree structure
6. quick sort: recursively "partition" data based on a middle value

另外还有一些其他的排序方法，比如桶排序(bucket sort)，(radix sort)等。对于排序方法，如何来衡量其性能，来说


衡量内排序的标准为时间与空间复杂度

## 选择排序

选择排序有两类，分别是直接选择排序和堆排序

### 直接选择排序

所谓直接选择排序，是指在每次排序的过程中，依次选出剩下的未排序记录中的最小记录，其步骤为：

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

不难看出，选择排序的时间复杂度为$O(n^2)$

|--|--|
|N| Runtime(ms)|
|1000| 0|
|2000| 16|
|4000| 47|
|8000| 234|
|16000| 657|
|32000| 2562|
|64000| 10265|
|128000| 41141|
|256000| 164985|

//y=[0,16,47,234,657,2562,10265,41141,164985];
//x=[1000,2000,4000,8000,16000,32000,64000,128000,256000];
### 堆排序


### 插入排序

插入排序类似我们队扑克牌进行排序，其过程为：

1. 

```cpp
void insertSort(vector<int>& v){
    for(int i=1;i<vec.size();++i>){
        int tmp = v[i];
        int j = i;
        while(j>=i && v[j-1] > tmp){
            v[j] = v[j-1];
            j--;
        }
        v[j] = temp;        
    }
}
```

- 时间复杂度

1. 最佳情况：n-1次比较，2(n-1)次移动，$\Theta(n)$
2. 最坏情况：$\Theta(n)$

### Shell排序

### 归并排序

归并是1945年由冯诺依曼提出的，是典型的分治法，使用递归实现，

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

- 时间复杂度





