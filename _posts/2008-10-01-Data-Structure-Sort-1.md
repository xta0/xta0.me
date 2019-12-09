---
layout: post
list_title: Basic Data Strutures | Sort Part 1
title: 内排序算法(一)
# sub_title: In-place Sort Algorithm
mathjax: true
categories: [DataStructure]
---

> 所谓内排序是指整个排序过程是在内存中完成的，与之相对应的是我们后要介绍的外排序算法，即排序过程需要用到外部存储。

### 直接选择排序

所谓直接选择排序，是指在每次排序的过程中，依次选出剩下的未排序记录中的最小记录，其**算法思想**为：

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
- **算法分析**

1. 空间代价 $O(1)$
2. 时间代价
    - 比较次数：$\Theta(n^2)$
    - 交换次数：$n-1$
    - 总时间代价：$\Theta(n^2)$

### 堆排序

上面介绍的直接排序是直接从剩余记录中线性查找最大记录，而堆排序则可以借助堆的性质，在`log(n)`的时间内找到最大数或最小数。

> 关于堆相关知识介绍可参考[之前文章](https://xta0.me/2010/07/22/Data-Structure-Binary-Heap.html)

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

- **算法分析**

1. 建堆：$Θ(n)$
2. 删除堆顶: $Θ(log{n})$
3. 一次建堆，n 次删除堆顶
4. <mark>总时间代价为$Θ(nlog{n})$</mark>
5. 空间代价为$Θ(1)$

### 直接插入排序

插入排序类似我们队扑克牌进行排序。我们可以首先将数组分为两部分，第一部分是有序的，第二部分是无序的。假设数组的前`i`个数是有序的，当插入元素`e`时，我们需要将该元素与前`i`个元素一次比较来找到插入位置，整个数组的有序部分会依次增长，最终达到整体有序。上述算法中，我们要做的就是每次将注意力都放到无序部分的首元素上，即`e`的值。

```cpp
Sorted     Unsorted
[.....]  e  [......]
L[0, i)     L[i+1, n)

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

- **算法分析**

1. 最佳情况：n-1次比较，2(n-1)次移动，$\Theta(n)$
2. 最坏情况：$\Theta(n^2)$
    - 比较次数：$\sum{i=1}{n-1}i=n(n-1)/2$ = $\Theta(n^2)$

直接插入排序的两个性质：

1. 在最好情况（序列本身已是有序的）下时间代价为$Θ(n)$
2. 对于短序列，直接插入排序比较有效

### 冒泡排序

- **算法思想**

冒泡排序的主要思想为，不停地比较相邻的记录，如果不满足排序要求，就交换相邻记录，直到所有的记录都已经排好序

```cpp
void bubbleSort(vector<int>& vec){
    int sz = vec.size();
    //每次外层循环归位一个数
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

- **算法分析**

1. 空间代价：$Θ(1)$
2. 时间代价分析
    - 最少：$Θ(n)$
    - 最多：交换次数最多为 $Θ(n^2)$，最少为$0$，平均为$Θ(n^2)$

### 归并排序

归并排序和接下来的快速排序是两种需要重点掌握的排序方法，它们在实际中应用非常广泛。我们先从归并排序开始。归并排序是1945年由冯.诺依曼提出的，是分治+递归的典型应用，其**算法思想**为：

1. 将待排序数组分成两个相等的子数组
2. 左子数组递归排序
3. 右子数组递归排序
4. 合并左右子数组的排序结果

归并排序的思路很简单，关键在于如何merge两个有序数组，其规则为：

1. 比较左边数组和右边数组的第一个元素(index = 0)，假设左边的元素小（如果右边元素小，方法相同），则将左边的元素放入merge后的数组，左边数组index+1=1，右边数组index=0不变。
2. 比较左边index = 1 和右边 index = 0，那边大那边的index+1
3. 重复步骤2

对于merge有两种不同的实现方式，一种方式引入一个临时数组来保存原数组，另一种则是原地归并，这里主要介绍原地归并算法

```cpp
//原地归并排序
void mergeSort(vector<int>& v){
    if(v.size() <= 1){
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
    int i = 0; //index of the merged array
    while(i1<left.size() && i2<right.size()){
        if(left[i1] <= right[i2]){
            v[i++] = left[i1++];
        }else{
            v[i++] = right[i2++];
        }
    }
    //append left half
    while(i1<left.size()){
        v[i++] = left[i1++];
    }
    //append right half
    while(i2<right.size()){
        v[i++] = right[i2++];
    }
}
```

观察上述代码可以发现，在merge的过程中，有两步操作：

1. `while`循环，通过比较`left[i1]`和`right[i2]`向`v`中填充，循环结束条件为两个数组谁先到达末尾
2. 未到达末尾的数组将其剩余部分追加到`v`中。

- **算法分析**

1. 空间代价：$Θ(n)$ / $O(1)$
2. 划分时间、排序时间、归并时间
    - $T(n) = 2T(n/2)+cn$, $T(1) = 1$
    - merge过程消耗的时间为$O(n)$
    - 最大、最小以及平均时间代价均为$Θ(nlog{n})$

### 快速排序

快速排序是20世纪十大算法之一，由Tony Hoare在1962年提出，同样是一种基于分治策略的排序算法，和归并排序不同的是，它不是都将数组划分为两个等长的子数组，子数组的长度是随机的，取决于选取的轴值（pivot point）。快速排序的**算法思想**为：

1. 选择轴值 (pivot)
2. 将序列划分为两个子序列 L 和 R，使得 L 中所有记录都小于或等于轴值，R 中记录都大于轴值
3. 对子序列 L 和 R 递归进行快速排序

关于轴值的选择，其原则是尽可能使L，R长度相等，常用的策略有

1. 选择最左边的记录
2. 随机选择
3. 选择平均值

快速排序的伪码如下：

```javascript
quick_sort(arr, l, r){
    if l >= r: return
    pivot = partition(arr,l,r) //获得轴值
    quick_sort(arr,l,pivot-1)
    quick_sort(arr,pivot+1,r)
}
```
对于partition函数的实现有多种，这里参考了《编程珠玑》中`sort3`版本，伪码如下

```javascript
partition(arr,lo,hi){
    i = lo;
    j = hi+1
    p = arr[lo]
    loop:
        do i++ while i<= hi && arr[i] < p
        do j-- while arr[j] > p
        if i > j:
            break
        swap( arr[i], arr[j])
    //swap pivot
    swap( arr[lo], arr[j] )
    return j
}
```
想写出没有bug的快排实际上并不容易，在上述代码中，有下面一些细节需要特别小心：

1. 初始化右边界为`hi+1`
2. 使用`do-while`结构，先移动`i,j`，后比较
3. 先移动左边界，后移动右边界，右边界无需判断`j>=lo`
4. 归位轴值时交换`lo`和`j`

C++代码如下：

```cpp
int partition(vector<int>& arr, int lo, int hi){
    //a[lo,...,i-1],a[i],a[i+1,...,hi]
    int p = arr[lo];
    int i = lo;
    int j = hi+1;
    while(true){
        //skip左边界小于pivot的值
        do { i++; } while( i<=hi && arr[i] < p );
        //skip右边界大于pivot的值
        do { j--; } while( arr[j] > p);
        //i，j相遇
        if( i>j ){
            break;
        }
        swap(arr[i],arr[j]);
    }
    //swap pivot
    swap(arr[lo],arr[j]);
    return j;
}
void quickSort(vector<int>& arr, int left, int right){
    if(left >= right){
        return ;
    }
    //返回pivot index
    int pivot = partition(arr,left,right);
    //两段分治
    quickSort(arr,left,pivot-1);
    quickSort(arr,pivot+1, right);
}
```

- **算法分析**

1. 最差情况：
    - 时间代价：$Θ(n^2)$
    - 空间代价：$Θ(n)$
2. 最佳情况：
    - 时间代价：$Θ(nlog{n})$
    - 空间代价：$Θ(log{n})$
3. 平均情况：
    - 时间代价：$Θ(nlog{n})$
    - 空间代价：$Θ(log{n})$

### 快速选择 (Quick Select)

快速选择算法是由快速排序演变而来的一种选择算法，它可以在`O(n)`时间内从无序列数组找到第k小的元素。该算法同样是由Tony Hoare提出，因此它也被称为霍尔选择算法。快速选择的总体思路与快速排序一致，选择一个元素作为基准来对元素进行分区，将小于和大于基准的元素分在基准左边和右边的两个区域。不同的是，快速选择并不递归访问双边，而是只递归进入一边的元素中继续寻找。这降低了平均时间复杂度，从O`(nlogn)`至`O(n)`，不过最坏情况仍然是`O(n^2)`。其算法思路为

1. 选择数组第一个数作为pivot，进行partition分区，数组将分为三部分`A[0,...,p-1], A[p], A[p+1,...,n-1]`
2. 分区后我们可以知道p右边的数都比p大，因此p是`arr.size()-partition_index`大的数，令该值为`pi`
2. 如果`k=pi`，则pivot就是第`k`大的数
3. 如果`k>pi`，则第k的大数在`A[0,...,p-1]`区间，递归求解该区间
4. 如果`k<pi`，则第k的大数在`A[p+1,...,n-1]`区间，递归求解该区间

举例来说，有下面一组无序的number；`6 1 3 5 7 2 4 9 11 8`，现在想要求出数组中第`3`大的数。按照上面思路，我们首先需要对数组进行分区，按照快排的逻辑，分区后的数组如下：

```
partition num : 6
partition index : 5
2 1 3 5 4 6 [ 7 9 11 8 ]
          |
```
此时`pi = arr.size() - partition_index = 5`，即`6`是数组中第`5`大的数，现在我们要找第`3`大的数，因此我们需要在`6`的右边区间继续寻找，递归求解该区间得到

```
paritition num: 7
paritition index: 6
2 1 3 5 4 6  7 [ 9 11 8 ]
             | 
```
此时`pi = 10-6 = 4 > 3`，因此继续递归求解`7`右边区间，得到

```
paritition num: 9
paritition index: 8
2 1 3 5 4 6 7 [ 8 ]  9  11
                     | 
```
此时`pi = 10-9 = 2 < 3`，递归求解`9`右边区间，该区间只有一个元素`8`，即是第`3`大的数

算法完整代码如下：

```cpp
int quickSelect(vector<int>& arr, int lo, int hi, int k){
    if(lo >= hi){
        return arr[lo];
    }
    //partition部分同快排
    int p = partition(arr, lo, hi);    
    int index = (int)arr.size() - p;
    if( k==index ){
        return arr[p];
    }else if( k < index){
        //递归右边区间
        return quickSelect(arr, p+1, hi, k);
    }else{
        //递归左边区间
        return quickSelect(arr,lo,p-1,k);
    }
}
```

接下来我们可以来分析一下该算法的时间复杂度，上面算法中，`partition`时间为线性`O(n)`，每次递归原数组的一般区间，因此有如下式子:

$$
T(n) = T(n/2) + O(n)
$$

上述式子实际上是等比数列求和$n+n/2+n/4+...+1 = 2n-1 = O(n)$


## 小结

1. 上述各种排序方法的时间复杂度如下：

    | -- | -- | -- |
    | bubble sort | swap adjacent pairs that are out of order | $O(n^2)$ | 
    | selection sort | look for the smallest element, move to the front | $O(n^2)$ | 
    | insertion sort | build an increasingly large sorted front portion | $O(n^2)$ | 
    | merge sort | recursively divide the data in half and sort it | $O(nlog{n})$ | 
    | heap sort | place the values into a sorted tree structure | $O(nlog{n})$ | 
    | quick sort |recursively "partition" data based on a middle value | $O(nlog{n})$ |
    | bucket sort | place the values as indexes into another array | $O(n)$  | 


2. <mark>当n很小或者数组基本有序时，插入排序比较有效</mark>
3. 综合性能，快速排序最好
4. 任何基于比较的排序算法，其时间复杂度不可能低于$O(n\log(n))$，可以用决策树来证明
    1. 决策树中叶结点的最大深度就是排序算法在最差情况下需要的最大比较次数
    2. 叶结点的最小深度就是最佳情况下的最小比较次数
    3. 对`n`个记录，共有`n!`个叶结点，判定树的深度至少为`log(n!)` 



## Resources 

- [编程珠玑]()
- [CS106B-Stanford-YouTube](https://www.youtube.com/watch?v=NcZ2cu7gc-A&list=PLnfg8b9vdpLn9exZweTJx44CII1bYczuk)
- [Algorithms-Stanford-Cousera](https://www.coursera.org/learn/algorithms-divide-conquer/home/welcome)
- [算法与数据结构-1-北大-Cousera](https://www.coursera.org/learn/shuju-jiegou-suanfa/home/welcome)
- [算法与数据结构-2-北大-Cousera](https://www.coursera.org/learn/gaoji-shuju-jiegou/home/welcome)
- [算法与数据结构-1-清华-EDX](https://courses.edx.org/courses/course-v1:TsinghuaX+30240184.1x+3T2017/course/)
- [算法与数据结构-2-清华-EDX](https://courses.edx.org/courses/course-v1:PekingX+04833050X+1T2016/course/)
- [算法设计与分析-1-北大-Cousera](https://www.coursera.org/learn/algorithms/home/welcome)
- [算法设计与分析-2-北大-EDX](https://courses.edx.org/courses/course-v1:PekingX+04833050X+1T2016/course/)

