---
layout: post
list_title: Algorithms | 动态规划 | Dynamic Programming
title: 动态规划 | Dynamic Programming
categories: [Algorithms]
mathjax: true
---

## 概述

在介绍动态规划之前，我们先来想一下最符合人类思维习惯的蛮力算法（brutal force），所谓蛮力算法是指穷举所有可能的解法，或者搜索解空间中的所有分支，然后在有解中寻找符合条件的解。蛮力算法的一个问题在于每两次迭代或者搜索之间的关系是相互独立的，相互独立意味前一步迭代或者搜索所产生的信息无法被后一步所利用，这会导致这些信息的浪费，而且在某些情况下，当前的这次迭代或者搜索还会重新走一遍上次“老路”来获得前一步已经得到的这些信息，这显然这是一个非常低效的策略。

而所谓的动态规划，是一种多阶段的决策过程，它将每个阶段得到的结果不断的积累起来("Memoization")，在后面阶段的决策中充分利用这些已经得到的结果进行决策，从而避免错误的选择，进而提高整个算法的效率。

动态规划最早是由Richard Bellman在1950年提出，起初是用来求解数学中的求解最优化问题，后来这种思想也被迁移到了计算机科学中，如果一个问题具有最优子结构（即一个问题可被分解成多个相同，但规模更小的子问题，并且可以递归的找到这些问题的最优解）,则可以使用动态规划算法求解。

接下来我们先给出求解DP问题的一般方法，然后分析一系列动态规划的经典问题，包括：

1. 对穷举或者搜索问题的优化
2. 求解最优化问题

通过对这些经典问题的分析和解法的比对来体会DP的解题思路

### 动态规划设计要素

1. 将原问题分解为子问题
    - 把原问题分解为若干个子问题，子问题和原问题形式相同或类似，只不过规模变小了。子问题解决，原问题即解决
    - 子问题的解<mark>可以缓存</mark>，所以每个子问题只需要求解一次

2. 确定状态
    - 将和子问题相关的各个变量的一组取值，称之为一个`状态`，一个`状态`对应于一个或多个子问题，所谓某个`状态`下的`值`，就是这个状态所对应的子问题的`解`
    - 所有`状态`的集合，构成问题的`状态空间`。状态空间的大小解解决问题的时间复杂度直接相关。整个问题的时间复杂度是<mark>状态数目乘以每个状态所需要的时间</mark>
    - 经常碰到的情况是，K个整型变量能够成一个状态。如果这个K个整型变量的取值范围分别是`N1,N2,...,Nk`，那么，我们就可以用一个K维数组`array[N1][N2]...[Nk]`来存储各个状态的`值`。这个`值`未必是一个整数或浮点数，也可以是一个复杂的数据结构
3. 确定一些初始状态（边界状态）的值
4. 确定状态转移方程
    - 找到不同状态之间如何迁移-即如何从一个或多个值已知的状态，求出另一个状态的值。状态的迁移可以用递推公式表示，递推公式也可被称作<mark>状态转移方程</mark>
        - 递推公式可以从前往后推导，也可以从后向前推导 
    - 当选取的状态，难以进行递推时（分解出的子问题和原问题形式不一样，或不具有无后效性），考虑将状态<mark>增加限制条件后分类细化，即增加维度</mark>，然后在新的状态上尝试递推

## DP的几个例子

接下来我们来一起分析一些动态规划问题的经典例子，举些例子的目的是为了观察DP问题的形式以及解DP问题的一般规律。DP之所以难是因为它的解法通常不直观，子问题不容易想到，进而找到不到正确的状态转移方程。想要熟练的解DP问题没有什么特别好的办法，只有多练，然后总结，然后再练，如此多循环几次就多少会找到一些感觉。

### 斐波那契数列

我们先从最经典的动态规划问题-斐波那契数列开始。该问题的描述是：

> 给定数字n，和斐波那契公式，求解第n个斐波那契数

- **使用递归**

不难想到，这个问题使用递归的方法接比较容易，因为每一步的计算过程都是相同的，我们只需要让计算机不断重复这个过程即可：

```python
def fib1(n):
    if n==0 or n==1:
        return n
    else:
        return fib1(n-1)+fib1(n-2)
```
我们来分析一下，当 `n=5`时，`fib(5)`的递归计算过程如下:

```
fib(5)
fib(4) + fib(3)
(fib(3) + fib(2)) + (fib(2) + fib(1))
((fib(2) + fib(1)) + (fib(1) + fib(0))) + ((fib(1) + fib(0)) + fib(1))
(((fib(1) + fib(0)) + fib(1)) + (fib(1) + fib(0))) + ((fib(1) + fib(0)) + fib(1))
```
可以看到，上面的递归过程存在大量的重复计算，例如`fib(3),fib(2),fib(1),fib(0)`，由于这些计算结果没有缓存，因此每次计算一个新的fib值时，这几个数都要重新计算。我们来分析一下使用递归解法的时间复杂度，我们先看看斐波那契数列的递推式$f_n = f_{n-1} + f_{n-2}$，其中$f_0=1, f_1 = 1$。可以对该数列求和得到

$$
f_n=\frac{1}{\sqrt 5}(\frac{1+\sqrt 5}{2})^{n+1} - \frac{1}{\sqrt 5}(\frac{1- \sqrt 5}{2})^{n+1}
$$

显然这是一个指数函数。接下来我们分析算法的时间复杂度，不难看出，时间复杂度的递推式为：$T(n) = T(n-1) + T(n-2) + 1 \thinspace (n>1)$，同样为一个斐波那契数列，可以得出$T(n)$正比于斐波那契的递推式$f(n)$，即

$$T(n) = 2 * fib(n+1)-1 = O(fib(n+1)) = O(\Phi^n) = O(2^n)$$

<img src="{{site.baseurl}}/assets/images/2007/09/fib-1.png" width="70%"/>

显然它的时间复杂度递推式也是只呈指数级增长的，这类量级的算法在实际应用中显然是不适用的，实际测试可发现当`n>60`时，算法运行时间将变成秒级。

- **使用递归+Memoization**

可以看到，使用简单的递归的一个比较大的问题是存在大量的重复运算，因此一个简单的优化是对中间计算的结果进行缓存（memoization），整个计算过程还是递归向下的

```python
#Recursion + Memoization
def fib2(n,memo):
    if n in memo:
        return memo[n]
    else:
        if n==0 or n==1:
            return n
        else:
            v = fib2(n-1,memo)+fib2(n-2,memo)
            memo[n] = v
            return v
```
引入缓存后，计算得到效率大大提升。我们再来分析下时间复杂度，读缓存需要$O(1)$的时间，fib序列中的每一项只计算一次，共需要$O(n)$时间，因此最后时间复杂度变成了：

$$
T(n) = O(n)+O(1) = O(n)
$$

- **使用迭代**

另一个思路就是将上面的递归+memoization的解法改为迭代：

```python
#DP
def fib3(n):
    fib ={} 
    fib[0] = 0
    fib[1] = 1
    for i in range(2,n+1):
        fib[i] = fib[i-1]+fib[i-2]

    return fib[n]
```

改为迭代算法后，时间复杂度依然为$O(n)$，空间复杂度仅为$O(n)$。这种计算方式和使用递归+缓存的方式基本一致，不同的是计算方向，将递归这种自顶而下的计算方式改为了自底向上的迭代。也可以将其理解为是一种拓扑序列结构，项与项之间有依赖关系:

<img src="{{site.baseurl}}/assets/images/2007/09/fib-2.png" width="50%" style="margin-left:auto; margin-right:auto;display:block"/>


- 启发

斐波那契数列这个例子，给我们的一个启示是，缓存每一步的计算结果的重要性，因此理解DP的一个角度为：

$$
DP \approx Recursion + Memoization
$$

但这个例子也会给人造成一种错觉，即DP就是在原来算法的基础上增加缓存即可。就这个例子而言，确实是这样，不过动态规划的思想远不止增加缓存这么简单，在接下来的几个例子中，我们将会看到DP的其它应用。

### 两点间路径

> A robot is located at the top-left corner of a m x n grid (marked 'Start' in the diagram below).The robot can only move either down or right at any point in time. The robot is trying to reach the bottom-right corner of the grid (marked 'Finish' in the diagram below).How many possible unique paths are there?

<img src="{{site.baseurl}}/assets/images/2007/09/dp-1.png">

上面问题是说，在一个`m x n`的棋盘上（n行，m列），在每一个格子上只能向右或者向下两种走法，那么从`start`(左上角)走到`finish`(右下角)有几种不同的走法？如上面例子中，到达`end`的路径有三条，分别是:

```
1. Right -> Right -> Down
2. Right -> Down -> Right
3. Down -> Right -> Right
```

- **蛮力算法**

首先想到的是使用蛮力算法，类似走迷宫，搜索可抵达边界的每条路径，当搜走到右下角时，记作一次发现，将每次发现的次数计起来即可得到最终解。由之前介绍的深搜+回溯的思路，不难得出穷举的解法：

1. 先一直向右走，走到边界回溯后向下
2. 重复上述过程

```cpp
//m列，n行，起点pt = {1,1}, 终点target = {m,n} , num用来收集结果
void dfs(int m, int n, pair<int,int>& pt, pair<int,int>& target, int& num ){
    //走到边界
    if(pt.second > n || pt.first > m){
        return ;
    }
    //走到右下角
    if(pt.first == target.first && pt.second == target.second){
        num ++;
        return;
    }
    //向右前进
    pt.second += 1;
    dfs(m,n,pt,target,num);
    pt.second -= 1;
    
    //向下前进
    pt.first += 1;
    dfs(m,n,pt,target,num);
    pt.first -= 1;
}
int uniquePaths(int m, int n) {
    pair<int,int> pt = {1,1};
    pair<int,int> target = {m,n};
    int num = 0;
    dfs(m,n,pt,target,num);
    return num;
}


dfs(m,n,{1,1},{m,n},num);
```
上述解法的确能够穷举出所有到达右下角的路径，然而效率确非常低。不难看出，上述算法是一种正向的，符合人类直觉的思考方式，即从起点出发穷举所有到达终点的可能性。我们来分析一下其时间复杂度，假设$m=3,n=2$，字母$R$表示向右走，字母$B$表示向下走，左上角为用`start`表示，右下角为`end`表示，则生成的递归树为：

<img src="{{site.baseurl}}/assets/images/2007/09/dp-2.png" style="margin-left:auto; margin-right:auto;display:block">

上述递归树可以看出，在所有的叶节点中，只有3个是有效的，其余的均为无效搜索。从某一点出发均有两条路径，因此算法的时间复杂度是呈几何级数增长的

$$
T(n) = 2T(n-1) ~ O(2^n)
$$

- **使用DP**

使用动态规划该如何思考这个问题呢，首先想到的是，能否和上个例子一样使用缓存，但是对于这个问题，由于每次搜索的路径都不同，不存在重复计算，因此缓存没有用。这时我们需要转变思路，寻找反直觉的方式，比如尝试从终点开始向前递推，则思路或许会被打开。

从终点出发，问题将简化成：“如果要到达`end`，需要先到达`(x,m-1)`，或者到达`(n-1,y)`，那么到达`end`的路径数就等于到达`(x,m-1)`加上到达`(n-1,y)`的路径数”。同理，对每个点均可应用上述条件，则可得出状态转移方程：

$$
dp(x,y) = dp(x-1, y) + dp(x, y-1)
$$

以`m=3,n=3`为例，则到达每个点的路径数为：

```cpp
/*
m=3, n=3
---------------
| 0 |  1 |  1 |
|---|----|----|
| 1 |  2 |  3 |
|---|----|----|
| 1 |  3 |  6 |
---------------
*/
int uniquePaths(int m, int n) {
    if(m == 0 || n == 0){
        return 0;
    }
    if( m == 1 || n == 1){
        return 1;
    }
    vector<vector<int>> dp(n,vector<int>(m,1));
    for(int x=1;x<n;x++){
        for(int y=1; y<m; y++){
            dp[x][y] = dp[x-1][y] + dp[x][y-1];
        }
    }
    return dp[n-1][m-1];
}
```

- 启发

这个例子给我们的启发是：使用DP可以优化蛮力算法，如果说蛮力的搜索算法是从“源头”出发的正向过程，那么DP的思路则是从“终局”出发的反向过程。这里所谓的反向是指从终点向前推进，将“终局”问题化成与之相等价的，规模更小的子问题。因此，当我们遇到需要蛮力解决的搜索问题时，不妨试着从后向前想，看能否找到突破口。但需要注意的是，不是所有的DP问题都是由终点向原点递推，递推方式取决于子问题的划分方式。


### 最长上升子序列（LIS）

求解上升子序列是另一个动态规划中比较经典的NP问题，问题如下：

> 一个数的序列ai，当a1 < a2 < ... < aS的时候，我们称这个序列是上升的。对于给定的一个序列(a1, a2, ..., aN)，我们可以得到一些上升的子序列(ai1, ai2, ..., aiK)，这里1 <= i1 < i2 < ... < iK <= N。比如，对于序列(5,7,4,-3,9,1,10,4,5,8,9,3)，有它的一些上升子序列，如(5, 7), (-3, 1, 4)等等。这些子序列中最长的长度是6，比如子序列(-3,1,4,5,8,9)。你的任务，就是对于给定的序列，求出最长上升子序列的长度。

1. 将原问题拆解成若干个子问题
    
    按照DP解题的思路，第一步还是划分子问题，这次的子问题相对来说还比较好想出来，即原问题是求解整个序列的LIS长度，那么子问题可以定义为求解某个子序列的LIS长度。为了存放每个子序列的LIS长度，我们需要一个数组`L`，其中`L[i]`用来存放每个子序列的LIS长度值。

2. 尝试计算`L[i]`

    虽然我们知道了子问题的大概模样，但是对子问题的很多细节还不是很清楚，比如：（1）如何划分子序列？（2) 子序列的LIS值`L[i]`怎么计算？（3）得到子序列的LIS值后，这个值和原序列的LIS值有什么对应关系？为了解答这些问题，我们不妨从第一个字符开始推演，假设子序列为`a[0..i]`，`L[i]`表示第前`i`个字符（不包括`a[i]`）前的LIS长度值，接下来我们我们可以来观察一下`L[i]`的值：

    ```shell
    a[i] = | 5 | 
    L[i] = | 1 |                //LIS: 5
    a[i] = | 5 | 7 |
    L[i] = | 1 | 2 |            //LIS: 5,7
    
    ...

    a[i] = | 5 | 7 | 4 | -3| 9 | 1 | 10 | 4 | 5 | 8 | 9 | 3 |        
    L[i] = | 1 | 2 | 2 | 2 | 3 | 3 | 4  | 4 | 4 | ? |           //LIS: 5,7,9,10      
    ```
    此时当`a[i]=8`时，根据我们追踪的LIS序列，8<10， 因此LIS不应该追加，但此时我们发现LIS不止一个，除了`5,7,9,10`之外，还有另一个`-3,1,4`，而8可以附加在该LIS之后，新的LIS序列变成了`-3,1,4,8`

    ```
    a[i] = | 1 | 7 | 3 | 5 |         
    L[i] = | 1 | 2 | 2 |        //LIS: 1,7
    ```


3. 计算子序列的`L[i]`值
    


- 确定状态

    每个子问题只和一个状态有关，即数字的位置`k`，而状态`k`对应的值为以`a[k]`做为终点的最长上升子序列的长度。状态个数为`N`

- 确定状态转移方程

    假设对于第`k`个位置的最长子序列值为`maxLen(k)`, 要找到它和前面值的递推关系，则有:

```
maxLen(1) = 1
maxLen(k) = max{ maxLen(i) | 1<=i<k and a[i] < a[k] and k != 1 } + 1
```
`maxLen(k)`的值，就是在`a[k]`左边，“终点”数值小于`a[k]` ，且长度最大的那个上升子序列的长度再加`1`。因为`ak`左边任何“终点”小于`ak`的子序列，加上`ak`后就能形成一个更长的上升子序列。

```c
int a[n];
int maxLen[n];
for(int i=2; i<n; i++){
    for(int j=1; j<i; j++){
        if(a[i]>a[j]){
            maxLen[i] = max(maxLeb[i],maxLen[j]+1)
        }
    }
}
```

### Maximum Subarray

第二个动态规划的例子是序列的最大子段和，问题描述如下:

> 给定$n$个数(可以为负数)的序列$(a_1,a_2,...,a_n)$，求其最长子段和。例如，给定序列 `[-2,1,-3,4,-1,2,1,-5,4]`,结果为`6`，最长子段为`[4,-1,2,1]`。
 



### 能用动规解决问题的特点

1. 问题具有最优子结构性质。如果问题的最优解所包含的子问题的解也是最优的，我们就称该问题具有最优子结构性质
2. <mark>无后效性</mark>。当前的若干状态值一旦确定，则此后过程的演变就只和这若干个状态的值有关，和之前的状态无关

### DP与递归

对于递归问题，我们可以用以下方法，将其转化为动归问题：

1. 递归函数有n个参数，就定义一个n为数组
2. 数组的下标是递归函数参数的取值范围
3. 数组的元素值是递归函数的返回值，这样就可以从边界值开始，逐步填充数组，相当于计算递归函数值的逆过程

因此，从某种意义上讲，<mark>所谓的动态规划，也可以理解为使用递归找出算法的本质，并给出一个初步的解之后，再将其等效的转化为迭代的形式</mark>



## Resources

- [MIT 6.006 Introduction to Algorithms, Fall 2011](https://www.youtube.com/watch?v=OQ5jsbhAv_M&t=1952s)
- [MIT 6.046J Design and Analysis of Algorithms, Spring 2015](https://www.youtube.com/watch?v=Tw1k46ywN6E)
- [CS106B-Stanford-YouTube](https://www.youtube.com/watch?v=NcZ2cu7gc-A&list=PLnfg8b9vdpLn9exZweTJx44CII1bYczuk)
- [Algorithms-Stanford-Cousera](https://www.coursera.org/learn/algorithms-divide-conquer/home/welcome)
- [算法与数据结构-1-北大-Cousera](https://www.coursera.org/learn/shuju-jiegou-suanfa/home/welcome)
- [算法与数据结构-2-北大-Cousera](https://www.coursera.org/learn/gaoji-shuju-jiegou/home/welcome)
- [算法与数据结构-1-清华-EDX](https://courses.edx.org/courses/course-v1:TsinghuaX+30240184.1x+3T2017/course/)
- [算法与数据结构-2-清华-EDX](https://courses.edx.org/courses/course-v1:PekingX+04833050X+1T2016/course/)
- [算法设计与分析-1-北大-Cousera](https://www.coursera.org/learn/algorithms/home/welcome)
- [算法设计与分析-2-北大-EDX](https://courses.edx.org/courses/course-v1:PekingX+04833050X+1T2016/course/)