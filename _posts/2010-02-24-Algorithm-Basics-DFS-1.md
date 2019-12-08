---
layout: post
title: 深搜与回溯 
list_title: Basic Algorithms | 深搜与回溯（一） | DFS & Backtracking Part 1
categories: [Algorithms]
mathjax: true
---

如果一个问题各状态之间的转移关系(解空间)可以用一个图来描述, 则可以使用深度搜索的方法对该问题进行求解。所谓深度搜索，就是以深度优先的原则，对所有的可能情况进行暴利枚举，直至找到最终解。所谓回溯(backtracking)是指当某次搜索发现不满足条件时，退回到前一步选择一个新的分之后重新出发，因此回溯往往伴随着状态的重置。

一个典型的深搜场景就是走迷宫，在每个分叉路口都有若干个方向供选择，我们可以沿着某一方向不断试探（深搜），当发现此路不通时，再沿原路退回到最近一个分叉路口（回溯）换另一个方向继续尝试，实际上就是去穷尽解空间中的所有分支，直到找到问题的解。

<mark>总的来说，对于深搜，非常重要的一点是，对要求解的问题建立<strong>正确的</strong>解空间树，或者决策树。解空间的结构决定了深搜策略。</mark>

深搜+回溯的伪码为：

```javascript
function dfs(array,index, ...) {
    if(some_condition){
        return;
    }
    for(i=index;i<array.size();++i){
        obj = array[i];
        choose();
        dfs(array,i, ...);
        //backtracking
        unchoose();
    }
}
```

## 排列组合问题

### 排列问题（Permutation)

Permutation问题是求解一个集合的全排列问题，例如`[1,2,3]`的全排列为`[1,2,3],[1,3,2],[2,1,3],[2,3,1],[3,1,2],[3,2,1]`。这个问题的解法有很多，比较常见的是使用DFS进行暴力搜索，搜索所有可能的结果。

前面曾提到，使用DFS进行搜索的一个先决条件是要构建正确的解空间或者决策树，我们以`[1,2,3]`为例，其解空间如下：

```
     1               2              3
    / \             / \            / \
   2   3           1   3          2   1
  /     \         /     \        /     \
 3       2       3       1      1       2
```

由这个解空间不难看出，我们可以将`1,2,3`分别作为根节点来构造一棵树，然后使用DFS对每个节点进行深度搜索，当搜索到叶子结点时便得到一个解，然后进行回溯。以第一棵树为例，我们从`1`开始沿着左路搜索到`1,2,3`，然后进行回溯，回溯到`2`之后，发现没有其它的孩子节点，因此继续回溯到`1`，然后继续沿着`1`的右边继续下降，得到`1,3,2`。依次类推深度遍历其它树，最终得到全部解。

套用前面给出的深搜+回溯的代码模板，我们首先需要构造一个`for`循环来遍历各自的孩子节点，其次，我们需要在每次递归前进行choose操作，然后在递归完成后进行unchoose操作，整个过程如下：

```cpp
//全排列-深搜
void dfs( vector<int>& nums, vector<int>& chosen, vector<vector<int>>& results){
    if(nums.size() == 0){
        results.push_back(chosen);
        return;
    }
    for(int i = 0; i<nums.size(); ++i){
        int n = nums[i];
        //set state
        chosen.push_back(n);
        nums.erase(nums.begin()+i);
        //dfs
        dfs(nums,chosen,results);
        //backtracking, unset_state
        chosen.pop_back();
        nums.insert(nums.begin()+i,n);
    }
}

vector<vector<int>> permute(vector<int>& nums) {
    vector<vector<int>> ret;
    vector<int> vec;
    dfs(nums,vec,ret);
    return ret;
};
```

接下来我们来分析下上面算法的时间复杂度，从`dfs`函数来看，不难看出，递归函数之间存在下面的关系：

$$
T(n) = n*T(n-1) = O(n!)
$$

使用主定理推导或者参考前面算法分析的文章可知，该算法的时间复杂度为`O(n!)`，非常高。

### 组合问题（Combination）

组合问题的解法和排列类似，不同之处在于组合不关心元素之间的顺序，比如`[1,2]`和`[2,1]`算作同一种组合，因此解空间中的分支没有排列那么多，我们还看上面`[1,2,3]`的例子：


```
     1             2            3
    / \             \           
   2   3             3
  /              
 3              
```
对于`[1,2,3]`的组合有下面几种情况`[1],[2],[3],[1,2],[1,3],[2,3],[1,2,3]`，从代码层面看，我们只需要将上述排列代码做如下修改

1. 在进行`for`循环遍历时，需要改变每次起始搜索的`index`的值，`index`值是递增的，例如当我们走完`[1,2,3]`后，我们需要回溯到`1`，然后接着走到`3`，此时对于`3`来说，它不能再回头遍历`2`，因为`[1,2,3]`和`[1,3,2]`是相同的组合
2. 由于`index`的递增性使我们在遍历数组的时候无需修改数组，例如第一次搜索我们choose了`1`，第二次搜索由于index递增，可以直接上我们选择到`2`,无需将`1`移除array


```python
def dfs(self,arr,index,choose,result):
    #不需要递归基
    #收集结果
    result.append("".join(choose))
    #for循环从index位置开始
    for i in range(index,len(arr)):
        #choose
        x = arr[i]
        choose.append(x)
        #dfs
        self.dfs(arr,i+1,choose,result) 
        #unchoose
        choose.pop()

def combination(self,arr,n):
    choose = []
    result = []
    self.dfs(arr,0,choose,result)
    return result
```

上面代码可以搜索出`arr`的所有组合，在`dfs`函数中无需增加递归基，因为由于`index`递增，当`for`循环执行完后，函数自动返回。组合问题的时间复杂度计算公式如下

$$
T(n) = T(n-1) + T(n-2) + ... + T(1) = O(2^n)
$$

使用主定理求解上面公式，其时间复杂度虽然比排列问题要低，但仍然是指数级别的。

关于组合的题目往往有很多变种，比如求解一个数组的所有subsets，本质上还是组合问题，或者求解一个数组中`k`个数的组合，此时只需要为`dfs`函数增加一个递归基，用来追踪递归深度即可:

```python
def dfs(self,curr,depth,arr,index,choose,result):
    #追踪递归深度
    if depth == curr:
        result.append("".join(choose))
        return 
    
    for i in range(index,len(arr)):
        #choose
        x = arr[i]
        choose.append(x)
        curr += 1
        #dfs
        self.dfs(curr,depth,arr,i+1,choose,result)
        #unchoose
        curr-=1
        choose.pop()
```

### 关于排列组合的更多问题

- [46. Permutations](https://leetcode.com/problems/permutations/description/)
- [47. Permutations II](https://leetcode.com/problems/permutations-ii/description/)
- [78. Subsets](https://leetcode.com/problems/subsets/description/)
- [77. Combinations](https://leetcode.com/problems/combinations/description/)
- [40. Combination Sum II](https://leetcode.com/problems/combination-sum-ii/description/)
- [216. Combination Sum III](https://leetcode.com/problems/combination-sum-iii/description/)

## 括号问题

括号问题是另一类比较经典的搜索问题，它包含下面几种类型的题目：

1. 判断括号是否匹配
2. 生成所有可能的括号组合
3. 添加或删除括号使所有括号匹配

上面三个问题中，掌握第一个问题是解决后面问题的先决条件，它并不需要使用到搜索，简单遍历一遍输入数据即可

```cpp
bool valid(string& s){
    int oc; //未闭合的左括号 数量
    int cc; //未闭合的右括号 数量
    for(auto c : s){
        if(c == '{'){
            oc ++;
        }else{
            //遇到右括号，先看是否有未闭合的左括号
            if(oc){
                oc--;
            }else{
                //如果左括号都闭合，那么右括号为非闭合
                cc ++;
            }
        }
        return cc+oc; //返回未闭合的左右括号数量
    }
}
```
有了上面函数做基础，我们便可以分析第二个和第三个问题

### 生成所有括号组合

### 添加删除括号



### 更多括号问题

- [22. Generate Parentheses](https://leetcode.com/problems/generate-parentheses/description/)




## Resources

- [CS106B-Stanford-YouTube](https://www.youtube.com/watch?v=NcZ2cu7gc-A&list=PLnfg8b9vdpLn9exZweTJx44CII1bYczuk)
- [Algorithms-Stanford-Cousera](https://www.coursera.org/learn/algorithms-divide-conquer/home/welcome)
- [算法与数据结构-1-北大-Cousera](https://www.coursera.org/learn/shuju-jiegou-suanfa/home/welcome)
- [算法与数据结构-2-北大-Cousera](https://www.coursera.org/learn/gaoji-shuju-jiegou/home/welcome)
- [算法与数据结构-1-清华-EDX](https://courses.edx.org/courses/course-v1:TsinghuaX+30240184.1x+3T2017/course/)
- [算法与数据结构-2-清华-EDX](https://courses.edx.org/courses/course-v1:PekingX+04833050X+1T2016/course/)
- [算法设计与分析-1-北大-Cousera](https://www.coursera.org/learn/algorithms/home/welcome)
- [算法设计与分析-2-北大-EDX](https://courses.edx.org/courses/course-v1:PekingX+04833050X+1T2016/course/)



