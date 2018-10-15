---
layout: post
title: 算法基础 | 深搜与回溯 | DFS & Backtracking
list_title: Algorithms | 深搜与回溯 | DFS & Backtracking
categories: [Algorithms]
mathjax: true
---

## 深搜与回溯

如果一个问题各状态之间的转移关系(解空间)可以用一个图来描述, 则可以使用深度搜索的方法对该问题进行求解。所谓深度搜索，就是以深度优先的原则，不断的进行尝试+回溯，直至找到最终解。所谓回溯(backtracking)是指当搜寻到某条分支后发现不满足条件，进而退回到前一步重新出发，回溯往往伴随着状态的重置。

一个典型的深搜场景就是走迷宫，在每个分叉路口都有若干个方向供选择，我们可以沿着某一方向不断试探（深搜），当发现此路不通时，再沿原路退回到最近一个分叉路口（回溯）换另一个方向继续尝试，实际上就是去搜索解空间中的所有分支，直到找到问题的解。

<mark>总的来说，对于深搜，非常重要的一点是，对要求解的问题建立**正确的**解空间树，或者决策树。解空间的结构决定了深搜策略。</mark>

深搜模板的伪码为：

```javascript
function dfs(array,index, ...) {
    if(some_condition){
        return;
    }else{
        for(i=index;i<array.size();++i){
            //choose
            obj = array[i];
            //mark states
            set_states();
            //DFS search
            dfs(array,i, ...);
            //unmark states
            unset_state();
        }
    }
}
```

### 排列组合

- 排列问题（Permutation）

Permutation问题是求解一个集合的全排列问题，例如`[1,2,3]`的全排列为`[1,2,3],[1,3,2],[2,1,3],[2,3,1],[3,1,2],[3,2,1]`。这个问题的解法有很多，比较常见的是使用递推，我们可以先从0个元素开始，然后是1个元素，2个元素去寻找规律，最后可以找到$A(n)$和$A(n-1)$之间的关系，进行递归求解。当然针对这道题，我们也可以使用DFS的思想去搜索所有可能的组合。

正如上文所述，使用DFS进行搜索的一个先决条件是要构建正确的解空间或者决策树，我们以`[1,2,3]`为例，其解空间如下：

```
     1               2              3
    / \             / \            / \
   2   3           1   3          2   1
  /     \         /     \        /     \
 3       2       3       1      1       2
```

由这个解空间不难看出，我们可以将`1,2,3`分别作为根节点来构造一棵树，然后使用DFS对每个节点进行深度搜索，当搜索到叶子结点时便得到一个解，然后进行回溯。以第一棵树为例，我们从`1`开始沿着左路搜索到`1,2,3`，然后进行回溯，回溯到`2`之后，发现没有其它的孩子节点，因此继续回溯到`1`，然后继续沿着`1`的右边继续下降，得到`1,3,2`。依次类推深度遍历其它树，最终得到全部解。

这里有个问题，由于递归函数的自相似性，我们算法主题是构造一个`for`循环来遍历各自的孩子节点，这个`for`循环的范围该怎么选取呢？我们来简单分析一下，对于根节点`1`，可以令`i`从`1`到`2`，对于第二层的`2`它只有一个节点，于是我们希望`i`从`2`到`2`,对于第三层`3`，显然它没有子节点，函数在这里返回，开始回溯，于是`3`回溯到了`2`,由于该层的`for`循环的`i`已经为`2`，`for`循环不在执行，于是回溯到`1`，此时第一层的`for`循环继续，`i`值为`2`，于是开始右边的下降过程。理解了这个过程，就不难写出代码：

```cpp
//全排列-深搜
class Solution {
private:
    void dfs( vector<int>& nums, vector<int>& chosen, vector<vector<int>>& results){
        if(nums.size() == 0){
            results.push_back(chosen);
        }else{
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
    }
    
public:
    vector<vector<int>> permute(vector<int>& nums) {
        vector<vector<int>> ret;
        vector<int> vec;
        dfs(nums,vec,ret);
        return ret;
    };
};
```

- 组合问题（Combination）

组合问题

### N Queen

历史上一个很经典的问题是八皇后问题，这个问题可以推广到N皇后问题，比如，我们可以先求解一下四皇后问题，问题如下：

> 在国际象棋的4x4的棋盘上放置4个皇后，使任意两个皇后不在同一行上，不在同一列上，不在同一斜对角线上，问有几种可能的摆法




### Sudoku问题

我们再来看一道数独问题，求解数独问题也是一个典型的DFS搜索问题，通过不停尝试来找到最终解，数独问题描述如下:
> 将数字1到9,填入9x9矩阵中的小方格，使得矩阵中的每行，每列，每个3x3的小格子内，9个数字都会出现"。
程序的输入(左边，其中`0`为待填充部分)，输出(右边)为:

<div style=" content:''; display: table; clear:both; height=0">
    <div style="width:110px; float:left">
        1 0 3 0 0 0 5 0 9
        0 0 2 1 0 9 4 0 0
        0 0 0 7 0 4 0 0 0
        3 0 0 5 0 2 0 0 6
        0 6 0 0 0 0 0 5 0
        7 0 0 8 0 3 0 0 4
        0 0 0 4 0 1 0 0 0
        0 0 9 2 0 5 8 0 0
        8 0 4 0 0 0 1 0 7
    </div>
    <div style="width:110px; margin-left:15px;float:left">
        1 4 3 6 2 8 5 7 9 
        5 7 2 1 3 9 4 6 8 
        9 8 6 7 5 4 2 3 1 
        3 9 1 5 4 2 7 8 6 
        4 6 8 9 1 7 3 5 2 
        7 2 5 8 6 3 9 1 4 
        2 3 7 4 8 1 6 9 5 
        6 1 9 2 7 5 8 4 3 
        8 5 4 3 9 6 1 2 7
    </div>
</div>

- 解题思路

这个题目的解法可以通过枚举空白处所有可能的情况，解法相对暴利。由于所有每个位置的解依赖它前面的解，因此这是一个深度搜索的过程。

```cpp
//确定数据结构：
int col[9][10]; //标志位，存放每列1-9出现的标志，1为放置，0为未放置
int row[9][10]; //标志位，存放每行1-9出现的标志，1为放置，0为未放置
int block[9][10]; //标志位，存放每个小块1-9出现的标志，1为放置，0为未放置
int board[9][9]; //棋盘
struct Value{
    int row;
    int col;
};//棋盘中的每个点
vector<Value> blanks; //待填充的空白数字

...

//可放置数字的条件
bool can_be_placed(int r, int c, int num){
    if( row[r][num] == 0 &&
        col[c][num] == 0 &&
        block[block_index(r,c)][num] == 0){
        return true;
    }
    return false;
}

//深度搜索过程
bool DFS(int index){
    if(index < 0){
        return true;
    }
    int row = blanks[index].row;
    int col = blanks[index].col;
    for(int num=1;num<=9;num++){
        //枚举num，如果可以被放置
        if(can_be_placed(row, col, num)){
            //填充板子上的值
            board[row][col] = num;
            //设置状态
            set_state(row,col,num);
            //继续递归
            if(DFS(index-1)){
                return true;
            }else{
                //递归失败，回溯清空状态
                clear_state(row,col,num);
            }
        }
    }
    return false;
}

```




```
struct Obj{
    bool is_visited
    Obj(x);
};
void DFS(v){
    stack<Obj> stk;
    Obj root = Obj(v)
    while(!stk.empty()){
        Obj o = stk.top();
        if(o.is_visited){
            stk.pop();
        }else{
            o.is_visited == true;
            if (some_condition){
                Obj o' = Obj(v')
                stk.push(o')
            }
        }
    }
}
```

### Resources

- [CS106B-Stanford-YouTube](https://www.youtube.com/watch?v=NcZ2cu7gc-A&list=PLnfg8b9vdpLn9exZweTJx44CII1bYczuk)
- [Algorithms-Stanford-Cousera](https://www.coursera.org/learn/algorithms-divide-conquer/home/welcome)
- [算法与数据结构-1-北大-Cousera](https://www.coursera.org/learn/shuju-jiegou-suanfa/home/welcome)
- [算法与数据结构-2-北大-Cousera](https://www.coursera.org/learn/gaoji-shuju-jiegou/home/welcome)
- [算法与数据结构-1-清华-EDX](https://courses.edx.org/courses/course-v1:TsinghuaX+30240184.1x+3T2017/course/)
- [算法与数据结构-2-清华-EDX](https://courses.edx.org/courses/course-v1:PekingX+04833050X+1T2016/course/)
- [算法设计与分析-1-北大-Cousera](https://www.coursera.org/learn/algorithms/home/welcome)
- [算法设计与分析-2-北大-EDX](https://courses.edx.org/courses/course-v1:PekingX+04833050X+1T2016/course/)