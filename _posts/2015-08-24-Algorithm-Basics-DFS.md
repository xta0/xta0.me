---
layout: post
title: 深搜与回溯 
list_title: 算法基础 | Algorithms | 深搜与回溯 | DFS & Backtracking
categories: [Algorithms]
mathjax: true
---

## 深搜与回溯

如果一个问题各状态之间的转移关系(解空间)可以用一个图来描述, 则可以使用深度搜索的方法对该问题进行求解。所谓深度搜索，就是以深度优先的原则，不断的进行尝试+回溯，直至找到最终解。所谓回溯(backtracking)是指当搜寻到某条分支后发现不满足条件，进而退回到前一步重新出发，回溯往往伴随着状态的重置。

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

### 排列组合

- **排列问题（Permutation**

Permutation问题是求解一个集合的全排列问题，例如`[1,2,3]`的全排列为`[1,2,3],[1,3,2],[2,1,3],[2,3,1],[3,1,2],[3,2,1]`。这个问题的解法有很多，比较常见的是使用DFS进行暴力枚举，去搜索所有可能的结果。

使用DFS进行搜索的一个先决条件是要构建正确的解空间或者决策树，我们以`[1,2,3]`为例，其解空间如下：

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
class Solution {
private:
    void dfs( vector<int>& nums, vector<int>& chosen, vector<vector<int>>& results){
        if(nums.size() == 0){
            results.push_back(chosen);
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
public:
    vector<vector<int>> permute(vector<int>& nums) {
        vector<vector<int>> ret;
        vector<int> vec;
        dfs(nums,vec,ret);
        return ret;
    };
};
```

接下来我们来分析下上面算法的时间复杂度，从递归函数来看，该

- **组合问题（Combination**

组合问题的解法和排列类似，但是组合的搜索分支没有排列那么多，我们还看上面`[1,2,3]`的例子：


```
     1             2            3
    / \             \           
   2   3             3
  /              
 3              
```
对于`[1,2,3]`的全组合有下面几种情况`[1],[2],[3],[1,2],[1,3],[2,3],[1,2,3]`，组合问题和排列问题有几点不同，如下：

1. 在进行`for`循环遍历时，需要改变每次起始搜索的`index`的值，`index`值是递增的，例如当我们走完`[1,2,3]`后，我们需要回溯到`1`，然后接着走到`3`，此时对于`3`来说，它不能再回头遍历`2`，因为`[1,2,3]`和`[1,3,2]`是相同的组合
2. 由于`index`的递增性使我们在遍历数组的时候无需修改数组，例如第一次搜索我们choose了`1`，第二次搜索由于index递增，可以直接上我们选择到`2`,无需将`1`移除array
3. 对于组合，我们可以求长度，例如长度为2的组合有`[1,2],[1,3],[2,3]`

```python
def dfs(self,arr,index,choose,result):
    #for循环从index位置开始
    for i in range(index,len(arr)):
        #choose
        x = arr[i]
        choose.append(x)
        result.append("".join(choose))
        #dfs
        self.dfs2(curr,depth,arr,i+1,choose,result)
        #unchoose
        choose.pop()

def combination(self,arr,n):
    choose = []
    result = []
    self.dfs(arr,0,choose,result)
    return result
```

上面代码可以搜索出`arr`的所有组合，在`dfs`函数中无需增加递归基，因为由于`index`递增，当`for`循环执行完后，函数自动返回。如果想要求解`k`个数的组合，只需要修改`dfs`函数，追踪递归深度即可:

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
        self.dfs2(curr,depth,arr,i+1,choose,result)
        #unchoose
        curr-=1
        choose.pop()
```

### N皇后问题

另外一个经典的DFS问题是N皇后问题，这个问题是DFS+剪枝的一个典型应用，所谓剪枝是指在DFS搜索过程中，对不必要的路径进行裁剪，从而加快搜索速度。N皇后问题描述如下:

> 给你个NxN的棋盘，在棋盘上摆放N个皇后，使得这N个皇后无法相互攻击（皇后可以横竖攻击，对角线攻击），如下图中是一个8皇后问题的一个解，请给出满足N皇后条件的所有摆法

<img src="{{site.baseurl}}/assets/images/2015/08/8-queens.png">

解这个问题的思路就是用DFS不断的递归+回溯，穷举所有的可能的情况。其算法步骤如下：

1. 由于每个行或者列只能放置一个皇后，因此DFS可以按行搜索，在每行中不断尝试每个列的位置
2. 当放置一个皇后后，在棋盘上将该位置以及其可能攻击的位置均置为不可用
3. 搜索到不可用的位置时，直接跳过，进行剪枝操作
4. 算法时间复杂度为指数级

```cpp
//表示盘中的每个点
struct PT{
    int i;
    int j;
    bool operator<(const PT& pt) const{
        if( i<pt.i){
            return true;
        }else if(i==pt.i){
            return j<pt.j;
        }else{
            return false;
        }
    }
};
class Solution {
    //放置一个皇后，更新棋盘状态
    set<PT> place(int i, int j, vector<vector<char>>& board){
        set<PT> us;
        //竖向
        for(int k=0;k<board.size();k++){
            if(board[k][j] !='.' && board[k][j] != 'Q'){
                board[k][j] = '.';
                us.insert({k,j});
            }
        }
        //横向
        for(int k=0;k<board.size();k++){
            if(board[i][k] != '.' && board[i][k] != 'Q'){
                board[i][k]= '.';
                us.insert({i,k});
            }
            
        }
        //对角线1
        for(int p=i, q=j; p>=0 && q>=0 ; p--,q--){
            if(board[p][q] != '.' && board[p][q] != 'Q'){
                board[p][q] = '.';
                us.insert({p,q});
            }
        }
        for(int p=i,q=j; p<board.size()&&q<board.size();p++,q++){
            if(board[p][q] != '.' && board[p][q] != 'Q'){
                board[p][q] = '.';
                us.insert({p,q});
            }
        }
        //对角线2
        for(int p=i,q=j; p>=0 && q<board.size();p--,q++){
            if(board[p][q] != '.' && board[p][q] != 'Q'){
                board[p][q] = '.';
                us.insert({p,q});
            }
        }
        for(int p=i,q=j; p<board.size() && q>=0;p++,q--){
            if(board[p][q] != '.' && board[p][q] != 'Q'){
                board[p][q] = '.';
                us.insert({p,q});
            }
        }
        board[i][j] = 'Q';
        us.insert({i,j});
        return us;
    }
    //回溯棋子后复原棋盘状态
    void unplace(set<PT>& us, vector<vector<char>>& board){
        for(auto itor = us.begin(); itor!=us.end(); itor++){
            auto p = *itor;
            board[p.i][p.j] = 'x';
        }
    }
    //深搜
    void dfs(int n, int row, int sz, vector<vector<char>>& board, vector<vector<string>>& result){
        if(n == 0){
            vector<string> v;
            for(auto vec:board){
                string tmp="";
                for(auto c:vec){
                    tmp+=c;
                }
                v.push_back(tmp);
            }
            result.push_back(v);
            return;
        }
        //按行搜索，尝试列位置，j代表列
        for(int j=0;j<sz && row<sz;j++){
            if(board[row][j] == 'x'){ //剪枝
                //choose
                auto pts = place(row, j, board);
                cout<<"set state: "<<endl;
                log(board);
                n-=1;
                //深搜
                dfs(n,row+1,sz,board,result);
                //backtrack
                n+=1;
                unplace(pts, board);
                cout<<"reset state: "<<endl;
                log(board);

            }
        }
    }
    //打印棋盘状态
   void log(vector<vector<char>>& board){
        for(auto vec:board){
            for(auto c:vec){
                cout<<c<<" ";
            }
            cout<<endl;
        }
        cout<<"--------"<<endl;
    }
    
public:
    vector<vector<string>> solveNQueens(int n) {
        vector<vector<char>> board(n,vector<char>(n,'x'));
        vector<vector<string>> ans;
        dfs(n,0,n,board,ans);
        return ans;
    }
};
```

### Sudoku问题

我们再来看一道数独问题，和N皇后问题一样，求解数独问题也是一个利用暴力搜索的典型应用。

> 将数字1到9,填入9x9矩阵中的小方格，使得矩阵中的每行，每列，每个3x3的小格子内，9个数字都会出现"。

<div class="md-flex-h md-flex-no-wrap md-margin-bottom-12">
<div><img src="{{site.baseurl}}/assets/images/2015/08/Sudoku-1.jpg"></div>
<div class="md-margin-right-12"><img src="{{site.baseurl}}/assets/images/2015/08/Sudoku-2.jpg"></div>
</div>

解数独问题的思路为在每个空位依次尝试从1-9的每个值，每放置一个后进行DFS搜索，如果所有空位都能填满则返回一组解，如果有空位不满足条件，则进行回朔，重置状态后再换另一个数字尝试DFS，知道尝试完所有的空位。

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

### 关于DFS的更多问题

- [22. Generate Parentheses](https://leetcode.com/problems/generate-parentheses/description/)
- [46. Permutations](https://leetcode.com/problems/permutations/description/)
- [47. Permutations II](https://leetcode.com/problems/permutations-ii/description/)
- [51. N-Queens](https://leetcode.com/problems/n-queens/description/)
- [52. N-Queens II](https://leetcode.com/problems/n-queens-ii/)
- [78. Subsets](https://leetcode.com/problems/subsets/description/)
- [77. Combinations](https://leetcode.com/problems/combinations/description/)
- [40. Combination Sum II](https://leetcode.com/problems/combination-sum-ii/description/)
- [216. Combination Sum III](https://leetcode.com/problems/combination-sum-iii/description/)

### Resources

- [CS106B-Stanford-YouTube](https://www.youtube.com/watch?v=NcZ2cu7gc-A&list=PLnfg8b9vdpLn9exZweTJx44CII1bYczuk)
- [Algorithms-Stanford-Cousera](https://www.coursera.org/learn/algorithms-divide-conquer/home/welcome)
- [算法与数据结构-1-北大-Cousera](https://www.coursera.org/learn/shuju-jiegou-suanfa/home/welcome)
- [算法与数据结构-2-北大-Cousera](https://www.coursera.org/learn/gaoji-shuju-jiegou/home/welcome)
- [算法与数据结构-1-清华-EDX](https://courses.edx.org/courses/course-v1:TsinghuaX+30240184.1x+3T2017/course/)
- [算法与数据结构-2-清华-EDX](https://courses.edx.org/courses/course-v1:PekingX+04833050X+1T2016/course/)
- [算法设计与分析-1-北大-Cousera](https://www.coursera.org/learn/algorithms/home/welcome)
- [算法设计与分析-2-北大-EDX](https://courses.edx.org/courses/course-v1:PekingX+04833050X+1T2016/course/)