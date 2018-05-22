---
layout: post
title: Algorithms-6 | 深度优先搜索 | DFS
---

## 深度搜索

将问题的各状态之间的转移关系描述为一个<mark>图</mark>, 这个图实际上就是该问题的所有解空间，深度搜索就是以深度优先的原则，不断的进行尝试+回溯，直至找到最终解。所谓回溯(back tracking)是指当搜寻到某条分支后发现不满足条件，进而退回到前一步重新出发，回溯往往伴随着状态的重置。深搜的伪码为：

```
dfs(v) {
    if( v 访问过)
        return;
    //1.将v标记为访问过;
    v.is_visited = true
    //2.(不是必须)设置某些状态
    setFlags();
    //3.对和u相邻的每个点v进行递归:
    dfs(v);
    //4.(不是必须)回溯，如果当前搜索不符合条件，重置状态
    setFlags();
}
int main() {
    while(在图中能找到未访问过的点 k)
        dfs(k);
}
```

对于标记位`is_visited`，如果解空间是二叉树或者普通树，则不需要该标记位；如果解空间为有向无环图，某个节点可能存在两条到达路径，则需要通过标志位来避免重复访问。我们也可以使用迭代来代替现递归

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
### Sudoku

求解数独问题是一个典型的DFS搜索问题，数独问题描述如下:
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

### 四皇后问题

历史上一个很经典的问题是八皇后问题，这里为了简化，改为求解四皇后问题，问题如下：

> 在国际象棋的4x4的棋盘上放置4个皇后，使任意两个皇后不在同一行上，不在同一列上，不在同一斜对角线上，问有几种可能的摆法


