---
layout: post
title: Data Structure Part 7 | Graph
mathjax: true
---

## 图

### 术语

- 图的定义

数学意义的图包含两个元素 $G=(V,E)$，顶点集合 $n = \| V \|$，边集合 $ e= \| E \| $。 假设一个图有三个顶点，彼此连通，则$V=\\{v_0,v_1,v_2\\}$，$E=\\{(v_1,v_2),(v_1,v_3),(v2_v3)\\}$。

- 邻接关系/关联关系

可以被边相连的两个点成为**邻接关系**(adjacency)，邻接关系是顶点与顶点之间的关系，顶点与某条边之间的关系称为**关联关系**(incidence)

- 无向图/有向图

若邻接顶点$u$和$v$的次序无所谓，则$(u,v)$为无向边(undirected edge)，若图中的所有边均为无向边，则这个图称为**无向图**。反之，**有向图**(digraph)中均为有向边(directed edge)，$u,v$分别称作边$(u,v)$的尾，头，表示从$u$出发，到达$v$

- 路径/环路
    - **路径**为一系列的顶点按照依次邻接的关系组成的序列，*$\pi = <v_0,v_1...,v_k>$，长度$\|\pi\|=k$，如果再一条通路中不含重复节点，我们称之为 **简单路径** ($v_i = v_j$除非$i=j$)。当路径的起点和终点重合时，称之为**环路**($v_0=v_k$)。如果再有向图中不包含任何环路，则称之为**有向无环图**(DAG)，树和森林是DAG图的一种特例。
    
    - 对于两个顶点的情况
        - 如果是无向图，则不认为是环路
        - 如果是有向图，且两个顶点之间有两条边，则认为是环路，例如$<v_0,v_1>$和$<v_1,v_0>$构成环

### 图的接口ADT

图结构的接口主要包括顶点操作，边操作，遍历算法等

```cpp
class Graph{ // 图的ADT
public:
    int VerticesNum(); // 返回图的顶点个数
    int EdgesNum(); // 返回图的边数
    Edge FirstEdge(int oneVertex); // 第一条关联边
    Edge NextEdge(Edge preEdge); //下一条兄弟边
    bool setEdge(int fromVertex,int toVertex,int weight); // 添一条边
    bool delEdge(int fromVertex,int toVertex); // 删边
    bool IsEdge(Edge oneEdge); // 判断oneEdge是否
    int FromVertex(Edge oneEdge); // 返回边的始点
    int ToVertex(Edge oneEdge); // 返回边的终点
    int Weight(Edge oneEdge); // 返回边的权
};
```

### 邻接矩阵/关联矩阵

在计算机中我们可以使用邻接矩阵来描述图，所谓邻接矩阵就是描述顶点之间链接关系的矩阵。设$G=<V,E>$是一个有$n$个顶点图，则邻接矩阵是一个$n \times n$的方阵，用二维数组`A[n,n]`表示，它的定义如下:

$$ 
A[i,j]=
\begin{cases}\
1, \qquad  若(v_i, v_j)∈ E 或<v_i, v_j> ∉ E \\
0, \qquad  若(v_i, v_j)∈ E 或<v_i, v_j> ∉ E \\
\end{cases}
$$

如果顶点$i,j$相连，对于无向图，则$A[i,j]$和$A[j,i]$的值相同；对于有向图，则分别对应各自的$A[i,j]$和$A[j,i]$的值；如果是带权图，则矩阵中元素的值为权值$w$。

<img src="/assets/images/2008/08/graph-2.png" style="margin-left:auto; margin-right:auto;display:block">

可见，对于一个$n$个顶点的图，邻接矩阵是一个对称阵（默认不考虑自环的情况，因此对角线的元素值为0)，空间代价为$O(n^2)$。

基于邻接矩阵的图结构，可以用二维数组来表达：

```cpp
template <class Te>
class Edge{
    int from=1, to=-1, weight=0;
};

template<class Tv, Class Te>
GraphMatrix:public Graph<Tv, Te>{
private:
    int n = 0; //顶点数
    int e = 0;  //边数
    vector<Vertex<Tv>> V; //顶点
    vector<vector<Edge<Te>*>> E; //边集合，邻接矩阵
public:
    Tv firstNbr(Tv v); //当前节点的第一个邻居
    Tv nextNbr(Tv v, Tv u); //前节点u的下一个邻居
};
```

- 顶点操作

对于任意顶点$i$，如何枚举其所有的邻接顶点(neighboor)。

```cpp
int nextNbr(int i, int j){ //若已经枚举邻居j，则转向下一个邻居
    while( -1 > j ){
        if(exists(i,j)){
            return j;
        }
        j--;
    }
}
```

假设$i$在邻接矩阵中的向量为`011001`,可以让$j$从末尾开始向前找，遇到1则返回

```
i 0 1 1 0 0 1
            j
```

- 边操作

```cpp
//插入一条边
void insert(Te const& edge, int w, int i, int j){ //插入(i,j,w)
    if(exists(i,j)){ //点i，j之间已经有一条边了
        return; 
    }
    E[i][j] = new Edge<Te>(edge, w); //创建新边
    e++; //更新计数
    V[i].outDegree++; //i节点的出度+1
    V[j].inDegree++;  //j节点的入读+1
}
```

- 邻接矩阵的优缺点
    - 优点
        - 直观，易于理解和实现
        - 适用范围广，包括有向图，无向图，带权图，自环图等等，尤其适用于稠密图
        - 判断两点之间是否存在联边: $O(1)$
        - 获取顶点的出度入度: $O(1)$
            - 添加删除边后更新度: $O(1)$
        - 扩展性强
    - 缺点
        - 空间复杂度为$\Theta(n^2)$，与边数无关
            

### 邻接表

对于任何一个矩阵，我们以定义它的稀疏因子：在$m \times n$的矩阵中，有$t$个非零单元，则稀疏因子$\delta$为

$$
\delta = \frac{t}{m \times n}
$$

若$\delta < 0.05$，可认为是稀疏矩阵。对于稀疏图，边较少，邻接矩阵会出现大量的零元素，耗费大量的存储空间和时间，因此可以采用邻接表存储法。

邻接表(adjacency list)采用的是**链式存储结构**，它为顶点和边各自定义了一个结构，对于顶点包含两部分：

1. 存放该节点$v_i$的数据域
2. 指向第一条边对象的指针

对于边结构包含三部分:

1. 与顶点$v_i$临街的另一顶点的序号
2. 顶点$v_i$下一条边的指针
3. 权重值(可选)


<img src="/assets/images/2008/08/graph-3.png" style="margin-left:auto; margin-right:auto;display:block">

上图分别为无向图，带权图和有向图的邻接表，对于有向图，有出度和入度两种邻接表，这里只给出了出度的邻接表，对于入度的情况类似。

- 邻接表的空间代价
    - n 个顶点 e 条边的无向图需用 (n + 2e) 个存储单元
    - n 个顶点 e 条边的有向图需用 (n + e) 个存储单元
    - 当边数 e 很小时，可以节省大量的存储空间
    - 边表中表目顺序往往按照顶点编号从小到大排列


### 图的遍历

图的遍历需要考虑两方面的问题：
1. 由于图有连通的问题，从一个点出发不一定能够到达所有点，比如非连通图
2. 由于图可能存在回路，因此遍历可能进入死循环

解决这两个问题，需要给顶点加一个状态位，标识该节点是否已经被访问过。另外，对图的遍历，可将其按照一定规则转化为对树的遍历

```cpp
void graph_traverse(Graph& G) {
// 对图所有顶点的标志位进行初始化
for(int i=0; i<G.VerticesNum(); i++)
    G.Mark[i] = UNVISITED;
// 检查图的所有顶点是否被标记过，如果未被标记，
// 则从该未被标记的顶点开始继续遍历
// do_traverse函数用深度优先或者广度优先
for(int i=0; i<G.VerticesNum(); i++)
    if(G.Mark[i] == UNVISITED)
    do_traverse(G, i);
}
```

- DFS

图的深搜类似树的先根遍历，基本步骤为

1. 选取一个未访问的点 v0 作为源点
2. 访问顶点 v0
3. 递归地深搜遍历 v0 邻接到的其他顶点
4. 重复上述过程直至从 v0 有路径可达的顶点都已被访问过
5. 再选取其他未访问顶点作为源点做深搜，直到图的所有顶点都被访问过

例如下图深度搜索的遍历次序为：`a b c f d e g`

<img src="/assets/images/2008/08/graph-4.png" style="margin-left:auto; margin-right:auto;display:block">

```cpp
void DFS(Graph& G, int v) { // 深度优先搜索的递归实现
    G.Mark[v] = VISITED; // 把标记位设置为 VISITED
    Visit(G,v); // 访问顶点v
    for (Edge e = G.FirstEdge(v); G.IsEdge(e);e = G.NextEdge(e))
        if (G.Mark[G.ToVertex(e)] == UNVISITED)
            DFS(G, G.ToVertex(e));
    PostVisit(G,v); // 对顶点v的后访问
}
```

- BFS

对图的广度优先遍历可转化为对树的层次遍历，的过程为

1. 从图中的某个顶点$v_0$出发，访问并标记该节点
2. 依次访问$v_0$所有尚未访问的邻接顶点
3. 依次访问这些邻接顶点的邻接顶点，如此反复
4. 直到所有点都被访问过

```cpp
void BFS(int v) {
    queue<int> Q; // 使用STL中的队列
    Visit(v); // 访问顶点v
    status(v) = VISITED;  //更新节点状态
    Q.push(v); // 标记,并入队列
    while (!Q.empty()) { // 如果队列非空
        int u = Q.front (); // 获得队列顶部元素
        Q.pop(); // 队列顶部元素出队
        Visit(u);//访问节点
        status(u) = VISITED; 
        //通过枚举与当前节点u相连的边，找到所有u的邻居顶点
        //也可以直接枚举u的所有邻居顶点
        //for(int u = firstNbr(v); u>-1; u=nextNbr(v,u)){
        for (Edge e = FirstEdge(u); IsEdge(e); e = NextEdge(e)){ 
            int u = ToVertext(e);
            if(status(u) == UNVISITED){
                Q.push(u);
            }
        }
    }
}
```

- 时间复杂度分析：

DFS 和 BFS 每个顶点访问一次，对每一条边处理一次 (无向图的每条边从两个方向处理)

1. 采用邻接表表示时，有向图总代价为 $\Theta(n + e)$，无向图为 $\Theta(n + 2e)$
2. 采用相邻矩阵表示时，处理所有的边需要 $\Theta(n^2)$的时间 ，所以总代价为$\Theta(n + n^2) = \Theta(n^2)$

### 拓扑排序


### 最短路径问题


### 最小生成树


### Resources 

- [算法与数据结构-北大MOOC](https://www.coursera.org/learn/shuju-jiegou-suanfa/lecture/6Kuta/tu-de-bian-li)
- [算法与数据结构-清华-邓俊峰]()