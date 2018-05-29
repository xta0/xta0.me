---
layout: post
title: Data Structure Part 7 | Graph
mathjax: true
---

## 图

### 术语

- 图的定义

数学意义的图包含两个元素 $G=(V,E)$，顶点集合 $n = \| V \|$，边集合 $ e= \| E \| $。 假设一个图有三个顶点，彼此连通，则$V=\\{v_0,v_1,v_2\\}$，$E=\\{(v_1,v_2),(v_1,v_3),(v_2,v_3)\\}$。

<img src="/assets/images/2008/08/graph-0.png" style="margin-left:auto; margin-right:auto;display:block">

- 邻接关系/关联关系

可以被边相连的两个点成为**邻接关系**(adjacency)，邻接关系是顶点与顶点之间的关系，顶点与某条边之间的关系称为**关联关系**(incidence)

- 路径

**路径**为一系列的顶点按照依次邻接的关系组成的序列，*$\pi = <v_0,v_1...,v_k>$，长度$\|\pi\|=k$。如上图中$V$到$Z$的一条路径为${b,h}$或${V,X,Z}$。路径的**长度**为顶点的个数或者边的个数。如果再一条通路中不含重复节点，我们称之为 **简单路径** ($v_i = v_j$除非$i=j$)。

- 连通/强连通

有向图 $G (V,E)$，如果两个顶点 $v_i,v_j$ 间有一条从$v_i$ 到 $v_j$ 的有向路径，同时还有一条从 $v_j$ 到 $v_i$ 的有向路径，则称两个顶点**强连通**,非强连通图有向图的极大强连通子图，称为**强连通分量**。

- 环路

当路径的起点和终点重合时，称之为**环路**($v_0=v_k$)。如果再有向图中不包含任何环路，则称之为**有向无环图**(DAG,Directed Acyclic Graph)，树和森林是DAG图的一种特例。 对于只有两个顶点的图，如果是无向图，则不认为是环路；如果是有向图，且两个顶点之间有两条边，则认为是环路，例如$<v_0,v_1>$和$<v_1,v_0>$构成环

- 无向图/有向图

若邻接顶点$u$和$v$的次序无所谓，则$(u,v)$为无向边(undirected edge)，若图中的所有边均为无向边，则这个图称为**无向图**。反之，**有向图**(digraph)中均为有向边(directed edge)，$u,v$分别称作边$(u,v)$的尾，头，表示从$u$出发，到达$v$。

<img src="/assets/images/2008/08/graph-5.png" style="margin-left:auto; margin-right:auto;display:block">

之前介绍的数据结构都可以看成是图的一种表现形式，比如二叉树是一种无权的有向无环图，节点的入度为1，出度最大为2，顶点之间只有一条路径。而单项链表也可以看成是一种无权的DAG，每个节点的入度出度都为1


### 图的接口ADT

图结构的接口主要包括顶点操作，边操作，遍历算法等

```cpp
//定义顶点
class Vertex{
    enum{
        VISITED,
        UNVISTED
    }STATUS;
    
    STATUS status; //顶点状态
    string name; //顶点数据
    int inDegree;
    int outDegree;
    double cost;
    Set<Edge* > edges;
    Vertex* previous;
};

//定义边
class Edge{
    Vertex *start;
    Vertex *end;
    double weight;
};

class Graph{ // 图的ADT
public:
    //顶点操作
    void addVertex(string name); //添加顶点
    void removeVertex(string name); //删除某个顶点
    bool containsVertex(string name); //图中是否包含某个顶点
    set<Vertex getVertexSet(); //返回顶点集合
    Vertex firstNbr(Vertex v); //当前节点的第一个邻居
    Vertex nextNbr(Vertex v, Vertex u); //节点u的下一个邻居
    set<Vertex> getNbrs(Vertex v); //返回某个点所有的邻居
    Edge firstEdge(Vertex oneVertex); // 某顶点的第一条关联边

    //边操作
    void addEdge(Vertex v1, Vertex v2); //添加边
    bool removeEdge(Vertex fromVertex,Vertex toVertex); // 删边
    set<Edge> getEdgeSet(); //返回所有边集合
    set<Edge> getEdgeSet(Vertex v); //返回某个点对应的边集合
    bool setEdge(Vertex fromVertex,Vertex toVertex,int weight); // 添一条边
    bool IsEdge(Edge oneEdge); // 判断oneEdge是否
    Edge nextEdge(Edge preEdge); //下一条兄弟边
    Vertex FromVertex(Edge oneEdge); // 返回边的始点
    Vertex ToVertex(Edge oneEdge); // 返回边的终点
    int Weight(Edge oneEdge); // 返回边的权
    
    //其它操作
    int VerticesNum() const; // 返回图的顶点个数
    int EdgesNum() const; // 返回图的边数
    clear(); //清空所有顶点，所有边
    isEmpty(); //图中是否有顶点
};
```

例如我们要添加两个点点

### 邻接矩阵/关联矩阵

上面我们已经了解了关于图的逻辑模型，但是在计算机中该如何表示这个模型呢？方法有很多种，这里我们主要介绍两种，分别是邻接矩阵表示法和邻接表表示法，顾名思义，前者使用矩阵（二维数组）作为数据结构，后者使用链表。

所谓邻接矩阵就是描述顶点之间链接关系的矩阵。设$G=<V,E>$是一个有$n$个顶点图，则邻接矩阵是一个$n \times n$的方阵，用二维数组`A[n,n]`表示，它的定义如下:

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
GraphMatrix:public Graph{
private:
    vector<Vertex> V; //顶点
    vector<vector<Edge<Te>*>> E; //边集合，邻接矩阵
};
```

- 顶点操作

使用邻接矩阵表示法，对于任意顶点$i$，如何枚举其所有的邻接顶点(neighboor)。

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


### 图的搜索与遍历

在图中搜索两点间的路径有很多种方式，常用的有DFS，BFS，Dijkstra，A*等，对于图的遍历，和树类似我们也可以使用DFS和BFS两种方式，但是图有两个树没有的问题：

1. 连通的问题，从一个点出发不一定能够到达所有点，比如非连通图
2. 可能存在回路，因此遍历可能进入死循环

解决这两个问题，需要给顶点加一个状态位，标识该节点是否已经被访问过。另外，对图的遍历，可将其按照一定规则转化为对树的遍历

```cpp
void graph_traverse(){

// 对图所有顶点的标志位进行初始化
for(int i=0; i<VerticesNum(); i++)
    status(V[i]) = UNVISITED;
    // do_traverse函数用深度优先或者广度优先
    do_traverse(v);
}
```

- **DFS**

图的DFS遍历过程和之前介绍的树的DFS遍历过程类似，都是从一个节点开始，不断的递归+回溯，最终走完全部路径。其基本步骤为

1. 选取一个未访问的点$v_0$作为源点,访问顶点$v_0$
2. 若$v_0$有未被访问的邻居，任选其中一个顶点$u_0$，进行递归地深搜遍历；否则，返回
3. 顶点$u_0$，重复上述过程
4. 不断递归+回溯，直至所有节点都被访问


```cpp
void DFS(Vertex v) { // 深度优先搜索的递归实现
    status(v) = VISITED;  // 把标记位设置为 VISITED
    Visit(v); // 访问顶点v
    //访问所有UNVISITED状态的节点
    for(auto u : getNbrs(v))
        if (status(u) == UNVISITED){
            DFS(u);
        }
    } 
}
```

<img src="/assets/images/2008/08/graph-4.png" style="margin-left:auto; margin-right:auto;display:block">

例如上图DFS的遍历次序为：`a b c f d e g`。这里有一点要注意，由于第二步对相邻节的未访问的节点选取规则不唯一（下图例子使用的是字母顺序），因此对全图进行遍历得到结果序列是不唯一的。类似的，如果使用DFS进行搜索，寻找两点间路径，得到的结果不一定是最短路径。

```
//使用DFS进行搜索
function dfs(v1,v2):
    v1.status = visited
    if v1 == v2:
        //found a path
        return true
    //遍历v的每一个未被访问的相邻节点
    for n : v1.unvisited_neighbors:
        if dfs(n,v2): 
        //found a path
    return false
```

- **BFS**

图的广度优先遍历过程类似从某个点出发，一圈一圈的向外层顶点扩散的过程

1. 从图中的某个顶点$v_0$出发，访问并标记该节点
2. 依次访问$v_0$所有尚未访问的邻接顶点
3. 依次访问这些邻接顶点的邻接顶点，如此反复
4. 直到所有点都被访问过

```cpp
//从v点开始便利啊
void BFS(Vertex v) {
    status(v) = VISITED; 
    queue<Vertex> Q; // 使用STL中的队列
    Q.push(v); // 标记,并入队列
    while (!Q.empty()) { // 如果队列非空
        Vertex v = Q.front (); // 获得队列顶部元素
        Q.pop(); // 队列顶部元素出队
        Visit(u);//访问节点
        //遍历v的每一个未被访问的相邻节点
        for(auto u : getNbrs(v)){
            if (status(u) == UNVISITED){
                status(u) = VISITED; 
                u.previous = v;//设置前驱节点
                Q.push(u);
            }
        }
    }
}
```

<img src="/assets/images/2008/08/graph-6.png" style="margin-left:auto; margin-right:auto;display:block">

以上图为例，假设我们从`s`开始遍历：

1. 随机选取一个`s`节点，入队
2. `s`出队并做标记,将`s`的相邻节点`a,c,d`，入队
3. `a`出队并做标记,将`a`相邻节点入队，由于`s,c`已经是被标记，于是只有`e`入队
4. 同理，`c`出队，`b`入队，以此类推
5. 直至节点全部被访问

对于无权图来说，BFS相比DFS可以用来寻找两点间（例如上图中的a和b）的最短路径（最少边），但却不容易保存到达b点的路径，解决这个问题，可以给每个节点加一个指向前驱节点的指针。

```
//使用BFS进行搜索
function bfs(v1,v2):
    queue := {v1}
    v1.status = visited
    while not queue.empty():
        v = queue.front
        if v == v2 :
            //path is found
        //遍历v的每一个未被访问的相邻节点
        for n : v1.unvisited_neighbors:
            n.status = visited
            queue.push(n)

    //if we get here, no path exists.
```

- 时间复杂度分析：

    DFS 和 BFS 每个顶点访问一次，对每一条边处理一次 (无向图的每条边从两个方向处理)
    1. 采用邻接表表示时，有向图总代价为 $\Theta(n + e)$，无向图为 $\Theta(n + 2e)$
    2. 采用相邻矩阵表示时，理论上，处理所有的边需要 $\Theta(n^2)$的时间 ，所以总代价为$\Theta(n + n^2) = \Theta(n^2)$。但实际上，在执行`nextNbr(v,u)`时，可认为是常数时间，因此它的时间复杂度也可以近似为$\Theta(n + e)$

### 最短路径问题（Dijkstra）

路径的权值在某些场合下是非常重要的，比如两地间飞机的票价，两个网络节点间数据传输的延迟等等。DFS和BFS在搜索两个节点路径时不会考虑边的权值问题，如果加入权值，那么两点间权值最小的路径不一定是BFS得到的最短路径，如下图中求$\\{a,f\\}$两点间的BFS的结果为$\\{a,e,f\\}$，cost为9，而cost最少的路径为$\\{a,d,g,h,f\\}$，其值为6

<img src="/assets/images/2008/08/graph-9.jpg" style="margin-left:auto; margin-right:auto;display:block">

Dijkstra算法研究的是单源最短路径(single-source shortest paths)问题，即给定带权图 $G = <V，E>$，其中每条边 $(v_i，v_j)$ 上的权 $W[v_i，v_j]$ 是一个**非负实数**。计算从任给的一个源点$s$到所有其他各结点的最短路径。其基本思想是维护一张表，表中记录当前两点间的最短路径，然后不断更新路径值，直到找到最终解。

```
function dijkstra(v1,v2):
    //初始化所有节点的cost值
    for v: all vertexes:
        v.cost = maximum

    v1.cost = 0
    //创建一个最小堆保存顶点，优先级最低的顶点在堆顶部
    priority_queue pq = {v1}

    while not pq.empty():
        v = pq.front(); //弹出优先级最低的vertex
        v.status = visited
        if v == v2:
            break;
        //遍历v的每一个未被访问的相邻节点
         for n : v.unvisited_neighbors:
            n.status = visited
            //计算到达n的cost
            cost := v.cost + weight of edge(v,n)
            if cost < n.cost:
                n.cost = cost
                n.prev = v
                pq.push(n)

    //使用v2的前驱指针重建到v1的路径
```

上述是Dijkstra算法的伪码，我们通过下面一个例子看看它是如何工作的。如下图所示，假设我们要求从$\\{a,f\\}$的权值最短路径。

1. 初始化各节点的cost为无穷大，令`a`的cost为0，放入优先级队列pq
2. 从pq中取出顶部节点，访问它的相邻节点`b,d`，计算到达`b,d`的cost，分别为`2,1`，由于`2,1`均小于`b,d`原来的cost(无穷大)，因此将`b,d`的cost更新，放入到优先队列，第一次选择（循环）结束，此时队列中的顶点为`pqueue = {d:1,b:2}`
3. 重复步骤2，pq顶部的节点为`d`,找到`d`相邻的节点`c,f,g,e`分别计算各自的权重为`3,9,5,3`，均小于各自cost值（无穷大），因此`c,f,g,e`入队，第二次循环结束，此时队列中的顶点为`pqueue = {b:2,c:3,e:3,g:5,f:9}`
4. 重复步骤2，pq顶部的节点为`b`，找到`d`相邻的节点`e`(`d`在上一步中已经被访问了)，计算`b`到`e`的cost为2+10=12，大于`e`在上一步得到的cost`3`，因此直接返回。第三次循环结束，此时队列中的顶点为`pqueue = {c:3,e:3,g:5,f:9}`
5. 重复步骤2，pq顶部的节点为`c`,找到`d`相邻的节点`f`，计算到达`f`的权重为`8`，小于队列中的`9`，说明该条路径优于第三步产生的路径，于是更新`f`的cost为`8`，更新`f`的前驱节点为`c`。第四次循环结束，此时队列中的顶点为`pqueue = {e:3,g:5,f:8}`
6. 重复步骤2，pq顶部的节点为`e`，找到`e`相邻的节点`g`，计算`e`到`g`的cost为`3+6=9`，大于`g`在之前得到的cost`5`，因此直接返回。第五次循环结束，此时队列中的顶点为`pqueue = {g:5,f:8}`
7. 重复步骤2，pq顶部的节点为`g`，找到`g`相邻的节点`f`，计算`g`到`f`的cost为`5+1=6`，小于`f`在之前得到的cost`8`，说明从`g`到达`f`这条路径更优，于是更新`f`的cost为`6`，更新`f`的前驱节点为`g`，此次循环结束，此时队列中的顶点为`pqueue = {f:8}`
8. 重复步骤2，发现已经到达节点`f`因此整个循环结束。然后从`f`开始根据前驱节点依次回溯，得到路径`f<-g<-d<-a`，为权值最优路径。

<img src="/assets/images/2008/08/graph-10.jpg" style="margin-left:auto; margin-right:auto;display:block">


### 拓扑排序

所谓拓扑排序，是一种对<mark>有向无环图</mark>顶点排序的方式，排序的规则为两个顶点之间的前后位置，比如从顶点$v_i$到顶点$v_j$有一条有向边$(v_i, v_j)$，那么我们认为$v_i$在$v_j$的前面。

生活中有很多拓扑排序的场景，比如任务之间的依赖关系，学生的选课等，以选课为例，假如我们有如下课程，它们之间的依赖关系为：

<div style=" content:''; display: table; clear:both; height=0">
<div style="float:left">
    <table>
    <thead>
        <tr><th>课程代号</th><th>课程名称</th><th>先修课程</th>
        </tr>
    </thead>
    <tbody>
        <tr><td>C1</td><td>高等数学</td><td></td></tr>
        <tr><td>C2</td><td>程序设计</td><td></td></tr>
        <tr><td>C3</td><td>离散数学</td><td>C1，C2</td></tr>
        <tr><td>C4</td><td>数据结构</td><td>C2，C3</td></tr>
        <tr><td>C5</td><td>算法分析</td><td>C2</td></tr>
        <tr><td>C6</td><td>编译技术</td><td>C4,C5</td></tr>
        <tr><td>C7</td><td>操作系统</td><td>C4,C9</td></tr>
        <tr><td>C8</td><td>普通物理</td><td>C1</td></tr>
        <tr><td>C9</td><td>计算机原理</td><td>C8</td></tr>
    </tbody>
    </table>
</div>
<div style="float:left;margin-left:10px;">
    <img src="/assets/images/2008/08/graph-7.png" width="500px" />
</div>
</div>

如上图所示，如果我们要修完图表中的课程，我们该按照怎样的顺序选课，才能保证课程能够正常修完呢(在修某一门课程前，要先修完它的先序课程)？我们可以首先按照节点间的先决条件（课程之间的关联关系）来构建一个有向无环图图，如右边图(a)所示，然后将图(a)等价的转化为图(b)，显然，图(b)就是一份可行的选课顺序。

实际上，许多应用问题，都可转化和描述为这一标准形式：给定描述某一实际应用的有向图（图(b)），如何在与该图“相容”的前提下，将所有顶点排成一个线性序列（图(c)）。此处的“相容”，准确的含义是：每一顶点都不会通过边，指向其在此序列中的前驱顶点。这样的一个线性序列，称作原有向图的一个拓扑排序（topological sorting）。

对于有向无环图，它的拓扑序列必然存在，但是却不一定唯一，如上面的例子，$C_1$,$C_2$互换后，仍然是一个拓扑排序。我们可以一种BFS遍历的方式来得到这样一组拓扑序列，方法如下

1. 从图中选择任意一个入度为0的顶点且输出
2. 从图中删掉此顶点及其所有的出边，则其所有相邻节点入度减少1
3. 回到第 1 步继续执行

```cpp
void TopsortbyQueue(Graph& G) {
    for (int i = 0; i < G.VerticesNum(); i++)
        G.status(G.V[i]) = UNVISITED; // 初始化
        queue<Vertex<int>> Q; // 使用STL中的队列
        for (i = 0; i < G.VerticesNum(); i++){ // 入度为0的顶点入队
            if (G.V[i].indegree == 0) {
                Q.push(i);
            }
        }
        while (!Q.empty()) { // 如果队列非空
            Vertex<int> v = Q.front(); 
            Q.pop(); // 获得队列顶部元素， 出队
            Visit(G,v); 
            G.status(v) = VISITED; // 将标记位设置为VISITED
            for (Edge e = G.FirstEdge(v); G.IsEdge(e); e = G.NextEdge(e)) {
                Vertex<int> v = G.toVertex(e);
                v.indegree--; // 相邻的顶点入度减1    
            if (v.indegree == 0){ // 顶点入度减为0则入队
                Q.push(v);
            } 
        }
        for (i = 0; i < G.VerticesNum(); i++){ // 判断图中是否有环
            if (G.status(V[i]) == UNVISITED) {
                cout<<“ 此图有环！”; break;
            }
        }
}
```




### 最小生成树


### Resources 

- [算法与数据结构-北大MOOC](https://www.coursera.org/learn/shuju-jiegou-suanfa/lecture/6Kuta/tu-de-bian-li)
- [算法与数据结构-清华-邓俊峰]()
- [拓扑排序](https://zh.wikipedia.org/wiki/%E6%8B%93%E6%92%B2%E6%8E%92%E5%BA%8F)