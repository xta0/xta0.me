---
layout: post
list_title: 数据结构基础 Data Structure | Graph | 图
title: 图
mathjax: true
categories: [DataStructure]
---

### 术语

- 图的定义

数学意义的图包含两个元素 $G=(V,E)$，顶点集合 $n = \| V \|$，边集合 $ e= \| E \| $。 假设一个图有三个顶点，彼此连通，则$V=\\{v_0,v_1,v_2\\}$，$E=\\{(v_1,v_2),(v_1,v_3),(v_2,v_3)\\}$。

<img src="{{site.baseurl}}/assets/images/2008/08/graph-0.png" style="margin-left:auto; margin-right:auto;display:block">

- 邻接关系/关联关系

可以被边相连的两个点成为**邻接关系**(adjacency)，邻接关系是顶点与顶点之间的关系，顶点与某条边之间的关系称为**关联关系**(incidence)

- 路径

**路径**为一系列的顶点按照依次邻接的关系组成的序列，*$\pi = <v_0,v_1...,v_k>$，长度$\|\pi\|=k$。如上图中$V$到$Z$的一条路径为${b,h}$或${V,X,Z}$。路径的**长度**为顶点的个数或者边的个数。如果再一条通路中不含重复节点，我们称之为 **简单路径** ($v_i = v_j$除非$i=j$)。

- 连通图

若图中任意两点都是连通的，那么该图称为**连通图**。对于有向图 $G (V,E)$，如果两个顶点 $v_i,v_j$ 间有一条从$v_i$ 到 $v_j$ 的有向路径，责成同时还有一条从 $v_j$ 到 $v_i$ 的有向路径，则称两个顶点**强连通**,称$G为**强连通图**。强连通图只有一个连通分量，即其自身。

- 环路

当路径的起点和终点重合时，称之为**环路**($v_0=v_k$)。如果再有向图中不包含任何环路，则称之为**有向无环图**(DAG,Directed Acyclic Graph)，树和森林是DAG图的一种特例。 对于只有两个顶点的图，如果是无向图，则不认为是环路；如果是有向图，且两个顶点之间有两条边，则认为是环路，例如$<v_0,v_1>$和$<v_1,v_0>$构成环

- 无向图/有向图

若邻接顶点$u$和$v$的次序无所谓，则$(u,v)$为无向边(undirected edge)，若图中的所有边均为无向边，则这个图称为**无向图**。反之，**有向图**(digraph)中均为有向边(directed edge)，$u,v$分别称作边$(u,v)$的尾，头，表示从$u$出发，到达$v$。

<img src="{{site.baseurl}}/assets/images/2008/08/graph-5.png" style="margin-left:auto; margin-right:auto;display:block">

之前介绍的数据结构都可以看成是图的一种表现形式，比如二叉树是一种无权的有向无环图，节点的入度为1，出度最大为2，顶点之间只有一条路径。而单项链表也可以看成是一种无权的DAG，每个节点的入度出度都为1


### 图的接口ADT

图结构的接口主要包括顶点操作，边操作，遍历算法等

``` cpp
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

## 图的表示

上面我们已经了解了关于图的逻辑模型和基本接口，但是在计算机中该如何表示这个模型呢？方法有很多种，这里我们主要介绍三种，分别是边表(edge list)，邻接矩阵(adjacency matrix)和邻接表(adjacency list)。

<img src="{{site.baseurl}}/assets/images/2008/08/graph-14.jpg" style="margin-left:auto; margin-right:auto;display:block">

### 边表

边表，顾名思义是使用一个无序链表来表示图，其中链表中的每个节点为一组边的集合。如上图中，我们可以使用如下链表来表示图:

| (1,2) | (1,4) | (1,7) | (2,3) | (2,5) | (3,6) | (4,7) | (5,6) | (5,7) | (6,7) |

对于每个节点的有序对，使用起点和终点来表示方向，值为顶点的内容。因此这种表示方法不会显示的存放顶点，所有顶点信息均包含在边表中。

```python
edge_list = [ [1,2],[1,4],[1,7],...,[6,7]]
```

显然这种方式对顶点操作不是很友好，如果想要找出`4`的邻居节点，则要遍历每个节点，不是很高效。

### 邻接矩阵/关联矩阵

所谓邻接矩阵就是描述顶点之间链接关系的矩阵。设$G=<V,E>$是一个有$n$个顶点图，则邻接矩阵是一个$n \times n$的方阵，用二维数组`A[n,n]`表示，它的定义如下:

$$ 
A[i,j]=
\begin{cases}\
1, \qquad  若(v_i, v_j)∈ E 或<v_i, v_j> ∉ E \\
0, \qquad  若(v_i, v_j)∈ E 或<v_i, v_j> ∉ E \\
\end{cases}
$$

如果顶点$i,j$相连，对于无向图，则$A[i,j]$和$A[j,i]$的值相同；对于有向图，则分别对应各自的$A[i,j]$和$A[j,i]$的值；如果是带权图，则矩阵中元素的值为权值$w$。

<img src="{{site.baseurl}}/assets/images/2008/08/graph-2.png" style="margin-left:auto; margin-right:auto;display:block">

可见，对于一个$n$个顶点的图，邻接矩阵是一个对称阵（默认不考虑自环的情况，因此对角线的元素值为0)，空间代价为$O(n^2)$。

基于邻接矩阵的图结构，可以用二维数组来表达：

``` cpp
GraphMatrix:public Graph{
private:
    vector<Vertex* > V; //顶点
    vector<vector<Edge<Te>*>> matrix; //边集合，邻接矩阵
    //map<Vertex* , vector<Edge* >> matrix; //也可以使用map
    //map<Vertex* , set<Edge* >> matrix; //也可以使用map
};
```
使用邻接矩阵的优点是：
    
1. 直观，易于理解和实现
2. 适用范围广，包括有向图，无向图，带权图，自环图等等，尤其适用于稠密图
3. 判断两点之间是否存在联边: $O(1)$
4. 获取顶点的出度入度: $O(1)$
    - 添加删除边后更新度: $O(1)$
5. 扩展性强

缺点为：
1. 空间复杂度为$\Theta(n^2)$，与边数无关

以左边无向图的邻接矩阵为例，Python表示如下：

```python
adjacency_matrix = [
    #v0,v1, v2,  v3
    [0, 1,  1,  0], #v0
    [1, 0,  1,  1], #v1
    [1, 1,  0,  0], #v2
    [0, 1,  0,  0]  #v3
]
```

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


<img src="{{site.baseurl}}/assets/images/2008/08/graph-3.png" style="margin-left:auto; margin-right:auto;display:block">

上图分别为无向图，带权图和有向图的邻接表，对于有向图，有出度和入度两种邻接表，这里只给出了出度的邻接表，对于入度的情况类似。由上图的结构可以看出使用邻接表，对每个节点的相邻节点的访问是非常高效的，另外，邻接表的空间利用率也相对较高。

以左边无向图的邻接表为例，Python表示如下：

```python
#使用list
adjacency_list = [
    [1,2], #v0
    [0,2,3], #v1
    [0,1], #v2
    [1,1] #v3
]
#也可以使用kev,value对表示
adjacency_map = {
    0: [1,2],
    1: [0,2,3],
    2: [0,1],
    3: [1,1]
}
```

### 几种图的不同实现方式的性能比较

||边表|邻接表|邻接矩阵|
|内存消耗| $V+E$| $V+E$ | $V^2$ |
|判断相邻节点| $E$|$degree(v)$ | V|
|添加顶点| $1$ | $1$ | $V^2$ |
|删除顶点| $E$ | $1$ | $V^2$ |
|添加边| $1$ |$1$ |$1$ |
|删除边| $E$ | $degree(v)$ | 1|



## 图的搜索与遍历

在图中搜索两点间的路径有很多种方式，常用的有DFS，BFS，Dijkstra，A*等，对于图的遍历，和树类似我们也可以使用DFS和BFS两种方式，但是图有两个树没有的问题：

1. 连通的问题，从一个点出发不一定能够到达所有点，比如非连通图
2. 可能存在回路，因此遍历可能进入死循环

解决这两个问题，需要给顶点加一个状态位，标识该节点是否已经被访问过。另外，对图的遍历，可将其按照一定规则转化为对树的遍历

``` cpp
void graph_traverse(){
    // 对图所有顶点的标志位进行初始化
    for(int i=0; i<VerticesNum(); i++)
        status(V[i]) = UNVISITED;
        // do_traverse函数用深度优先或者广度优先
        do_traverse(v);
    }
}
```

### DFS

图的DFS遍历过程和之前介绍的树的DFS遍历过程类似，都是从一个节点开始，不断的递归+回溯，最终走完全部路径。其基本步骤为

1. 选取一个未访问的点$v_0$作为源点,访问顶点$v_0$
2. 若$v_0$有未被访问的邻居，任选其中一个顶点$u_0$，进行递归地深搜遍历；否则，返回
3. 顶点$u_0$，重复上述过程
4. 不断递归+回溯，直至所有节点都被访问


``` cpp
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

<img src="{{site.baseurl}}/assets/images/2008/08/graph-4.png" style="margin-left:auto; margin-right:auto;display:block">

例如上图DFS的遍历次序为：`a b c f d e g`。这里有一点要注意，由于第二步对相邻节的未访问的节点选取规则不唯一（下图例子使用的是字母顺序），因此对全图进行遍历得到结果序列是不唯一的。类似的，如果使用DFS进行搜索，寻找两点间路径，得到的结果不一定是最短路径。

```javascript
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

### BFS

图的广度优先遍历过程类似从某个点出发，一圈一圈的向外层顶点扩散的过程

1. 从图中的某个顶点$v_0$出发，访问并标记该节点
2. 依次访问$v_0$所有尚未访问的邻接顶点
3. 依次访问这些邻接顶点的邻接顶点，如此反复
4. 直到所有点都被访问过

``` cpp
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

<img src="{{site.baseurl}}/assets/images/2008/08/graph-6.png" style="margin-left:auto; margin-right:auto;display:block">

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

### Dijkstra算法

路径的权值在某些场合下是非常重要的，比如两地间飞机的票价，两个网络节点间数据传输的延迟等等。DFS和BFS在搜索两个节点路径时不会考虑边的权值问题，如果加入权值，那么两点间权值最小的路径不一定是BFS得到的最短路径，如下图中求$\\{a,f\\}$两点间的BFS的结果为$\\{a,e,f\\}$，cost为9，而cost最少的路径为$\\{a,d,g,h,f\\}$，其值为6

<img src="{{site.baseurl}}/assets/images/2008/08/graph-9.jpg" style="margin-left:auto; margin-right:auto;display:block">

Dijkstra算法研究的是单源最短路径(single-source shortest paths)问题，即给定带权图 $G = <V，E>$，其中每条边 $(v_i，v_j)$ 上的权 $W[v_i，v_j]$ 是一个**非负实数**。计算从任给的一个源点$s$到所有其他各结点的最短路径。其基本思想是维护一张表，表中记录当前两点间的最短路径，然后不断更新路径值，直到找到最终解。

```javascript
function dijkstra(v1,v2):
    //初始化所有节点的cost值
    for v in all vertexes{
        v.cost = maximum
    }
    v1.cost = 0
    //创建一个最小堆保存顶点，优先级最低的顶点在堆顶部
    priority_queue pq(v1.cost);

    while !(pq.empty()){
        v = pq.front(); //弹出优先级最低的vertex
        v.status = visited
        if v == v2:
            //找到v2
            break;
        //遍历v的每一个未被访问的相邻节点
         for n : v.unvisited_neighbors:
            //计算到达n的cost
            cost := v.cost + weight of edge(v,n)
            if cost < n.cost:
                n.cost = cost
                //记录前驱节点
                n.prev = v
                pq.push( n.cost )
    }
    //使用v2的前驱节点，重建到v1的路径
```

上述是Dijkstra算法的伪码，我们通过下面一个例子看看它是如何工作的。如下图所示，假设我们要求从$\\{a,f\\}$的权值最短路径。

<img src="{{site.baseurl}}/assets/images/2008/08/graph-10.jpg" style="margin-left:auto; margin-right:auto;display:block">


1. 初始化各节点的cost为无穷大，令`a`的cost为0，放入优先级队列pq

2. 从pq中取出顶部节点，访问它的相邻节点`b,d`，计算到达`b,d`的cost，分别为`2,1`，由于`2,1`均小于`b,d`原来的cost(无穷大)，因此将`b,d`的cost更新，放入到优先队列，第一次选择（循环）结束，此时队列中的顶点为`pqueue = {d:1,b:2}`

3. 重复步骤2，pq顶部的节点为`d`,找到`d`相邻的节点`c,f,g,e`分别计算各自的权重为`3,9,5,3`，均小于各自cost值（无穷大），因此`c,f,g,e`入队，第二次循环结束，此时队列中的顶点为`pqueue = {b:2,c:3,e:3,g:5,f:9}`

4. 重复步骤2，pq顶部的节点为`b`，找到`b`相邻的节点`d,e`，由于`d`在上一步中已经被访问了，于是略过`d`，计算`b`到`e`的cost为2+10=12，大于`e`在上一步得到的cost`3`，因此直接返回。第三次循环结束，此时队列中的顶点为`pqueue = {c:3,e:3,g:5,f:9}`

5. 重复步骤2，pq顶部的节点为`c`,找到`d`相邻的节点`f`，计算到达`f`的权重为`8`，小于队列中的`9`，说明该条路径优于第三步产生的路径，于是更新`f`的cost为`8`，更新`f`的前驱节点为`c`。第四次循环结束，此时队列中的顶点为`pqueue = {e:3,g:5,f:8}`

6. 重复步骤2，pq顶部的节点为`e`，找到`e`相邻的节点`g`，计算`e`到`g`的cost为`3+6=9`，大于`g`在之前得到的cost`5`，因此直接返回。第五次循环结束，此时队列中的顶点为`pqueue = {g:5,f:8}`

7. 重复步骤2，pq顶部的节点为`g`，找到`g`相邻的节点`f`，计算`g`到`f`的cost为`5+1=6`，小于`f`在之前得到的cost`8`，说明从`g`到达`f`这条路径更优，于是更新`f`的cost为`6`，更新`f`的前驱节点为`g`，此次循环结束，此时队列中的顶点为`pqueue = {f:8}`

8. 重复步骤2，发现已经到达节点`f`因此整个循环结束。然后从`f`开始根据前驱节点依次回溯，得到路径`f<-g<-d<-a`，为权值最优路径。


从上面的求解过程可以发现Dijkstra算法实际上是一种<mark>贪心算法</mark>，即每一步都找当前最优解（最小堆堆顶元素)。对于贪心法，它实际上是动态规划算法的特例，因此要求每一步的重复子结构解具有无后效性。对应到Dijkstra算法，要求路径的权值不能为负数。因为如果出现负数，当前的最优选择在后面不一定是最优。

- Dijkstra算法时间复杂度

对于稀疏图，Dijkstra算法使用最小堆实现效率较高：

1. 初始化: $O(v)$
2. While循环： $O(v)$
    - remove vertex from pq: $O(\log{V})$
    - potentially perform E updates on cost/previous
    - update costs in pq: $O(\log{V})$
3. 路径重建： $O(E)$
4. 总的时间复杂度为: $O(V\log{V}+E\log{V}) = O(E\log{V})$ (如果图是连通的，有$V=O(E)$)


### A* 算法

A*是另一种寻找权值最优路径的方法，它是对Dijkstra算法的一种改进。Dijkstra虽然可以找到最短路径，但是BFS的寻找过程却不是最高效的，如下图所示

<img src="{{site.baseurl}}/assets/images/2010/08/a-star-2.png" width="60%">

假设我们要从中心点走到最右边的点，由于从中心扩散出去的每个点权值都相同，Dijkstra算法会在四个方向上不断尝试每个扩散出去的点，显然，这种搜法包含大量的无效搜索。仔细思考不难发现，其原因在于Dijkstra算法基于贪心策略每次只能确定当前最短距离，而不知道哪个方向才能真正逼近重点，如下图中，Dijkstra每次只能确定由a节点确定b节点，而对于终点c在哪则毫无所知，没有任何信息：

<img src="{{site.baseurl}}/assets/images/2010/08/a-star-1.png">

针对这个问题，A*算法改进了Dijkstra，引入了一个Heuristic的估计函数来来确定节点的扩散方向，使其可以沿着终点方向逼近，如下图所示

<img src="{{site.baseurl}}/assets/images/2010/08/a-star-3.png">

在引入了一个Heuristic函数后，我们相当于知道了一些额外的信息，这些信息可以帮助我们减少不必要的搜索。假设我们想要找从`a`到`c`的权值最小路径，对于任何中间节点`b`，我们要计算两个值

1. 从`a`到`b`确定的权值(同Dijkstra)
2. 从`b`到`c`的估计值（estimated cost）

A*的整体算法框架同Dijkstra相同，只需要将最小堆中存放的cost改为`cost(n) + H(n,target)`即可

```cpp
v1.cost = H(v1,v2)
priority_queue pq(v1);
//...
pq.push( n.cost + H(n,v2) )
```
A* 算法的难点在于如何找到合适的Heuristic函数，不同的搜索场景，使用的Heuristic也不相同，这里不再继续展开。

### 最小生成树

所谓生成树(Spanning Tree)是连接无环图中所有顶点的边的集合。如下图所示，我们将左边的图去环后得到了右边的无环图，该图即是一棵生成树

<img src="{{site.baseurl}}/assets/images/2008/08/graph-11.jpg" style="margin-left:auto; margin-right:auto;display:block">

所谓最小生成树，是图中所有生成树中权值之和最小的一棵,简称 MST(minimum-cost spanning tree)。

- **Kruskal's algorithm**

Kruskal算法是一种贪心算法，主要步骤如下：

1. 将图$G$中所有边放入最小堆$E$
2. 删除图$G$中的所有边，剩下$n$个顶点，此时图的状态为无边的森林$T=<V,{}>$
3. 在$E$中弹出权值最小边，如果该边的两个顶点在$T$中不连通，则将其加入到$E$中，否则忽略这条边
4. 依次类推，直到$E$为空，此时就得到图$G$的一颗最小生成树

<img src="{{site.baseurl}}/assets/images/2008/08/graph-12.jpg" style="margin-left:auto; margin-right:auto;display:block">

如上图所示，首先将所有边放入优先队列，则权值最小的`a`在堆顶，然后`a`出队，其两个顶点不连通，因此将该边放入图中（标红），当`e`出队的时候，我们发现`e`的两个顶点可以已通过`a,d`连通，因此`e`被忽略。按照此规则，以此类推，最终得到最小生成树（图中红色边）为:`a,b,c,d,f,h,i,k,p`总权值为`1+2+3+4+6+8+9+11+16 = 60`。不难看出，上述规则依旧是贪心法，每次选择权值最小的路径，其伪码如下：

```javascript
function kruskal(graph):
    //创建一个最小堆
    priority_queue pq
    //将图中所有边放最小堆中，则权值cost最小的边在堆顶
    for edge : graph.all_edges:
        pq.push(edge) 
    //此时产生n个顶点，对应n个等价类
    while equal_num > 1: //等价类个数>1,说明还没有形成树
        //循环
        while not pq.empty():
            e = pq.front()
            pq.pop()
            v1 = e.from()
            v2 = e.to()
            //如果v1，v2两点不连通，则把该边放入图中，否则忽略这条边
            if graph.different(v1, v2): //判断v1,v2是否不连通
                graph.union(v1,f2); //合并两个顶点所在的等价类
                //将该条边放入图中
                graph.addEdge(v1,v2)
                //等价类个数-1
                equal_num -= 1
```

Kruskal算法使用了路径压缩（并查集）来合并等价类，`different()` 和 `Union()` 函数几乎是常数。假设可能对几乎所有边都判断过了，则最坏情况下算法时间代价为$\Theta(e\log{e})$，即堆排序的时间,通常情况下只找了略多于 n 次，MST 就已经生成，因此，<mark>时间代价接近于$\Theta(e\log{e})$</mark>


- **Prim's algorithm**

Prim算法和上面算法类似，也是采用贪心的策略，不同的是Prim算法每次取权值最小边对应的顶点，具体如下（代码见附录）：

1. 从图中任意一个顶点开始 (例如A)，首先把这个顶点包括在MST中
2. 然后从图中选一个与A点连通，但不再MST中的顶点B，并且A到B的权值最小的一条边连同B一起加入到MST中
3. 如此进行下去，每次往 MST 里加一个顶点和一条权最小的边，直到把所有的顶点都包括进 MST 里
4. 算法结束时, MST中包含了原图中的n-1条边

<img src="{{site.baseurl}}/assets/images/2008/08/graph-13.jpg" style="margin-left:auto; margin-right:auto;display:block">

Prim 算法非常类似于 Dijkstra 算法，算法中的距离值不需要累积，直接用最小边，而确定代价最小的边就需要总时间$O(n^2)$；取出权最小的顶点后，修改 D 数组共需要时间$O(e)$，因此<mark>共需要花费$O(n^2)$的时间</mark>。Prim算法适合于稠密图，对于稀疏图，可以像 Dijkstra 算法那样用堆来保存距离值。

### 拓扑排序

所谓拓扑排序，是一种对<mark>有向无环图</mark>顶点排序的方式，排序的规则为两个顶点之间的前后位置，比如从顶点$v_i$到顶点$v_j$有一条有向边$(v_i, v_j)$，那么我们认为$v_i$在$v_j$的前面。

生活中有很多拓扑排序的场景，比如任务之间的依赖关系，学生的选课等，以选课为例，假如我们有如下课程，它们之间的依赖关系为：

<div>
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


如下图所示，如果我们要修完图表中的课程，我们该按照怎样的顺序选课，才能保证课程能够正常修完呢(在修某一门课程前，要先修完它的先序课程)？我们可以首先按照节点间的先决条件（课程之间的关联关系）来构建一个有向无环图图，如图(a)所示，然后将图(a)等价的转化为图(b)，显然，图(b)就是一份可行的选课顺序。

<img src="{{site.baseurl}}/assets/images/2008/08/graph-7.png" width="500px" class="md-img-center"/>

实际上，许多应用问题，都可转化和描述为这一标准形式：给定描述某一实际应用的有向图（图(b)），如何在与该图“相容”的前提下，将所有顶点排成一个线性序列（图(c)）。此处的“相容”，准确的含义是：每一顶点都不会通过边，指向其在此序列中的前驱顶点。这样的一个线性序列，称作原有向图的一个拓扑排序（topological sorting）。

对于有向无环图，它的拓扑序列必然存在，但是却不一定唯一，如上面的例子，$C_1$,$C_2$互换后，仍然是一个拓扑排序。我们可以一种<mark>BFS遍历</mark>的方式来得到这样一组拓扑序列，方法如下

1. 从图中选择任意一个入度为0的顶点且输出
2. 从图中删掉此顶点及其所有的出边，则其所有相邻节点入度减少1
3. 回到第 1 步继续执行

```cpp
void TopsortbyQueue(Graph& G) {
    for (int i = 0; i < G.VerticesNum(); i++)
        G.status(G.V[i]) = UNVISITED; // 初始化
    }
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

### Resources 

- [A* Algorithm For Beginners](https://www.gamedev.net/articles/programming/artificial-intelligence/a-pathfinding-for-beginners-r2003/)
- [Introduction to A*](http://theory.stanford.edu/~amitp/GameProgramming/AStarComparison.html)
- [CS106B-Stanford-YouTube](https://www.youtube.com/watch?v=NcZ2cu7gc-A&list=PLnfg8b9vdpLn9exZweTJx44CII1bYczuk)
- [Algorithms-Stanford-Cousera](https://www.coursera.org/learn/algorithms-divide-conquer/home/welcome)
- [算法与数据结构-1-北大-Cousera](https://www.coursera.org/learn/shuju-jiegou-suanfa/home/welcome)
- [算法与数据结构-2-北大-Cousera](https://www.coursera.org/learn/gaoji-shuju-jiegou/home/welcome)
- [算法与数据结构-1-清华-EDX](https://courses.edx.org/courses/course-v1:TsinghuaX+30240184.1x+3T2017/course/)
- [算法与数据结构-2-清华-EDX](https://courses.edx.org/courses/course-v1:PekingX+04833050X+1T2016/course/)
- [算法设计与分析-1-北大-Cousera](https://www.coursera.org/learn/algorithms/home/welcome)
- [算法设计与分析-2-北大-EDX](https://courses.edx.org/courses/course-v1:PekingX+04833050X+1T2016/course/)

### 附录1

- Kruskal's algorithm

```cpp
void Kruskal(Graph& G, Edge* &MST) { // MST存最小生成树的边
    ParTree<int> A(G.VerticesNum()); // 等价类
    MinHeap<Edge> H(G.EdgesNum()); // 最小堆
    MST = new Edge[G.VerticesNum()-1]; // 为数组MST申请空间
    int MSTtag = 0; // 最小生成树的边计数
    for (int i = 0; i < G.VerticesNum(); i++) // 将所有边插入最小堆H中
        for (Edge e = G. FirstEdge(i); G.IsEdge(e); e = G. NextEdge(e))
            if (G.FromVertex(e) < G.ToVertex(e))// 防重复边
                H.Insert(e);
    int EquNum = G.VerticesNum(); // 开始有n个独立顶点等价类
    while (EquNum > 1) { // 当等价类的个数大于1时合并等价类
        if (H.isEmpty()) {
            cout << "不存在最小生成树." <<endl;
            delete [] MST;
            MST = NULL; // 释放空间
            return;
        }
        Edge e = H.RemoveMin(); // 取权最小的边
        int from = G.FromVertex(e); // 记录该条边的信息
        int to = G.ToVertex(e);
        if (A.Different(from,to)) { // 边e的两个顶点不在一个等价类
            A.Union(from,to); // 合并边的两个顶点所在的等价类
            AddEdgetoMST(e,MST,MSTtag++); // 将边e加到MST
            EquNum--; // 等价类的个数减1
        }
    }
}
```

- Prim's algorithm

```cpp
void Prim(Graph& G, int s, Edge* &MST) { // s是始点，MST存边
    int MSTtag = 0; // 最小生成树的边计数
    MST = new Edge[G.VerticesNum()-1]; // 为数组MST申请空间
    Dist *D;
    D = new Dist[G. VerticesNum()]; // 为数组D申请空间
    for (int i = 0; i < G.VerticesNum(); i++) { // 初始化Mark和D数组
        G.Mark[i] = UNVISITED;
        D[i].index = i;
        D[i].length = INFINITE;
        D[i].pre = s; // D[i].pre = -1 呢？
    }
    D[s].length = 0;
    G.Mark[s]= VISITED; // 开始顶点标记为VISITED
    int v = s;
    for (i = 0; i < G.VerticesNum()-1; i++) {// 因为v的加入，需要刷新与v相邻接的顶点的D值
        for (Edge e = G.FirstEdge(v); G.IsEdge(e); e = G.NextEdge(e))
            if (G.Mark[G.ToVertex(e)] != VISITED &&(D[G.ToVertex(e)].length > e.weight)) {
                D[G.ToVertex(e)].length = e.weight;
                D[G.ToVertex(e)].pre = v;
            }
            v = minVertex(G, D); // 在D数组中找最小值记为v
            if (v == -1) return; // 非连通，有不可达顶点
            G.Mark[v] = VISITED; // 标记访问过
            Edge edge(D[v].pre, D[v].index, D[v].length); // 保存边
            AddEdgetoMST(edge, MST, MSTtag++); // 将边加入MST
    }
}
int minVertex(Graph& G, Dist* & D) {
    int i, v = -1;
    int MinDist = INFINITY;
    for (i = 0; i < G.VerticesNum(); i++){
        if ((G.Mark[i] == UNVISITED) && (D[i] < MinDist)){
            v = i; // 保存当前发现的最小距离顶点
            MinDist = D[i];
        }
    }
    return v;
}
```
