---
layout: post
list_title: Algorithms-7 | 广度优先搜索 | BFS
title: 广度优先搜索
mathjax: true
---

## 广度优先搜索

广度搜索相比深度搜索，只需要知道目标节点位于第几层，便可确定到达的路径数目，可以确保找到最优解，因此通常用于求解层次相关问题（步数问题，路径问题）。广搜在遍历的过程中，通常使用队列存储节点，对每一个节点，可以用一个结构定义：

```cpp
struct node{
    int value;
    int layer; //在第几层的位置
    bool isVisited; //是否被遍历过
    node* parent; //便于路径搜索
    vector<node* >children; //该节点所能到到的子节点
}
```

### 广搜算法

1. 将初始节点`s0`放入Queue中
2. 如果Queue为空，则问题无解，失败退出
3. 从Queue中取出第一个节点，记为`n`, 将其状态标记为visit
4. 考察`n`是否为目标节点，若是，则退出
5. 若`n`不是目标节点，则看`n`是否有子节点，若没有，则转到第2步
6. 遍历`n`的子节点，如果没有被visit·，则放入Queue中，转到第2步

```cpp
void bfs{
    queue<node> q;
    q.push(node(x));
    
    while(!q.empty()){
        node = q.front();
        node.isVisited = true;
        if(node.value==K){ //找到目标
            return；
        }
        for(auto child : node.children){
            if(!child.isVisited){
                q.push(child);
            }
        }
        q.pop();
    }
}
```

广度搜索的一个重点是要先构造**状态空间**，即如何从一个状态，生成一棵树，树中的每个节点对应状态空间的一个状态。而状态空间的生成是一个逐层扩展的过程，枚举根节点所能到达的状态构成第二层状态，这些状态可以作为根节点的子节点，依次类推展开完整的空间。

### 广搜与深搜的比较

- 广搜一般用于状态表示比较简单、求最优策略的问题
    - 优点：**是一种完备策略**，即只要问题有解，它就一定可以找到解。并且，广度优先搜索找到的解，**还一定是路径最短的解**。
    - 缺点：盲目性较大，尤其是当目标节点距初始节点较远时，将产生许多无用的节点，因此其搜索效率较低。需要保存所有扩展出的状态，占用的空间大

- 深搜几乎可以用于任何问题
    - 只需要保存从起始状态到当前状态路径上的节点

### 双向BFS

- DBFS算法是对BFS算法的一种扩展。
    - BFS算法从起始节点以广度优先的顺序不断扩展，直到遇到目的节点
    - DBFS算法从两个方向以广度优先的顺序同时扩展，一个是从起始节点开始扩展，另一个是从目的节点扩展，直到一个扩展队列中出现另外一个队列中已经扩展的节点，也就相当于两个扩展方向出现了交点，那么可以认为我们找到了一条路径
- 比较
    - DBFS算法相对于BFS算法来说，由于采用了双向扩展的方式，搜索树的宽度得到了明显的减少，时间复度和空间复杂度上都有提高！
    - 假设1个节点能扩展出n个节点，单向搜索要m层能找到答案，那么扩展出来的节点数目就是:`(1-n^m)/(1-n)`
    - 双向广搜，同样是一共扩展m层，假定两边各扩展出`m/2`层，则总节点数目 `2 * (1-n^m/2)/(1-n)`
    - 每次扩展节点总是选择节点比较少的那边进行扩展，并不是机械
的两边交替。

- 实现思路

```
void dbfs()
{
    1. 将起始节点放入队列q0,将目标节点放入队列q1；
    2. 当两个队列都未空时，作如下循环：
        1) 如果队列q0里的节点比q1中的少,则扩展队列q0；
        2) 否则扩展队列q1
    3. 如果队列q0未空，不断扩展q0直到为空；
    4. 如果队列q1未空，不断扩展q1直到为空；
}
int expand(i) //其中i为队列的编号，0或1
{
    取队列qi的头节点H；
    对H的每一个相邻节点adj：
    1 如果adj已经在队列qi之中出现过，则抛弃adj；
    2 如果adj在队列qi中未出现过，则:
    1） 将adj放入队列qi；
    2) 如果adj 曾在队列q1-i中出现过, 则：输出找到的路径
} 
```
- [CS106B-Stanford-YouTube](https://www.youtube.com/watch?v=NcZ2cu7gc-A&list=PLnfg8b9vdpLn9exZweTJx44CII1bYczuk)
- [Algorithms-Stanford-Cousera](https://www.coursera.org/learn/algorithms-divide-conquer/home/welcome)
- [算法与数据结构-1-北大-Cousera](https://www.coursera.org/learn/shuju-jiegou-suanfa/home/welcome)
- [算法与数据结构-2-北大-Cousera](https://www.coursera.org/learn/gaoji-shuju-jiegou/home/welcome)
- [算法与数据结构-1-清华-EDX](https://courses.edx.org/courses/course-v1:TsinghuaX+30240184.1x+3T2017/course/)
- [算法与数据结构-2-清华-EDX](https://courses.edx.org/courses/course-v1:PekingX+04833050X+1T2016/course/)
- [算法设计与分析-1-北大-Cousera](https://www.coursera.org/learn/algorithms/home/welcome)
- [算法设计与分析-2-北大-EDX](https://courses.edx.org/courses/course-v1:PekingX+04833050X+1T2016/course/)