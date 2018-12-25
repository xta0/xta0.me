---
layout: post
list_title: 数据结构基础 | Data Structre | 索引-倒排索引 | Inverted Indexing
title: 倒排索引
mathjax: true
categories: [DataStructure]
---

## 倒排索引(Invered Index)

倒排索引主要用来处理对非主码信息（属性）检索的场景，对于这些属性的信息，如果要进行检索，同样也要对其建立索引。对属性建索引分为两大类

1. 对数据库这种结构化信息建索引
2. 对文本文件进行索引（以文中的词word为索引项建立的索引）

|EMP#| NAME | DEPT | PROF| 
|--|---|---|---|
| 0155 | Kian | Math | Professor|
| 0421 | Mason | Math | Teacher|
| 0208 | Carter | CS | Assistant |
| 0343 | Lily | CS | Professor|
| 0223 | William | Math | Assistant|

例如上述数据库表中，主键为`EMP#`，辅码（属性）为`NAME,DEPT,PROF`，显然不同记录中的属性值会有相同的情况，现在想要索引`DEPT`为`Math`的员工，则需要为`DEPT`这个属性建立索引

### 基于属性的倒排

 对某属性按属性值建立索引表，称倒排表。每个索引对用`<attr,ptrList>`表示。和主索引表中的索引对不同的是，`ptrList`指针指向的不是磁盘文件中某条记录的地址，而是这个属性所对应的主码值，再通过主码找到实际文件中的记录。由于基于属性建立起来的索引是无序的，因此成为**倒排索引**，存储这些倒排索引对的文件称为倒排文件。

 还是上面的例子，可生成如下的倒排索引表：

<div style=" content:''; display: table; clear:both; height=0">
    <div style="float:left">
        <table>
                <thead>
                    <tr><th>PROF</th><th>EMP#</th></tr>
                </thead>
                <tbody>
                    <tr> <td>Professor</td><td>0155,0343</td></tr>
                    <tr> <td>Teacher</td><td>0421</td></tr>
                    <tr> <td>Assistant</td><td>0208, 0223</td></tr>
                </tbody>
        </table> 
    </div>
    <div style="float:left;margin-left:40px;">
        <table>
            <thead>
                <tr><th>DEPT</th><th>EMP#</th></tr>
            </thead>
            <tbody>
                <tr> <td>Math</td><td>0155, 0421, 0223</td></tr>
                <tr> <td>CS</td><td>0208,0343</td></tr>
            </tbody>
        </table>
    </div>
</div>

使用倒排表的优点是能够对于基于属性的检索进行较高效率的处理，缺点是，（1）建立倒排表的存储代价，（2）有一定的维护成本，降低了更新运算的效率。
 
### 对文本文件的倒排

对文本的检索是平时用的最多的场景，比如搜索引擎对的关键字检索，会首先根据对关键字的索引得到一组倒排列表作为结果，然后对该结果进行一些算法调整（广告）最终得到一组网页序列返回。所谓对文本文件的倒排是指以文本文件中出现的词（word）作为一个索引项，得到它一个倒排列表，这个列表包含了该词出现在哪一篇文章（网页），以及文章中的位置，或者哪一个文章里面出现了多少次等等的信息。

对文本建索引可以从两个角度出发，一种是对词建索引，得到一个词表（比如Google对关键词建索引），根据词来检索。另一种比较老的方式是对全文建索引。

- **词索引 ( word index )**

词索引的基本思想为把正文看作由符号和词所组成的集合，从正文中抽取出关键词，然后用这些关键词组成一些适合快速检索的数据结构。

词索引适用于多种文本类型，特别是那些可以很容易就解析成一组词的集合的文本。比如按空格分割英文单词，中文则需要要经过“切词”处理。

<mark>基于词的索引是使用最广泛的建索引的方式</mark>，它需要一个已经排过序的关键词的列表,其中每个关键词和一个指针组成一个索引对，该指针可以指向该关键词出现文档集合或者这个单词在文档中的位置，举例如下，假设我们有6个文档，每个文档都是长字符串：

|文档编号 | 文本内容|
|----|----|
|1 | Pease porridge in the pot|
|2| Nine days old.|
|3| Some like it hot, some like it cold,|
|4 |Some like it in the pot,|
|5 | Nine days old.|

对上述表建立倒排索引结构如下：

| 编号  | 词语 |  （文档编号，位置）|
| ----| ----| ----|
|1|cold|(1,6)|
|2|days||
|3|hot|(1,3)|
|4|in||
|5|like||
|6|nine||
|7|old||
|8|pease|(1,1) (1,4) (2,1)|
|9|porridge|(1,2),(1,5)|
|10|it||
|11|pot||
|12|some||
|13|the||

总结一下，以分词的方式建索引的步骤大致为：

1. 对文档集中的所有文件都进行分割处理，把正文分成多条记录文档，切分正文记录取决于程序的需要
，可以是定长的块、段落、章节，甚至一组文档 
2. 给每一条记录赋一组关键词,以人工或者自动的方式从记录中抽取关键词,比如
    - 停用词( Stopword )
        - 去掉英文的冠词，定冠词，助动词，中文的“的，地，得”等等
    - 抽词干( Stemming )
        - Computer/Compute/Computation
    - 切词（Segmentation）
        - 英文可以用空格
        - 中文需要借助NLP切词
    
3.  建立正文倒排表、倒排文件得到各个关键词的集合对于每一个关键词，检索所有文档得到其倒排表，然后把所有的倒排表存入文件

- **检索关键词**

建好索引后，当输入关键词时，只需要按照下面步骤进行：

1. 在倒排文件中检索关键词
2. 如果找到了关键词，那么获取文件中的对应的倒排表，并获取倒排表中的记录。
3. 在第一步中，如果关键词表特别大，顺序查找效率低，则通常使用另一个索引结构（字典）进一步对关键词表进行 有效索引（Trie，散列）

### 倒排文件的优劣

使用倒排文件是一种高效的检索方式，用于文本数据库系统。但是索引文件的空间代价往往非常高，经常是文档集的数倍，如果组织的不是很好，索引的查找效率也会很低。

## Resources 

- [CS106B-Stanford-YouTube](https://www.youtube.com/watch?v=NcZ2cu7gc-A&list=PLnfg8b9vdpLn9exZweTJx44CII1bYczuk)
- [Algorithms-Stanford-Cousera](https://www.coursera.org/learn/algorithms-divide-conquer/home/welcome)
- [算法与数据结构-1-北大-Cousera](https://www.coursera.org/learn/shuju-jiegou-suanfa/home/welcome)
- [算法与数据结构-2-北大-Cousera](https://www.coursera.org/learn/gaoji-shuju-jiegou/home/welcome)
- [算法与数据结构-1-清华-EDX](https://courses.edx.org/courses/course-v1:TsinghuaX+30240184.1x+3T2017/course/)
- [算法与数据结构-2-清华-EDX](https://courses.edx.org/courses/course-v1:PekingX+04833050X+1T2016/course/)
- [算法设计与分析-1-北大-Cousera](https://www.coursera.org/learn/algorithms/home/welcome)
- [算法设计与分析-2-北大-EDX](https://courses.edx.org/courses/course-v1:PekingX+04833050X+1T2016/course/)

