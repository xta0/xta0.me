---
layout: post
list_title:   云计算 | Clouds Computing | Concepts Part 1 | 
title: 云计算概述
categories: [Notes, Cloud Compute]
---

### Course Overview

- Internal of cloud computing
- Go underneath the hood and look at **distributed systems** that underlie today's cloud computing technologies
- Part1
    - Introduction: Clouds, MapReduce, Key-Value stores
    - Classical Precursors: P2P systems, Grids
    - Widely-used algorithms: Gossip, Membership, Paxos
    - Classical algorithms: Time and Ordering, Snapshots, Multicast

### Overview

- Introduction to Clouds
    - Hisotory, What, Why
    - Comparison with previous generation of distributed systems
- Clouds are distributed systems
- Mapreduce and Hadoop
    - Paradigm, Examples

- Cloud Providers
    - AWS Amazon Web Services:
        - ES2：Elastic Compute Cloud
        - S3: SImple Storage Service
        - EBS: Elastic Block Storage
    - Microsoft Azure
    - Google Compute Engine
    - Many, many more

- Four features in Today's clouds
    1. **Massive Scale**
        - Facebook
            - 30k servers in 2009
            - 60k in 2010
            - 180k in 2012
        - Microsoft has 150k servers in 2008, growth rate of 10k per month, 80K total running Bing
    - AWS EC2
        - 40k machines
        - 8 cores/machine
    - eBay
        - 50k machines
    2. **On-demand access**
        - HaaS: Hardware as a Service
        - IaaS: Infrastructure as a Service
            - You get access to flexible computing and storage infrastructure.Virtualization is one way of achieving this
            - Amazon EC2, S3, Microsoft AZure
        - Platform as a Service
            - You get access to flexible computing and storage, coupled with a software platform
            - Google's AppEngine(Python,Java,Go)
        - SaaS: Software as a Service
            - You get access to software services, when you need them.
            - Google docs, MS Office on demand
    3. **Data-Intensive Nature**
        - What was MBs has now become TBs, PBs and XBs
    4. **New Cloud Programming Paradigms**
        - Google: MapReduce
        - Amazon: Elastic MapReduce service(pay as you go)
        - Google(MapReduce)
            - Indexing a chain of 24 MapReducer Jobs
            - 200k mapreduce jobs processing 50PB/month (in 2006)
        - Yahoo(Hadoop + Pig)
            - WebMap: a chain of 100 MapReduce jobs
        - Facebook(Hadoop + Hive)
            - 300 Tb total, adding 2TB/day (in 2008)
            - 3k jobs processing 55TB/day
        - NoSQL
            - MySQL is an industry standard, but Cassandra is 2400 times faster!

### Distributed Systems

- A "cloud" is the latest nickname for a distributed system
- Previous nicknames for "distributed system"
    - P2P systems
    - Grids
    - Clusters
    - Timeshared computers(Data Processing Industry)
- Definitions from textbooks
    - A distributed system is a collection of independent computers that appear to the uses of the system as a single computer [Andrew Tanenbaum]
    - A distributed system is serverl computers doing something together. Thus, a distributed system has three primary characteristics: multiple computer, interconnections and share state. [Michael Schroeder]
- A working definition of "Distributed System"
    - A distributed system is a collection of entities, each of which is <mark>autonomous</mark>, <mark>programmable</mark>, <mark>asynchronous</mark> and <mark>failure-prone</mark>, and which communicate through an <mark>unreliable communication</mark> medium.

- A range of interesting problmes for distributed system designeers
    - P2P systems [Gnutella, Kazaa, BitTorrent]
    - Cloud Infrastrutures [AWS, Azure, Google Cloud]
    - Cloud Storage [Key-value stores, NoSQL, Cassandra]
    - Cloud Programming [MapReduce, Storm, Pregel]
    - Coordinations [Zookeeper, Paxos, Snapshots]
    - Managing Many Clients and Servers Concurrently

## MapReduce

- Terms are borrowed from Functional Language(e.g., Lisp)
    - Sum of squares
    
    ```lisp
    (map square'(1 2 3 4)) 
    (reduce +'(1 4 9 6))
    ```

- **Map**

Let's consider a sample application: Wordcount

> You are given a huge dataset(e.g., Wikipedia dump all of Shakespeare's works) and asked to list the count for each of the words in each of the documents therein.

Map process individual records to generate intermediate key/value pairs.

```
//input<filename, file.txt>
str = "Welcome every Hello Everyone" 

map(str) => {key,value} =>  key         | value
                            ------------|-------
                            Welcome     | 1
                            Everyone    | 1
                            Hello       | 1
                            Everyone    | 1
```
- Parallelly process individual records to generate intermediate key/value pairs


```
task #1: 
map(str1) => {key,value} => key         | value
                            ------------|-------
                            Welcome     | 1
                            Everyone    | 1

task #2: 
map(str2) => {key,value} => key         | value
                            ------------|-------
                            Hello       | 1
                            Everyone    | 1
```

- Hadoop Code - map

```java
public static class MapClass extands MapReduceBase implements Mapper<LongWritable, Text, Text, IntWritable>{
    private final static IntWritable one = new IntWritable();
    private Text word = new Text();
    public void map(LongWritable key, Text value, OutputCollector<Text,IntWritable> output, Reporter reporter) throws IOException{
        String line = value.toString();
        StringTokenizer itr = new StringTokenizer(line);
        while(itr.hasMoreTokens()){
            word.set(itr.nextToken());
            output.collect(word,one); //<key,value>
        }
    }
}
```

- **Reduce**

Reduce processes and merges all the intermediate values associated per key

```
reduce(<k,v>) => {key,value}

key         | value            key     | value
------------|-------    =>   ----------|---------
Welcome     | 1               Everyone | 2
Everyone    | 1               Hello    | 1
Hello       | 1               Welcome  | 1
Everyone    | 1
```

- Each key assigned to one Reducer
- Parallelly process and merges all intermediate values by partitioning keys
    - Popular: Hash Partitioning, i.e., key is assigned to reduce# = hash(key) % #reducer servers

```
------------                       ------------
Everyone 1 |  => Reduce Task #1 => | Everyone 2
Everyone 1 |                       -------------
------------

----------                        ------------
Hello   1 |  => Reduce Task #2 => | Hello   1
Welcome 1 |                       | Welcome 1
----------                        -------------
```

- Hadoop Code - Reduce

```java
public static class ReduceClass extands MapReduceBase implements Reducer<Text,IntWritable, Text,IntWritable>{
    public void reduce(Text key, Iterator<IntWritable> values, OutputCollector<Text,IntWritable> output, Reporter reporter) throws IOException{
        int sum = 0;
        while(values.hasNext()){
            sum += values.next().get();
        }
        output.collect(key,new IntWritable(sum)); //<key,value>
    }
}
```

### Some Applications of MapReduce 

- **Distributed Grep**
    - Input: large set of files
    - Output: lines that match pattern
    - Map: Emits a line if it matches the supplied pattern
    - Reduce: Copies the intermediate data to output
- **Reverse Web-Link Graph**
    - Input: Web graph: tuples(a,b) where (page a -> page b)
    - Output: For each page, list of pages that link to it 
    - Map: Process web log and for each input`<source, target>`, outputs `<target, source>`
    - Reduce: emits `<target, list(source)>`
- **Sort**
    - Input: Series of (key,value) pairs
    - Output: Sorted `<value>s`
    - Map: `<key,value>` -> `<value,_>`
        - output is sorted (e.g., quick sort)
        - sort part of the whole file
            - { {1,10,30,49,88}, {2,5,9,66,87}, ... }
    - Reducer: `<key,value>` -> `<key,value>`
        - combile all sorted sub array using merge sort
        - Partition function - partition keys accross reducers based on ranges(1001-2000->reducer #1, 2001-3000 -> reducer #2)
            - Take data distribution into account to balance reducer tasks

### MapReduce Scheduling

- **For User**
    1. Write a Map program(short), write a Reduce program(short)
    2. Submit job: wait for the result
    3. Need to know nothing about distributed programming
- **For System(Hadoop)**
    1. Parallelize Map
    2. Transfer data from Map to Reduce
    3. Parallelize Reduce
    4. Implement Storage for Map input, Map output, Reduce input,and Reduce output.

    > Ensure that no Reduce starts before all Maps are finished. That is, ensure the barrier between the Map phase and Reduce phase.

### Resource

- [Cloud Computing](https://www.coursera.org/learn/cloud-computing)