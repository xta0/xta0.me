---
layout: post
list_title: 分布式计算 | Distributed Systems | Overview
title: 分布式计算概述
categories: [Cloud Compute]
---

### Course Overview

- Internal of cloud computing
- Go underneath the hood and look at **distributed systems** that underlie today's cloud computing technologies
- Part1
    - Introduction: Clouds, MapReduce, Key-Value stores
    - Classical Precursors: P2P systems, Grids
    - Widely-used algorithms: Gossip, Membership, Paxos
    - Classical algorithms: Time and Ordering, Snapshots, Multicast

## Part 1

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

1. Massive Scale
2. On-demand access
3. Data-Intensive Nature
    - What was MBs has now become TBs, PBs and XBs
4. New Cloud Programming Paradigms:


- Massive Scale
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

- ON-Demand Access: *AAS
    - HaaS: Hardware as a Service
    - IaaS: Infrastructure as a Service
        - You get access to flexible computing and storage infrastructure.Virtualization is one way of achieving this
        - Amazon EC2, S3, Microsoft AZure
    - Platform as a Service
        - You get access to flexible computing and storage, coupled with a software platform
        - Google's AppEngine(Python,Java,Go)
    - SaaS: Software as a Service
        - YOu get access to software services, when you need them.
        - Google docs, MS Office on demand

- New Cloud Programming Paradigms
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
    - NoSQL: MySQL is an industry standard, but Cassandra is 2400 times faster!

### Distributed Systems

- A "cloud" is the latest nickname for a distributed system
- Previous nicknames for "distributed system"
    - P2P systems
    - Grids
    - Clusters
    - Timeshared computers(Data Processing Industry)