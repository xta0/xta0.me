---
updated: "2018-08-23"
layout: post
title: 高性能，高可用，可扩展
list_title: 系统设计（二）| System Design | 原则 | Performance, Availability and Scalibilty
categories: [backend]
---

## 高性能

高性能分为单机高性能以及集群高性能

### Latency vs Throughput

延时是指每次请求的链路耗时，单位是时间单位；吞吐量是指单位时间内能处理多少次请求。打个简单的比方，有一家汽车修理厂，修理一辆汽车耗时8小时，一天能够修理120量车，那么：

1. Latency: 8 hours
2. Throughput: 120 cars/day or 5 cars/hour

通常情况下，系统优化的目标是在可接受的延迟时间范围内达到最大的吞吐量。

## 高可用



## Resource

- [高可用系统](https://coolshell.cn/articles/17459.html)