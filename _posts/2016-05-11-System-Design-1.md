---
updated: "2018-08-20"
layout: post
title: Scalability Overview
list_title: 系统设计 | System Design | Overview
categories: [backend]
---

### Motavition

我对这个问题一直很感兴趣，但苦于一直没有实操的机会，虽然在阿里工作，但长期专注客户端，后端的scale问题基本接触不到，而对于大部分做业务的后端的工程师也接触不到这类问题。在阿里，这种基础性的问题在很早就被架构组的人解决了，而这些人目前在不在公司都不知道了，资料也很少，大部分在内网分享的系统设计都是建立在已经比较完善的底层服务之上了，要么做一些中间件服务，要么做一些性能调优之类的工作。因此想从头搞清楚阿里是如何Scale的基本上是找不到头绪的。

恰好最近在Youtube上看了Redit创始人Huffman在Udacity上分享的一些课程，其中有一部分讲到了Reddit是如何从一个单机Server Scale到千万用户量级的，听下来收获很大，但毕竟不是做后端出身，对于他讲的大部分内容在原理上是可以理解的，但涉及到的开源技术由于没有实操经验，一时间无法完全消化。

Github上有一个很全面的[System Design学习资料](https://github.com/donnemartin/system-design-primer)，本文及后面的一系列文章将是个人对这些学习资料的实践与总结。

### Overview

{% include _partials/components/lightbox-center.html param='/assets/images/2016/05/sd-1.png' param2='sd-1' %}

<p class="md-p-center"><a href="https://github.com/donnemartin/system-design-primer">source: System Design Primer</a></p>

### DNS

DNS就是域名系统，通常的理解就是将域名解析为IP，通常DNS服务是由电信运营商提供。其原理简单来说就查表，每台DNS服务器看自己的缓存中是否有有该域名的信息，如果有则返回，如果没有则会去查ROOT DNS Server。当浏览器或者OS拿到DNS结果后，会根据TTL的时间对IP地址进行缓存。

例如，Chrome可通过`chrome://net-internals/#dns`地址查看缓存的DNS信息，对于OSX可通过下面命令查看某域名对应的IP,如果要查看完整的DNS缓存，可参考文章最后一节的参考文献。

```shell
➜  ~ host -t a youtube.com
youtube.com has address 172.217.15.110
```

DNS服务的维护非常复杂，通常由政府或者运营商完成。DNS也容易被DDos攻击，一旦被攻击（如果不知道Twitter的IP地址），则无法通过域名进行访问。更多关于DNS的架构内容可参考文章最后一节的参考文献。

### CDN

CDN这个感念也不陌生，它是基于地理位置的分布式proxy server。主要作用是代理用户请求以及支持用户对静态文件的访问。打个比方，假设你开一家鞋店，你的App Server是总店，各个地区的CDN节点就是分店，由于是分店，鞋的种类、型号肯定没有总店那么全，但也基本包含了最热销的款式，所以只要不是特别独特的需求，分店是完全可以满足的，这样既缓解了总店和交通的压力，也提高了用户的购物体验。

具体来说，用户的请求首先到达DNS，DNS会将请求转发到离自己最近的CDN节点，如果该请求不带有任何Session信息，仅仅是浏览，那么该请求是不会到达后端的，如果带有用户信息，则由该CDN节点转发请求到后端的App Server。另外，当处理大促秒杀等场景时，CDN同样可以拦截掉大部分请求，从而达到控制流量的目的。

{% include _partials/components/lightbox-center.html param='/assets/images/2016/05/sd-3.jpeg' param2='sd-3' %}

<p class="md-p-center"><a href="https://www.creative-artworks.eu/why-use-a-content-delivery-network-cdn/">Source: Why use a CDN</a></p>

CDN的关键技术是对资源的管理和分发，按照分发的模式，可分为Push CDN和Pull CDN

- Push CDNs

Push是由管理人员主动upload资源到各个CDN节点，并将资源地址指向CDN节点，还要配置资源过期时间等。PUSH的内容一般是比较热点的内容，这种方式需要考虑的主要问题是分发策略，即在什么时候分发什么内容；对于流量小的网站，对资源的更新不会很频繁，因此适合采用这种方式，如果流量大的网站，使用这种方式则会带来较高操作成本。

- Pull CDNs

Pull的方式是被动的按需加载，当获取静态资源的请求到达CDN后，CDN发现没有该资源，则会"回源"，向App Server询问，在得到资源后，CDN将该资源缓存在自己的节点上。

使用这种方式CDN往往需要一个预热的阶段，通过一系列少量请求将资源推到CDN节点上，然后再开放给大量的用户，如果不经过预热，则会导致大规模的"回源"请求，很容易给内部的Server带来较高的负载。因此这种方式适用于流量较高的网站，当运行一段时间后，CDN将保留当前最新的最热门的资源。

### Load Balancer

负载均衡是一个非常重要的系统，也可以说是无处不在的一个系统，比如在DNS中可以使用负载均衡动态分配请求到不同的CDN上，对于内部App Server的集群也需要使用负载均衡动态分配请求到不同Server上。


### App Servers

由前面小节的讨论可知，当用户处于非登录状态时，可直接访问CDN获取静态资源，但是如果用户是登录状态，则请求会通过Load Balancer路由到Web Server的cluster上。而对于所有cluster中的Web Server它们代码相同，是彼此的clone。最重要的是，它们不能在本地存储任何用户相关信息以及状态。

用户的Session应该统一存放在一个外部的缓存中，比如Redis server。如何维护以及更新Session的状态后面文章中将会详细讨论。另外，对于分布式Server的代码同步以及部署也是一个问题，好在目前有很多开源项目可以解决这个问题，比如Ruby体系的Capistrano等

这种基于Load Balancer + 分布式Server的扩展方式可以称为horizontally scale。这种方式可以handle大量的并发请求，但是如果后端只有一两个Database，那么数据库的读写将会很快成为瓶颈。

### Databases

对于数据库的扩展可以选择两种方式，一种方式是使用Master-Slave的架构来复制出多份数据库，如上图中所示，让所有读请求打到Slave上，写请求路由到Master上。这种方式会随着业务不断的变复杂，数据表字段不断增加，查询速度也会变得越来越复杂，尤其是JOIN操作将会非常耗时，这时可能需要一个DBA要对数据库进行sharding（分库分表），做SQL语句调优等一系列优化，但这些优化并不解决根本问题。

第二种方式是在开始的时候就使用NoSQL数据库，比如MongoDB，CouchDB。使用这类数据库，JOIN操作可以在应用层代码中实现，从而减少DB操作的时间。但是无论哪种方式，随着数据量增多，业务不断复杂化，对DB的查询依旧会变得越来越慢，这时候就要考虑使用缓存。

### Cache

这里说的缓存指的是内存级别的缓存，比如Memcached或者Redis。永远不要做文件级别的缓存，这对server的clone或者scale都非常不利。

对于内存级别的缓存，其存储以及访问形式为key-value。当app server想要从DB取数据时，首先去缓存里查一下，没有再做DB查询，例如，Redis可以支持10万/秒级别的并发查询，写缓存也要比DB快很多。对缓存的读写也有两种方式：

- 缓存DB Query的结果

这种方式仍然是目前使用比较多的一种方式。每当DB查询完成时将结果缓存在Cache中，key是query的hash值。再下一次执行相同query时，先检查缓存，如果命中则缓存结果。这种方式实际上有很多问题，一个比较大的问题是缓存数据的时效性，当DB中的数据发生变化时，相同query的结果会发生变化，但是缓存是无法感知的，这时要删掉该query对应的所有缓存数据，这是件很被动很困难的事

- 缓存Objects

这种方式是个人比较推荐的一种方式，它将从DB取出的数据进行ORM，将得到的Object存入到缓存中。假设有一个类`Product`，它有几个property分别是`prices`,`reviews`,`category`等等，现在需要根据某个ID来查询某个Product，显然首先要从DB里根据查出这个ID对应的一系列上述propertiy的值，然后利用查出来的这些数据构建一个类型为Product的Object，最后将这个Object放入缓存中。这样当每次数据库中数据发生变化时，App Server可以监听到这个Event，然后异步的对缓存中的Obejct进行更新，而不是像第一种方式一样去删除某条缓存。这种缓存方式显然更高效。

缓存的内容可以包括用户的session信息（不要存到数据库中），静态页面，用户-朋友关系等。另外，使用Redis起单独的server做缓存要好过在同一个App Server中使用memcached，后者在维护数据一致性方面难度很大。在某些条件下，甚至可以使用Redis取代DB做持久化存储。

### Asynchronism

如果网站的流量大了，所有的操作尽量异步。异步操作又分为很多种，这里讨论两种，一种是异步任务，一种是定时任务。所谓定时任务是指将一些时效性不是特别强，但是却消耗资源，消耗时间的计算（比如投票，日志收集等）提前做好，或者每一到两个小时执行一次，然后将得到结果缓存，缓存的方法有很多，比如将计算结果更新到数据库中，或者upload静态页面到CDN上，等等

另一种就是所谓的异步任务，这种异步操作对时效性有要求，同时又消耗大量的计算资源，用户需要留在前台等待计算结果。这时候需要首先一个任务队列（如上图中所示）来缓存异步任务，常用的有[RabbitMQ](http://www.rabbitmq.com/),Apache 家族的[ActiveMQ](http://activemq.apache.org/)，基于Redis的[Kue](https://github.com/Automattic/kue)等等。

### CAP 理论

### Resource

- [CS75 (Summer 2012) Lecture 9 Scalability Harvard Web Development David Malan](https://www.youtube.com/watch?v=-W9F__D3oY4&t=955s)
- [HOw to view DNS cache in OSX](https://stackoverflow.com/questions/38867905/how-to-view-dns-cache-in-osx)
- [DNS Architechture](https://docs.microsoft.com/en-us/previous-versions/windows/it-pro/windows-server-2008-R2-and-2008/dd197427(v=ws.10))
- [Scalability for Dummies](http://www.lecloud.net/post/7295452622/scalability-for-dummies-part-1-clones)
- [System Design Primer](https://github.com/donnemartin/system-design-primer)

