---
updated: "2018-08-20"
layout: post
title: 系统设计 | Load Balancer
list_title: 系统设计 | System Design | 负载均衡 | Load Balancer
categories: [backend]
---

### Motavition

我对这个问题一直很感兴趣，但苦于一直没有实操的机会，虽然在阿里工作，但长期专注客户端，后端的scale问题基本接触不到，而对于大部分做业务的后端的工程师也接触不到这类问题。在阿里，这种基础性的问题在很早就被架构组的人解决了，而这些人目前在不在公司都不知道了，资料也很少，大部分在内网分享的系统设计都是建立在已经比较完善的底层服务之上了，要么做一些中间件服务，要么做一些性能调优之类的工作。因此想从头搞清楚阿里是如何Scale的基本上是找不到头绪的。

恰好最近在Youtube上看了Redit创始人Huffman在Udacity上分享的一些课程，其中有一部分讲到了Reddit是如何从一个单机Server Scale到千万用户量级的，听下来收获很大，但毕竟不是做后端出身，对于他讲的大部分内容在原理上是可以理解的，但涉及到的开源技术由于没有实操经验，一时间无法完全消化。因此，我决定先从方法论入手，搞清楚原理，然后有机会再动手实操演练


## Load Balancer

我们先从负载均衡开始，由于这个概念较容易理解，这里不做过多介绍，只要记住它主要完成两个任务，一是负责分发请求，二是负责冗余，即某个server failed了，能够及时发现并redirect请求到其它server上。

### Round-Robin

我们可以使用Nginx + PHP在本地简单模拟一个Load Balancer。由于Nginx默认提供对Load Balancer的支持，只需做简单配置即可。我们先在本地起三个Server，分别监听`10001,10002,10003`三个端口：

{% include _partials/components/lightbox-center.html param='/assets/images/2015/05/lb-1.png' param2='1' %}

接下来，我们需要写一个Nginx配置来启动负载均衡的Server

```yaml
http{
    #load balancer
    upstream php_servers{
        server localhost: 10001;
        server localhost: 10002;
        server localhost: 10003;
    }
    server {
        listen       8080;
        server_name  localhost;
        location /{
            proxy_pass http://php_servers
        }
    }
}
```
上述配置中，我们令负载均衡Server监听8080端口，当有请求到来时，转发到三台`php_servers`上。我用一段脚本来模拟请求，假设每隔0.5s向负载均衡Server上发一个请求，观察分配结果

{% include _partials/components/lightbox-center.html param='/assets/images/2015/05/lb-2.png' param2='1' %}

我们看到Load Balancer会依次将request投递到三台服务器上，从输出来看，Ngixn默认的策略为**Round-Robin**，即轮训算法。

1. 这种策略的优点是适用性更强，不依赖于客户端的任何信息，完全依靠后端服务器的情况来进行选择。能把客户端请求更合理更均匀地分配到各个后端服务器处理。
2. 缺点是同一个客户端的多次请求可能会被分配到不同的后端服务器进行处理，无法满足做会话保持的应用的需求。

### IP Hash

除了Round Robin外，Nginx还支持其它的负载均衡策略，比如使用`ip hash`。所谓`ip hash`是指将所有来自相同IP的Request均投递到同一个server上，这个特性对于维护登录信息之类的操作很有用，我们可以修改Nginx配置如下：

```yaml
upstream php_servers{
    ip_hash;
    server localhost: 10001;
    server localhost: 10002;
    server localhost: 10003;
}
```

{% include _partials/components/lightbox-center.html param='/assets/images/2015/05/lb-3.png' param2='1' %}

由于本地模拟请求的IP地址相同，我们看到所有的请求均被投递到PHP Server #1上，如果此时该Server #1挂掉，Nginx则会自动将请求转向Server #2。

1. 这种策略的优点是能较好地把同一个客户端的多次请求分配到同一台服务器处理，避免了轮询无法适用会话保持的需求。
2. 缺点是当某个时刻来自某个IP地址的请求特别多，那么将导致某台后端服务器的压力可能非常大，而其他后端服务器却空闲的不均衡情况、

### Least Connection

Nginx提供的另一种基于最少链接(least_conn)的分配策略。我们知道轮询算法是把请求平均的转发给各个后端，使它们的负载大致相同。这有个前提，就是每个请求所占用的后端时间要差不多，如果有些请求占用的时间很长，会导致其所在的后端负载较高。在这种场景下，把请求转发给连接数较少的后端，能够达到更好的负载均衡效果，这就是least_conn算法。least_conn算法很简单，首选遍历后端集群，比较每个后端的conns/weight，选取该值最小的后端。如果有多个后端的conns/weight值同为最小的，那么对它们采用加权轮询算法。

为了模拟这种情况，我们假设Server #1此时流量升高，响应时间变慢，修改Server #1的代码为

```php
<?php

sleep(20);

echo "Sleepy server finally done! \n";%
```
修改Nginx负载均衡策略为：

```yaml
#load balancer
upstream php_servers{
    least_conn;
    server localhost:10001;
    server localhost:10002;
    server localhost:10003;
}
```

{% include _partials/components/youtube.html param='https://www.youtube.com/embed/BZ_zdZYUDwk' %}

观察上述视频可发现，我们在左上角的console中，先用`while`循环模拟了一次请求，由于第一个request被分配到Server #1上，而Server #1此时处于sleep状态，因此后面的请求将会被自动分到Server #2和#3上。为了验证Nginx此时的行为，我们在右下角的console中再次模拟请求，发现Nginx此时感知到Server #1处于阻塞状态而自动的将请求投递到Server #2和Server #3上

### Redundancy

最后来我们可以进行冗余测试，测试步骤为

1. 令PHP Server #1暂停运行
2. 令PHP Server #2暂停运行
3. 令PHP Server #1恢复运行
4. 令PHP Server #2恢复运行

{% include _partials/components/youtube.html param='https://www.youtube.com/embed/_ROsTIJkTyg' %}

上述视频中可观察到，当Server #1暂停运行后，Load Balancer将请求分配到第2,3台Server上，当Server #2暂停运行后，Load Balancer将请求分配到第三台Server上，所有两台server恢复运行后，又重新回到Round Robin的分配策略。

{% include _partials/post-footer-1.html %}

### Resource

- [CS75 (Summer 2012) Lecture 9 Scalability Harvard Web Development David Malan](https://www.youtube.com/watch?v=-W9F__D3oY4&t=955s)
- [Using nginx as HTTP load balancer](http://nginx.org/en/docs/http/load_balancing.html)
- [NGINX Load Balancing - HTTP Load Balancer](https://docs.nginx.com/nginx/admin-guide/load-balancer/http-load-balancer/)
- [Module ngx_http_upstream_module](http://nginx.org/en/docs/http/ngx_http_upstream_module.html)

