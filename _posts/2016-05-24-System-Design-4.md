---
updated: "2018-08-20"
layout: post
title: 系统设计 （四）| Nginx中的负载均衡 | Load Balancer In Nginx
list_title: 系统设计 | System Design | Load Balancer
categories: [backend]
---

## Load Balancer

负载均衡这个概念相对来说比较容易理解，其工作方式如下图所示，由于在之前文章中已经做过介绍，这里就不再展开了。Load Balancer主要完成两个任务，一是负责分发请求，二是处理冗余(即某个server failed了，能够及时发现并redirect请求到其它server上)。本文将使用Nginx来模拟实现三种路由策略，分别是Round-Robin， Least busy以及Session/cookies。通过观察这几种策略的表现来带给大家一些直观的感受。

{% include _partials/components/lightbox-center.html param='/assets/images/2016/05/sd-2.png' param2='sd-2' %}

<p class="md-p-center"><a href="http://horicky.blogspot.com/2010/10/scalable-system-design-patterns.html">source: Scalable System Design Patterns</a></p>

### 轮训 Round-Robin

Round-Robin是一种简单高效的策略，我们熟悉的DNS服务，P2P网络均适用这种策略。简单的说Roound-Robin算法就是维护一个机器列表，当请求过来时，对当前的列表进行轮训，找到下一个可投递的机器进行路由。

> Round-Robin通常用于"single-point-of-entry to multiple servers in the background"的场景


寻找下一个可投递机器的算法可以是简单粗暴的按照index等比例分配，比如列表中有1，2，3三台机器，那么路由的顺序也是1，2，3，请求被投递到三台机器的概率相同。另一种方式是按照权重来进行非等比分配，即为每台机器配置一个权重，这种方式也叫做**Weighted Round-Robin**，即加权轮训。比如有列表中有三台机器，三台机器的QPS依次为: 100 req/sec, 300 req/sec, 25 req/sec。这时候可以为其配置一个权重表：

```shell
Resource Weight
——– ——
server1.fqdn 4
server2.fqdn 12
server3.fqdn 1
```
因此对于17个请求，有4个被投递到server #1，12个被投递到server #2，1个被投递到server #3。

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
上述配置中，我们令负载均衡Server监听8080端口，当有请求到来时，该请求会被转发到三台`php_servers`中的一台上。我用一段脚本来模拟请求，假设每隔0.5s向负载均衡Server上发一个请求，观察分配结果

{% include _partials/components/lightbox-center.html param='/assets/images/2015/05/lb-2.png' param2='1' %}

我们看到Load Balancer会依次将request投递到三台服务器上，从输出来看，Ngixn默认的策略为**Round-Robin**。

总结一下，Round-Robin策略的<mark>优点</mark>是适用性强，不依赖于客户端的任何信息，完全依靠后端服务器的情况来进行选择。能把客户端请求更合理更均匀地或者按比例的分配到各个后端服务器处理。<mark>缺点</mark>是同一个客户端的多次请求可能会被分配到不同的后端服务器进行处理，无法满足做会话保持的应用的需求。此外，它并不考虑server的负载情况，因此对负载较高的server压力会持续升高。


### Session / Cookie

除了Round Robin外，Nginx还支持其它的负载均衡策略，比如使用`ip hash`。所谓`ip hash`是指将所有来自相同IP的Request均投递到同一个server上，这个特性对于维护用户登录信息之类的操作很有用，我们可以修改Nginx配置如下：

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

### Least Busy

我们知道轮询算法是把请求平均的转发给各个后端，使它们的负载大致相同。这有个前提，就是每个请求所占用的后端时间要差不多，如果对于某台Server有些请求占用的时间很长，而Round-Robin并不会感知这种情况，依旧轮训该Server，并将请求投递过去，则该台Server的负载会持续升高。

为了解决这个问题，Nginx提供了另一种基于最少链接(least_conn)的分配策略。这种策略会让Load Balancer感知每个server的负载情况，并把请求转发给连接数较少的后端，从而达到更好的负载均衡效果。

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

## Redundancy

最后来我们可以对Load Balancer进行冗余测试，观察当某个Server挂掉时，Load Balancer对请求的路由情况，测试步骤为：

1. 令PHP Server #1暂停运行
2. 令PHP Server #2暂停运行
3. 令PHP Server #1恢复运行
4. 令PHP Server #2恢复运行

{% include _partials/components/youtube.html param='https://www.youtube.com/embed/_ROsTIJkTyg' %}

上述视频中可观察到，当Server #1暂停运行后，Load Balancer将请求分配到第2,3台Server上，当Server #2暂停运行后，Load Balancer将请求分配到第三台Server上，当两台server恢复运行后，Load Balancer又重新回到Round Robin的分配策略。

{% include _partials/post-footer-1.html %}

### Resource

- [CS75 (Summer 2012) Lecture 9 Scalability Harvard Web Development David Malan](https://www.youtube.com/watch?v=-W9F__D3oY4&t=955s)
- [Round Robin](http://g33kinfo.com/info/archives/2657)
- [Using nginx as HTTP load balancer](http://nginx.org/en/docs/http/load_balancing.html)
- [NGINX Load Balancing - HTTP Load Balancer](https://docs.nginx.com/nginx/admin-guide/load-balancer/http-load-balancer/)
- [Module ngx_http_upstream_module](http://nginx.org/en/docs/http/ngx_http_upstream_module.html)

