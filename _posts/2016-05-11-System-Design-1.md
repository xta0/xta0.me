---
updated: "2018-08-20"
layout: post
title: Load Balancer
list_title: 系统设计 | System Design | 负载均衡 | Load Balancer
categories: [backend]
---

### Motavition

我对这个问题一直很感兴趣，但苦于一直没有实操的机会，虽然在阿里工作，但长期专注客户端，后端的scale问题基本接触不到，而对于大部分做业务的后端的工程师也接触不到这类问题。在阿里，这种基础性的问题在很早就被架构组的人解决了，而这些人目前在不在公司都不知道了，资料也很少，大部分在内网分享的系统设计都是建立在已经比较完善的底层服务之上了，要么做一些中间件服务，要么做一些性能调优之类的工作。因此想从头搞清楚阿里是如何Scale的基本上是找不到头绪的。

恰好最近在Youtube上看了Redit创始人Huffman在Udacity上分享的一些课程，其中有一部分讲到了Reddit是如何从一个单机Server Scale到千万用户量级的，听下来收获很大，但毕竟不是做后端出身，对于他讲的大部分内容在原理上是可以理解的，但涉及到的开源技术由于没有实操经验，一时间无法完全消化。因此，我决定先从方法论入手，搞清楚原理，然后有机会再动手实操演练


## Load Balancer

负载均衡这个概念相对来说比较容易理解，它主要完成两个任务，一是负责分发请求，二是负责冗余，即某个server failed了，能够及时发现并redirect请求到其它server上。


### Nginx 

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
上述配置中，我们令负载均衡Server监听8088端口，当有请求到来时，转发到三台`php_servers`上。我用一段脚本来模拟请求，假设每隔0.5s向负载均衡Server上发一个请求，观察分配结果

{% include _partials/components/lightbox-center.html param='/assets/images/2015/05/lb-2.png' param2='1' %}

我们看到Load Balancer会依次将request投递到三台服务器上，从输出来看，Ngixn默认的策略为**Round Robin**。

接下来我们可以进行冗余测试，测试步骤为

1. 令PHP Server #1暂停运行
2. 令PHP Server #2暂停运行
3. 令PHP Server #1恢复运行
4. 令PHP Server #2恢复运行

{% include _partials/components/youtube.html param='https://www.youtube.com/embed/_ROsTIJkTyg' %}

除了Round Robin外，Nginx还支持其它的附在均衡策略，比如使用`ip hash`。所谓`ip hash`是指将所有的request均绑投递到某一个server上，如果该server挂掉，则自动切换到其它server上，这个特性对于维护登录信息之类的操作很有用，我们可以修改Nginx配置如下：

```yaml
upstream php_servers{
    ip_hash;
    server localhost: 10001;
    server localhost: 10002;
    server localhost: 10003;
}
```

{% include _partials/components/lightbox-center.html param='/assets/images/2015/05/lb-3.png' param2='1' %}

我们看到所有的server均投递到PHP Server 1上，如果此时该Server挂掉，Nginx则会自动将请求转向Server 2。

Nginx提供的另一种策略是基于流量分配，我们假设Server #1此时流量升高，响应时间变慢，我们用`sleep(20)`来模拟这种情况，修改Server #1的代码为

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

{% include _partials/post-footer.html %}

### Resource

- [CS75 (Summer 2012) Lecture 9 Scalability Harvard Web Development David Malan](https://www.youtube.com/watch?v=-W9F__D3oY4&t=955s)
- [Using nginx as HTTP load balancer](http://nginx.org/en/docs/http/load_balancing.html)
- [NGINX Load Balancing - HTTP Load Balancer](https://docs.nginx.com/nginx/admin-guide/load-balancer/http-load-balancer/)
- [Module ngx_http_upstream_module](http://nginx.org/en/docs/http/ngx_http_upstream_module.html)

