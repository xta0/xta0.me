---
title: HTTP/HTTPs工作原理
layout: post
list_title: HTTP/HTTPs工作原理 | How HTTP/HTTPs works
categories: [Backend, HTTP]
updated: '2016-09-14'
---

### Verbs

- **HEAD**

HEAD返回HTTP Response Header信息，这个操作主要用于查询Response大小，资源过期时间等等，通常在HEAD操作之后进行GET操作

```shell
➜  web git:(master) ✗ nc example.com 80
HEAD / HTTP/1.1
Host: example.com

HTTP/1.1 200 OK
Content-Encoding: gzip
Accept-Ranges: bytes
Cache-Control: max-age=604800 ## 缓存时间
Content-Type: text/html; charset=UTF-8
Date: Fri, 14 Sep 2016 04:59:20 GMT
Etag: "1541025663+ident"
Expires: Fri, 21 Sep 2016 04:59:20 GMT
Last-Modified: Fri, 09 Aug 2013 23:54:35 GMT
Server: ECS (dca/532C)
X-Cache: HIT
Content-Length: 606 ## 数据大小
```

- **OPTIONS**

OPTIONS用于查询Server支持的HTTP Verb，但并不是每个Server都支持OPTIONS操作

```shell
➜  web git:(master) ✗ nc example.com 80
OPTIONS / HTTP/1.1
Host: example.com

HTTP/1.1 200 OK
Allow: OPTIONS, GET, HEAD, POST ##支持verb类型
Cache-Control: max-age=604800
Content-Type: text/html; charset=UTF-8
Date: Fri, 14 Sep 2018 05:03:14 GMT
Expires: Fri, 21 Sep 2018 05:03:14 GMT
Server: EOS (vny006/044F)
Content-Length: 0
```

### Performance Basics

- Head of the line blocking 

当浏览器使用HTTP 1.x请求一个网页时，如果第一个GET请求阻塞了，则会并发最多六个线程完继续后面的请求。但是，这里会有一个性能问题，如果后续的5个线程在并发时仍要重新建立HTTP连接，则会造成较大的性能开销。因此HTTP 1.1引入了`keep-alive`，它允许client复用第一个连接的TCP链路，即使当第一个连接请求完成成后，如果后面5个请求所要传输的资源未完成，则该链路不会关闭，直到资源传输完成。

```shell
HTTP/1.1 302 Found
Location: https://facebook.com/
Content-Type: text/html; charset="utf-8"
X-FB-Debug: d9bIAXXQY7n1vitm38t6AexXGyCH/Tqa6acd2Dvs7C7/7EwZ3pcv5drbFhV6pwG+Tq5zfkVdfAQ1ajCTcVTGaA==
Date: Fri, 14 Sep 2016 05:21:42 GMT
Connection: keep-alive #开启keep alive
Content-Length: 0
```

## Security