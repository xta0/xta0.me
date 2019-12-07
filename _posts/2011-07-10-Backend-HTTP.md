---
title: HTTPs与非对称加
layout: post
list_title: HTTPs与非对称加密 | Asymmetric Encryption and HTTPs
categories: [Backend, HTTPs]
updated: '2018-09-14'
---

### HTTP Verbs

- **GET/POST**
    - Parameters in URL / Parameters in Body
    - used for fetching documents / used for updating
    - maximum URL length / no max length
    - Ok to cache / not OK to cache
    - shouldn't change the server / OK to change the server

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

> 由于GET，POST，PUT和DELETE操作相对来说比较好理解，这里忽略对他们的介绍

### HOL (Head of the line blocking)


当浏览器使用HTTP 1.x请求一个网页时，如果第一个GET请求阻塞了，则会并发最多**六个**线程完继续后面的请求。但是，这里会有一个性能问题，原因是后续的无请求在并发时仍要重新建立HTTP连接，这会造成较大的性能开销。

为了解决这个问题，HTTP 1.1引入了`keep-alive`机制，它允许后面请求复用第一个连接的TCP链路，即使当第一个连接请求完成成后，如果后面五个请求所要传输的资源未完成，则该链路不会关闭，直到资源传输完成。

```shell
HTTP/1.1 302 Found
Location: https://facebook.com/
Content-Type: text/html; charset="utf-8"
X-FB-Debug: d9bIAXXQY7n1vitm38t6AexXGyCH/Tqa6acd2Dvs7C7/7EwZ3pcv5drbFhV6pwG+Tq5zfkVdfAQ1ajCTcVTGaA==
Date: Fri, 14 Sep 2016 05:21:42 GMT
Connection: keep-alive #开启keep alive
Content-Length: 0
```

### HTTP/2 Improvements

针对HOL的问题，仅有`keep-alive`机制还不能解决根本问题。考虑到目前Web请求数量激增，以及传输效率问题，HTTP/2 提供了一系列优化措施，包括

1. Binary Protol 传输二进制而不是plain/text
2. Compressed Header 头部压缩，重复信息复用
3. Persistent Connections 短链改长链，减少频繁建立短连接的开销
4. Multiplex Streaming 多路复用，合并资源请求
5. Server Push 支持push

## HTTPs

{% include _partials/components/lightbox.html param='/assets/images/2008/07/tls.png' param2='1' %}

HTTPs是HTTP + TLS的简写，TLS是用来对HTTP传输的内容进行加密的协议，它工作在应用层以下，传输层以上。要搞清楚TLS的工作方式，需要先搞清楚Hash，数字签名，数字证书，非对称加密等概念。接下来我们便逐一介绍这些名词。

### 非对称加密

所谓非对称加密是指对数据的加密和解密用所用的秘钥不同（公钥加密，私钥解密）。假设Alice要发送一短消息给Bob，Alice首先从Bob那里获得了一个加密消息用的公钥`public_key`，接下来，Alice用`public_key`对消息进行加密后得到了一串密文`12xEm6U`，当Bob收到密文后，用自己的私钥`private_key`进行解密，得到了原文，整个流程如下图所示：

```
Alice: "msg" --public_key("msg")-->  "12xEm6U"
--------------------------------------|------
                                      |
--------------------------------------|------
 Bob: "msg" <--private_key("msg")-- "12xEm6U"
```                                   

在上面的流程中，即使有人在中间窃听得到信息`12xEm6U`，但是由于没有`private_key`，因此也无法解密。非对称加密的方式有很多种，常用的有RSA，RSA的加密方式主要基于对大数的质因数分解，目前看起来还是很难被暴利破解。

### Hash函数

Hash函数用来对字符串进行单向加密，常用的Hash函数有`sha256,sha512,md5`等，Hash函数有两个特点，第一个特点是不可逆，第二个特点是计算结果定长，比如不论输入字符有多长，`sha256`的哈希值永远是256个字符。通常来说，相同的文本会产生相同的Hash值，但由于第二个特性的存在，会导致哈希碰撞，即两个内容不同的文本却可能得到相同的哈希值，因此不能根据Hash值来判断原文本是否相同。

### 数字签名

上面提到的非对称加密方式虽然可以解决信息不被泄漏，但是无法避免另一个问题——信息可能会被“掉包”。我们假设有中间人Doug，拦截了Alice和Bob的信息，虽然Doug无法知道`12xEm6U`的含义，但是他却可以使用Alice的公钥来加密一份恶意数据`evil`来掉包Alice原先的`msg`，而当Bob收到加密信息`3xgiC0Z`后，它并不知道该信息已被掉包，从而受到来自Doug攻击。

```
Alice: "msg" --public_key("msg")--> "12xEm6U"
--------------------------------------|---
                                      |
Doug: "evil" --public_key("evil")--> "3xgiC0Z"
                                      |
--------------------------------------|---
Bob: "evil" <--private_key("msg")-- "3xgiC0Z"
```

当Bob发现自己被骗后，他立刻明白这种通信方式无法核实信息的来源，于是他决定设计一种数字签名来解决这个问题，具体做法如下:

1. 制作数字签名：
    - Bob将要传输给Alice的内容`msg`进行哈希运算得到`e46b3f`（digest）
    - Bob用自己的私钥对digest进行加密得到自己的签名signature
2. 生成传输文本
    - Bob将signature附在原传输的文本下面，得到`["msg",signature]`
    - 发送给Alice

当Alice收到消息后，她需要首先验证消息的来源，具体做法如下

1. 检查signature
    - Alice使用公钥解密signature，如果能解密，说明来源肯定是Bob
    - Alice将解密后的结果`e46b3f`保存起来
2. 核实内容
    - Alice用Hash函数对`msg`进行哈希运算
    - 将得到的结果和`e46b3f`比对，如果相同则表示内容没有被篡改过

上述流程如下：

```
Bob -- hash("msg") = "e46b3f" --> private_key("e46b3f") --> signature --> ["msg",signature]
------------------------------------------------------------------------------------|------
                                                                                    |
------------------------------------------------------------------------------------|------
Alice <-- hash("msg") == "e46b3f"? <-- publicy_key(signature) <-- signature <-- ["msg",signature]
```

假如此时中间人Doug再次拦截了Alice与Bob的通信，虽然Doug可以用公钥解开signature窃取通信信息，但是他却没有办法再将篡改后的信息发送给Alice，原因是他没有Bob的私钥，无法对该信息进行加密。此时Doug有两种选择：

1. 篡改`msg`，signature不动，将该信息发给Alice
2. 用公钥解密signature，篡改`msg`，然后再用自己的私钥加密后发给Alice

针对第一种情况，当Alice用公钥解开signature后，会得到`e46b3f`，此时由于Doug篡改了`msg`，使得`hash("msg")!="e46b3f"`校验失败，Alice知道该信息被篡改。针对第二种情况，当Alice试图用Bob的公钥解密signature时，发现无法解密，则Alice知道该信息被并非来自Bob，可能被"掉包"了。

上述流程，其安全的核心在于Bob用自己的私钥进行了加密，即使Bob发出的信息被拦截，篡改，但是由于中间人没有Bob私钥，无法伪造Signature。因此，对于数字签名来说，保护好私钥是最重要的。

### 数字证书

但是上述过程仍有一点瑕疵，假设Doug通过某种方式将Alice电脑上的公钥换成了自己的公钥，然后在拦截下Bob的请求后，Doug首先篡改了`msg`为`evil`，然后用哈希函数算出了`evil`的哈希值为`dx90yf`，最后用自己的私钥对`dx90yf`加密来伪造签名。此时由于Alice的公钥被Doug更换了，因此Alice在收到消息后，用Dogg的公钥可以正常解密，从而再次受到Dogg的攻击。

```
Bob -- hash("msg") = "e46b3f" --> private_key_bob("e46b3f") --> signature_bob --> ["msg",signature_bob]
-----------------------------------------------------------------------------------------------|--------
                                                                                               |
Doug -- hash("evil") = "dx90yf" --> private_key_dogg("dx90yf") --> signature_doug --> ["evil",signature_doug]
                                                                                               |
-----------------------------------------------------------------------------------------------|---------
Alice <-- hash("evil") == "dx90yf"? <-- publicy_key_dogg(signature) <-- signature_doug <-- ["evil",signature_doug]
```

Alice此时发现她被攻击的原因在于她无法知道自己公钥的真伪，她希望能够有一个第三方中介机构（certificate authority，简称CA）可以为她的公钥出具合法的证明，于是Alice让Bob去CA对公钥做认证，认证的方式为：CA用自己的私钥对Bob的公钥以及一些其它信息（比如CA的名称等）进行加密，生成一份**数字证书（Digital Certificate）**。

当Bob得到数字证书后，他在每次和Alice通信之前，Alice都会让Bob先发送证书，Alice收到证书后，会用自己的保存的CA的公钥进行解密，如果可以解密说则明这份证书有效，证书内的公钥是可信的。

```
Bob  -->  public_key_bob --> private_key_CA(public_key_bob) --> Certificate 
---------------------------------------------------------------------|--
                                                                     |
---------------------------------------------------------------------|--
Alice      <-- public_key_bob <-- public_key_CA(Certiface)? <-- Certificate
```
当Alice知道公钥是可信的之后，再通知Bob校验成功，可以发送消息了。

### TLS的工作方式

如果明白了上面介绍的各部分工作的原理，就基本上可以搞清楚TLS的工作方式了。我们可以将上面的Alice比作客户端（用户浏览器），将Bob比作服务端。整个TLS的通信流程分为两部分，第一部分是握手阶段，采用非对称加密方式，第二阶段为通信阶段，采用对称加密方式。

{% include _partials/components/lightbox.html param='/assets/images/2008/07/ssl-handshake.png' param2='1' %}

Alice向Bob索要证书的过程称为TLS的握手阶段，握手流程如下：

1. Alice通知Bob自己支持的TLS版本，支持的加密方式等信息，同时Alice还生成了一个随机数（client random），用于加密后面的对话
2. Bob收到消息后，确认TLS通信版本，并给出数字证书，同时生成一个随机数（server random）
3. Alice收到证书后，确认证书有效，然后生成一个新的随机数（Premaster secret），并使用数字证书中的公钥（Bob的公钥），加密这个随机数，发给Bob
4. Bob使用自己的私钥，获取爱丽丝发来的随机数（Premaster secret）
5. Alice和Bob根据约定的加密方法，使用前面的三个随机数，生成"对话密钥"（session key），用来加密接下来的整个对话过程。

握手阶段完成后，第一阶段的非对称加密过程就结束了，后面Alice和Bob的通信将使用对称加密的方式，两人均使用上面得到的`session key`加密通信内容。


## Resources 

- [Rising number of requests](https://httparchive.org/reports/state-of-the-web)
- [HPACK - Header Compression for HTTP/2](https://http2.github.io/http2-spec/compression.html)
- [Current support of HTTP/2 from caniuse.com](https://http2.github.io/http2-spec/compression.html)
- [数字签名是什么？](http://www.ruanyifeng.com/blog/2011/08/what_is_a_digital_signature.html)
- [What is a Digital Signature?](http://www.youdzone.com/signature.html)
- [SSL Handshakes and HTTPs Bindings on IIS](https://blogs.msdn.microsoft.com/kaushal/2013/08/02/ssl-handshake-and-https-bindings-on-iis/)


