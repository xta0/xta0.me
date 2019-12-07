---
updated: "2018-08-12"
layout: post
title: Web中的同源策略
list_title: Web中的同源策略
categories: [Web,Backend]
---

## 同源策略

同源策略是指A网页设置的 Cookie，其它网页不能使用，除非这两个网页"同源"。所谓"同源"指的是"三个相同": <mark>协议相同，域名相同，端口相同</mark>。同源政策的目的，是为了保证用户信息的安全，防止恶意的网站窃取数据。

设想这样一种情况，用户在浏览A网站时，会读取用户登录Facebook留下的Cookie，然后拿着该Cookie去获取该用户的隐私数据等等。由此可见，"同源政策"是必需的，否则 Cookie 可以共享，互联网就毫无安全可言了。
在同源策略下，在当前网页发送AJAX请求到非同源页面的行为是禁止的，比如我们随便打开一个网页，向Google发送一个HTTP的GET请求，该请求将会被禁止：

```shell
Failed to load https://www.google.com/:
No 'Access-Control-Allow-Origin' header is present on the requested resource. 
```

随着互联网的发展，"同源政策"越来越严格。目前，如果非同源，共有三种行为受到限制。

1. Cookie、LocalStorage 和 IndexDB 无法读取。
2. DOM 无法获得。
3. AJAX 请求不能发送。

虽然这些限制是必要的，但是有时很不方便，但有时合理的用途也受到影响，比如前后端分离后，前端页面调用后端的API就会遇到同源问题。

### CORS跨域通信的基本原理

CORS是一个W3C标准，全称是"跨域资源共享"（Cross-origin resource sharing）。它允许浏览器向非同源的地址发送请求，从而克服了AJAX请求只能同源的问题。但实际操作起来却没那么简单，要完成CORS跨域，前端后端均需要配合改动，具体做法是在Request和Response Header中添加一些跨域协商信息，有时在请求之前还会多出一次附加请求用于协商跨域，因此，实现CORS通信的关键是服务器，只要服务器实现了CORS接口，就可以跨源通信。

为了便于理解，我们在本机模拟一个跨域场景，假设客户端网页地址为`127.0.0.1:5500`，本地Server的地址为`127.0.0.1:9000`，客户端向服务端发送GET请求，如下图所示

<img class="md-img-center" src="{{site.baseurl}}/assets/images/2011/07/cors-1.png">

由于Server的端口号不同，参照上一节的规则，客户端的GET请求将会触发跨域规则，浏览器会Block这次请求，并给出下面信息：

```shell
Failed to load http://127.0.0.1:9000/: 
No 'Access-Control-Allow-Origin' header is present on the requested resource. 
Origin 'http://127.0.0.1:5500' is therefore not allowed access.
```

此时客户端和服务端的代码都很简单，没有做任何跨域相关的配置：

<div class="md-flex-h md-margin-bottom-24">
<div>
<pre class="highlight language-javascript md-no-padding-v md-height-full">
<code class="language-javascript">
//client
fetch('http://127.0.0.1:9000')
  .then(res => console.log(res))
  .catch(err => console.log(err));
</code>
</pre>
</div>
<div class="md-margin-left-12">
<pre class="highlight md-no-padding-v md-height-full">
<code class="language-python">
//server
const express = require('express');
const app = express();
app.get('/', (req, res) => {
  res.send('hello');
});
app.listen(9000);
</code>
</pre>
</div>
</div>

接下来，我们可以讨论并实践如何使用CORS完成跨域请求

### 两种请求

浏览器将CORS请求分成两类：简单请求（simple request）和非简单请求（not-so-simple request）。只要同时满足以下两大条件，就属于简单请求：

1. 请求方法是以下三种方法之一：
    - HEAD
    - GET
    - POST
2. HTTP的头信息不超出以下几种字段：
    - Accept
    - Accept-Language
    - Content-Language
    - Last-Event-ID
    - Content-Type：只限于三个值`application/x-www-form-urlencoded`、`multipart/form-data`、`text/plain`

凡是不同时满足上面两个条件，就属于非简单请求，浏览器对这两种请求的处理，是不一样的。

### 简单请求

对于简单请求，浏览器直接发出CORS请求。具体来说，当浏览器识别出该请求是跨域请求后，检查该请求是否满足跨域请求的条件，如果满足则会在头部增加一个`Origin`字段。上面例子中，浏览器发出的请求Header如下：

```shell
GET / HTTP/1.1
Host: 127.0.0.1:9000
Origin: http://127.0.0.1:5500
...
Connection: keep-alive
Pragma: no-cache
```
当服务端收到请求后，需要判断该`Origin`是否可被接受，如果可以接受则要在Response Header中告诉客户端，具体做法是在Header中增加`Access-Control-Allow-Origin`字段，并返回可接受的`Origin`。为了模拟这种情况，修改服务端代码如下：

```javascript
const express = require('express');
const app = express();
app.use((req, res, next) => {
  res.header('Access-Control-Allow-Origin', 'http://127.0.0.1:5500');
  next();
}
app.get('/', (req, res) => {
  res.send('hello');
});
app.listen(9000);
```
在Header中加入该字段后，浏览器可成功执行跨域请求，报错消失。重新观察Request和Response的Header信息，与CORS有关的字段有

- `Access-Control-Allow-Origin` 该字段是必须的。它的值要么是请求时Origin字段的值，要么是一个*，表示接受任意域名的请求。
- `Access-Control-Allow-Credentials` 该字段可选。它的值是一个布尔值，表示是否允许发送Cookie。默认情况下，Cookie不包括在CORS请求之中。设为true，即表示服务器明确许可，Cookie可以包含在请求中，一起发给服务器。这个值也只能设为true，如果服务器不要浏览器发送Cookie，删除该字段即可。

如果想要浏览器带上Cookie，只有`Access-Control-Allow-Credentials:true`是不够的，客户端这边也需要做一定的处理来告诉浏览器可以携带Cookie

```javascript
fetch('http://127.0.0.1:9000', {credentials: 'include'})
```
此时可以看到客户端请求的Header中包含了Cookie值。需要注意的是，如果要发送Cookie，`Access-Control-Allow-Origin`就不能设为星号，必须指定明确的、与请求网页一致的域名

### 非简单请求

非简单请求是那种对服务器有特殊要求的请求，比如请求方法是PUT或DELETE，或者`Content-Type`字段的类型是`application/json`。此时在正式通信之前，浏览器和Server之前会增加一次HTTP的OPTION请求，称为"preflight"。

我们继续修改上面的例子，在客户端请求的Header中增加`"Content-Type:application/json"`，此时客户端和Server在GET请求前会先发一次OPTIONS请求，交换跨域信息，Request header中多出了下面两个字段

```shell
Access-Control-Request-Headers: content-type
Access-Control-Request-Method: GET
```

- `Access-Control-Request-Method` 该字段是必须的，用来列出浏览器的CORS请求会用到哪些HTTP方法，上例是GET。
- `Access-Control-Request-Headers` 该字段是一个逗号分隔的字符串，指定浏览器CORS请求会额外发送的头信息字段，上例是`content-type`

如果要接受客户端的跨域请求，服务端也需要配合做相应的修改，其目的是告诉客户端自己可以接受哪些跨域操作：

```javascript
app.use((req, res, next) => {
  res.header('Access-Control-Allow-Origin', 'http://127.0.0.1:5500');
  res.header('Access-Control-Allow-Methods', 'GET');
  res.header('Access-Control-Allow-Headers', ' Content-Type');
  next();
});
```
- `Access-Control-Allow-Methods`
    该字段必需，它的值是逗号分隔的一个字符串，表明服务器支持的所有跨域请求的方法。注意，返回的是所有支持的方法，而不单是浏览器请求的那个方法。这是为了避免多次”Preflight"请求。
- `Access-Control-Allow-Headers`
    如果浏览器请求包括A`ccess-Control-Request-Headers`字段，则`Access-Control-Allow-Headers`字段是必需的。它也是一个逗号分隔的字符串，表明服务器支持的所有头信息字段，不限于浏览器在"预检"中请求的字段。

一旦服务器通过了"预检"请求，以后每次浏览器正常的CORS请求，就都跟简单请求一样，会有一个`Origin`头信息字段。服务器的回应，也都会有一个`Access-Control-Allow-Origin`头信息字段。在跨域信息沟通完成后，接下来客户端便可以向服务端发送GET请求来获取数据。

## CSRF/XSRF

CSRF指的是跨站点请求伪造(Cross-Site Request Forgery)，其本质是利用身份校验的漏洞进行攻击。考虑下面一个场景，用户Alice登录某银行网站后，进行转账操作，form表单地址为`somebank.com/transfer`，请求为POST，当银行server收到这个请求后，服务器会根据cookie验证该请求是否来自一个合法的session。假设有一个恶意用户Bob，了解该银行系统的API设计，于是他伪造了一个A的转账请求，将转账对象改为自己，但由于Bob无法得到Alice的cookie信息，进而服务端session校验失败，因此该伪造的请求并未成功。这时，Bob为了获取A的cookie信息，做了一个钓鱼网站，诱导用户Alice点击了某个button，随后便触发了一段脚本：

```javascript
function myFunction(event){
    const url = 'http://somebank.com/transfer';
    const data = "name=Bob&amount=1000";
    fetch(url,{
            method:"POST",
            headers:{
                "Content-Type":"application/x-www-form-urlencoded; charset=utf-8"
            },
            credentials: 'include',
            body:data
        })
}
```
该脚本会向银行系统发出转账请求，转账对象为`Bob`, 转账金额为`1000`。 大多数情况下，该请求会失败，因为他要求Alice的认证信息。但是，如果Alice次时恰巧刚访问他的银行后不久，他的浏览器与银行网站之间的 session 尚未过期，浏览器的cookie之中含有Alice的认证信息。此时，上述POST请求很容易通过银行的校验，从而完成对Bob的转账。等以后Alice发现账户钱少了，即使他去银行查询日志，他也只能发现确实有一个来自于他本人的合法请求转移了资金，没有任何被攻击的痕迹。

透过例子能够看出，攻击者并不能通过CSRF攻击来直接获取用户的账户控制权，也不能直接窃取用户的任何信息。他们能做到的，是骗用户浏览器来获取用户cookie，模拟用户操作。

### 检查Referer字段

HTTP头中有一个`Referer`字段，这个字段用以标明请求来源于哪个地址。在处理敏感数据请求时，通常来说，`Referer`字段应和请求的地址位于同一域名下。以上文银行操作为例，`Referer`字段地址通常应该是转账按钮所在的网页地址，应该也位于`somebank.com`之下。而如果是CSRF攻击传来的请求，`Referer`字段会是包含恶意网址的地址，不会位于`somebank.com`之下，这时候服务器就能识别出恶意的访问。

然而，这种方法并非万无一失。Referer 的值是由浏览器提供的，虽然 HTTP 协议上有明确的要求，但是每个浏览器对于 Referer 的具体实现可能有差别，并不能保证浏览器自身没有安全漏洞。使用验证 Referer 值的方法，就是把安全性都依赖于第三方（即浏览器）来保障，从理论上来讲，这样并不安全。

即便是使用最新的浏览器，黑客无法篡改 Referer 值，这种方法仍然有问题。因为 Referer 值会记录下用户的访问来源，有些用户认为这样会侵犯到他们自己的隐私权，特别是有些组织担心 Referer 值会把组织内网中的某些信息泄露到外网中。因此，用户自己可以设置浏览器使其在发送请求时不再提供 Referer。当他们正常访问银行网站时，网站会因为请求没有 Referer 值而认为是 CSRF 攻击，拒绝合法用户的访问。

### 使用Token

另一个防御措施是改变用户的校验规则，CSRF 攻击之所以能够成功，是因为黑客可以完全伪造用户的请求，该请求中所有的用户验证信息都是存在于 cookie 中，因此黑客可以在不知道这些验证信息的情况下直接利用用户自己的 cookie 来通过安全验证。要抵御 CSRF，关键在于在请求中放入黑客所不能伪造的信息，并且该信息不存在于 cookie 之中。可以在 HTTP 请求中以参数的形式加入一个随机产生的 token，并在服务器端建立一个拦截器来验证这个 token，如果请求中没有 token 或者 token 内容不正确，则认为可能是 CSRF 攻击而拒绝该请求。


### XSS

XSS(Cross-site Scripting)是一种代码注入技术，恶意攻击者往Web页面里插入恶意javaScript代码，当用户浏览该页之时，嵌入其中Web里面的javaScript代码会被执行，从而达到恶意攻击用户的目的。


## Password

最后讨论一下如何存储Password的问题，保存用户密码的策略和上一篇文章中介绍计算Cookie的策略类似，都是使用哈希函数，对明文密码 + 一个随机数（salt）进行hash

```python
import random
import string

def make_salt():
    seed = "1234567890abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ!@#$%^&*()_+=-"
    sa = []
    salt=''
    for i in range(5):
        sa.append(random.choice(seed))
    salt = ''.join(sa)
    return salt

>>> print(make_salt())
vZMV1
```
上述代码可以生成5个字符的随机字符串, 用该字符串对密码进行加密，密码加密以及用户登录校验密码的逻辑如下

```python
def make_pwd_hash(name,pwd,salt=None):
    if not salt:
        salt = make_salt()
    key = (name+pwd+salt).encode('utf-8')
    hash_code = hashlib.sha256(key).hexdigest()
    return f"{hash_code},{salt}"

#check user password
#hash_code comes from database
def valid_pw(name,pw,hash_code):
    salt = h.split(',')[1]
    return h == make_pwd_hash(name,pwd,salt)
```
在密码的加密算法上，sha256比较慢，可以选择使用bcrypt。许多成熟的web framework均自带`bcrypt`方法。

## Resources

- [Same Origin Policy](https://www.w3.org/Security/wiki/Same_Origin_Policy)
- [浏览器同源政策及其规避方法](http://www.ruanyifeng.com/blog/2016/04/same-origin-policy.html)
- [跨域资源共享CORS详解](http://www.ruanyifeng.com/blog/2016/04/cors.html)
- [CSRF 攻击的应对之道](https://www.ibm.com/developerworks/cn/web/1102_niugang_csrf/index.html)
- [Learn JSON Web Tokens](https://auth0.com/learn/json-web-tokens/)
