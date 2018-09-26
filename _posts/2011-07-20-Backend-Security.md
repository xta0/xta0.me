---
updated: "2018-08-12"
layout: post
title: Web Security
list_title: 端到端通信（四）| Client Server Communiction - Security
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

虽然这些限制是必要的，但是有时很不方便，但有时合理的用途也受到影响，比如前后端分离后，前端调用页面后端的API就会遇到同源问题。

### CORS

CORS是一个W3C标准，全称是"跨域资源共享"（Cross-origin resource sharing）。它允许浏览器向非同源的地址发送请求，从而克服了AJAX请求只能同源的问题。但实际操作起来却没那么简单，要完成CORS跨域，前端后端均需要配合改动，具体做法是在Request和Response Header中添加一些跨域协商信息，有时在请求之前还会多出一次附加请求用于协商跨域，因此，实现CORS通信的关键是服务器，只要服务器实现了CORS接口，就可以跨源通信。

为了便于理解，我们在本机模拟一个跨域场景，假设客户端网页地址为`127.0.0.1:5500`，本地Server的地址为`127.0.0.1:9000`，客户端向服务端发送GET请求，如下图所示

<img class="md-img-center" src="{{site.baseurl}}/assets/images/2011/07/cors-1.png">

由于Server的端口号不同，参照上一节的规则，客户端的调用请求将会触发跨域规则，浏览器会Block这次请求，并给出下面信息：

```shell
Failed to load http://127.0.0.1:9000/: No 'Access-Control-Allow-Origin' header is present on the requested resource. 
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


### 简单请求

浏览器将CORS请求分成两类：简单请求（simple request）和非简单请求（not-so-simple request）。只要同时满足以下两大条件，就属于简单请求。凡是不同时满足上面两个条件，就属于非简单请求。

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



对于简单请求，浏览器直接发出CORS请求。具体来说，就是在头信息之中，增加一个`Origin`字段。
- CORS Header

另一种方式是直接在Header中标明

## Web攻击

### CSRF(Cross-Site Request Forgery)

### XSS(Cross-site Scripting)

### SQL注入

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
- [Learn JSON Web Tokens](https://auth0.com/learn/json-web-tokens/)
