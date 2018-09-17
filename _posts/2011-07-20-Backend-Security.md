---
updated: "2018-08-12"
layout: post
title: Web Security
list_title: 端到端通信（四）| Client Server Communiction - Security
---

### 同源策略

同源策略是指A网页设置的 Cookie，其它网页不能使用，除非这两个网页"同源"。所谓"同源"指的是"三个相同": <mark>协议相同，域名相同，端口相同</mark>。同源政策的目的，是为了保证用户信息的安全，防止恶意的网站窃取数据。

设想这样一种情况，用户在浏览A网站时，会读取用户登录Facebook留下的Cookie，然后拿着该Cookie去获取该用户的隐私数据等等。由此可见，"同源政策"是必需的，否则 Cookie 可以共享，互联网就毫无安全可言了。

在同源策略下，在当前网页发送AJAX请求到非同源页面的行为是禁止的，比如我们随便打开一个网页，向Google发送一个HTTP的GET请求，该请求将会被禁止：

```
Failed to load https://www.google.com/:
No 'Access-Control-Allow-Origin' header is present on the requested resource. 
```

### CORS(Cross Orign Resource Sharing)

CORS是一种可以规避掉同源策略的方案，在CORS之前，普遍使用的方式是JSONP

- CORS Header

另一种方式是直接在Header中标明

### CSRF(Cross-Site Request Forgery)

### XSS(Cross-site Scripting)


## Resources

- [Same Origin Policy](https://www.w3.org/Security/wiki/Same_Origin_Policy)