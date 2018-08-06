---
updated: "2018-08-04"
layout: post
title: Backend Authentication
list_title: 用户认证 | Backend Authentication
categories: [Network,Backend]
---

### Cookies

a small piece of data stored in the browser for a website, key-value paires, 20 cookies per website

```
browser                 server
    |   --- post --->     |
    |                     |
    |    {uid = 1234}     |
    |   <--- cookie ---   |
    |                     |
    |    {uid = 1234}     |
    |   --- cookie --->   |
    |                     |
```
Cookies are sent in HTTP headers, something like this:

```
HTTP Response
--------------
set-cookie: user_id = 12345
set-cookie: last_seen = Dec 25 1985

HTTP Request
-------------
cookie: user_id = 12345; last-seen=Dec 25 1985
```

我们可以实际查看一下server返回的Cookie

```
➜  curl -I www.google.com
HTTP/1.1 200 OK
Date: Mon, 06 Aug 2018 02:08:03 GMT
Expires: -1
Cache-Control: private, max-age=0
Content-Type: text/html; charset=ISO-8859-1
P3P: CP="This is not a P3P policy! See g.co/p3phelp for more info."
Server: gws
X-XSS-Protection: 1; mode=block
X-Frame-Options: SAMEORIGIN
Set-Cookie: 1P_JAR=2018-08-06-02; expires=Wed, 05-Sep-2018 02:08:03 GMT; path=/; domain=.google.com
Set-Cookie: NID=136=CCO0I2iLHXevLQ9iTtMQcUOKDRmByXVIPRmrnW-6i6b59ZcCDiTASvbns6Dc9C7Sq39qg4wsF3WJ88zwNj2DC27dE2kshTSuE9KSsOsW00Xbhgnyn6ZY4QnJHdCEZNZc; expires=Tue, 05-Feb-2019 02:08:03 GMT; path=/; domain=.google.com; HttpOnly
Transfer-Encoding: chunked
Accept-Ranges: none
Vary: Accept-Encoding
```

- Cookie Domain

上面日志中可以看出Cookie的格式为

```
set-cookie: name=steve; Domain = www.reddit.com; Path = /foo
```








