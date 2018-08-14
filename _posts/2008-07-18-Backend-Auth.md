---
updated: "2018-08-04"
layout: post
title: Cookies
list_title: Backend用户认证 | Cookies
categories: [Network,Backend]
---

## Cookies

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

### Hash Cookies

常用的Hash函数有，CRC,MD5,Sha1,Sha256，其安全性由低到高，Hash的速度由快到慢。Python提供了一些常用的Hash API

```python
import hashlib
key = "udacity"
key = key.encode('utf-8')
x = hashlib.sha256(key)
x.hexdigest() #
```
可以使用Hash来校验Cookie，假设cookie的格式为`{visits | hashcode}`其中visits表示访问server的次数

```python
#server
#-----------
#set-cookie: {5|e4da3b7fbbce2345d7772b0674a318d5}

#client
#-----------
#cookie: {5|e4da3b7fbbce2345d7772b0674a318d5}

# check cookie 
def hash_str(s):
    return hashlib.md5(s.encode('utf-8')).hexdigest()

def check_secure_val(h):
    val, hashstr = h.split('|')
    if hashstr == hash_str(val):
        return True
    else:
        return False
```

上述Hash Cookie的方法任然有一个缺陷，就是可以被伪造，例如可以将`{5|e4da3b7fbbce2345d7772b0674a318d5}`替换为`{123|202cb962ac59075b964b07152d234b70}`，校验仍然有效。

可以对上述方法做个修改，在计算hash时，加入一个secret key

```
Hash(secret_key,cookie) = hash_code
```

Python的hash库中提供HMAC(Hash-based Meesage Authentication Code)的API来应对上述场景

```python
secret_key = bytes([0x13, 0x00, 0x00, 0x00, 0x08, 0x00])
cookie = "some_cookie".encode('utf-8')
hash_code = hmac.new(secret_key,cookie).hexdigest()
#f2b280549c1c9edb18d5500d6c01ea51
```

### Password

Hash password的方法和cookie类似，对明文密码 + 一个随机数（salt）进行hash

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
在密码的加密算法上，sha256比较慢，可以选择使用bcrypt

<p class="md-center-p">（）</p>