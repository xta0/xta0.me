---
updated: "2018-08-04"
layout: post
title: Authentication
list_title: 端到端通信（三）| Client Server Communiction - Authentication
categories: [Network,Backend]
---

## Cookies

a small piece of data stored in the browser for a website, key-value paires, 20 cookies per website

```
browser                 Server
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

我们可以实际查看一下Server返回的Cookie

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
可以使用Hash来校验Cookie，假设cookie的格式为`{visits | hashcode}`其中visits表示访问Server的次数

```python
#Server
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

## Password

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
在密码的加密算法上，sha256比较慢，可以选择使用bcrypt。许多成熟的web framework均自带`bcrypt`方法。

## OAuth 认证

前面介绍了使用Email + Password的登录流程，除了这种登录方式以外，目前比较流行的还有使用第三方平台账号登录，以及使用JSON Web Token的方式。这一节先来介绍使用OAuth 2.0的登录方式，后面一节会介绍如何使用JSON Web Token。

OAuth是使用第三方平台账号进行认证，其认证流程图（假设使用Google账号登录）如下

{% include _partials/components/lightbox.html param='/assets/images/2008/07/oauth2-passport.png' param2='1' %}

上面的认证过程分为两部分，一部分是从Google获取登录信息，一部分是将得到的登录信息进行处理，生成cookie，存到自己的Server上。第一部分的任务流程相对标准化，基本上每个平台都有相应的三方库支持，比如Node.js可以使用`passport`(图中虚线部分)

`passport`由两部分构成，一部分是`passport`核心库，用来处理登录逻辑，另一部分是`passpoart strategy`用来支持各类登录策略，比如JWT，OAuth等等。上述两个framework分别为

```shell
npm install --save passport passport-google-oauth20
```

### Config Passport 

我们以Google的OAuth认证为例，介绍如何使用Passport完成OAuth流程。首先需要对`Passport`进行初始化：

```javascript
passport.use(
  new GoogleStrategy(
    {
      clientID: keys.googleClientID,
      clientSecret: keys.googleClientSecret,
      callbackURL: '/auth/google/callback'
    },
    accessToken => {
      console.log(accessToken);
    }
  )
);
```
上述代码初始化`passport`，传入`client_id`和`secret`以及`callbackURL`，注意`callbackURL`需要在Google Account中预先配置好，当OAuth请求成功后，Google会使用这个URL返回token。接下来，在Server上配置OAuth登录API:

```javascript
app.get(
  '/auth/google',
  passport.authenticate('google', {
    scope: ['profile', 'email'] //允许访问用户的profile和email信息
  })
);
```
此时，当我们访问`/auth/google`时，`passport`会将上述请求的URL替换为：

```
https://accounts.google.com/o/oauth2/v2/auth?response_type=code
&redirect_uri=http%3A%2F%2F127.0.0.1%3A5000%2Fauth%2Fgoogle%2Fcallback
&scope=profile%20email
&client_id=999661698345-ivms5t1s778qp5n4k55sep4odp7t4her.apps.googleusercontent.com
```
接下来我们还要定义Callback API处理回调，Passport内部通过url中是否存在`code`字段来区分该请求是否是callback请求

```javascript
//callback API
app.get('/auth/google/callback', passport.authenticate('google'));
```

此时，我们再访问`/auth/google`，选择一个账号登录，Google会通过callback URL返回登录token以及一些用户相关信息，用户信息中重要的是用户`id`，我们后面会用这个`id`来生成cookie

```javascript
//token
ya29.GlsgBiHIcv4rAhcYaxNkAn7nJadfm1oNQpCbnz1FO3QczeEke9zdWGc0ZExklr0b6WSJVQEuv_x6oe_cH5YAYrj9cNXeLWLjo3ATvZ_0pM0agI4_ju8-KxhUxIkhG

//user profile
{ id: '1128784079818168156265',
  displayName: 'JOHN DOE',
  name: { familyName: 'DOE', givenName: 'JOHN' },
  emails: [ { value: 'johndoe@gmail.com', type: 'account' } ],
}
```

### Cookie Management

通过OAuth拿到`token`和用户`id`之后，我们就可以为用户生成cookie，这个过程同样也是被`passport`在内部封装了，我们只需要在代码中做一些简单的配置即可。首先初始化`passport`中间件：

```javascript
//cookie
app.use(
  cookieSession({ //cookieSesssion
    maxAge: 30 * 24 * 60 * 60 * 1000,
    keys: [keys.cookieKey]
  })
);
app.use(passport.initialize());
app.use(passport.session());
```
`cookieSession`是一个用来从生成以及解析cookie的三方库，参考前面一节对cookie的介绍可知我们需要在Server上放一个私钥同来对cookie进行非对称加密解密。接下来当拿到Google返回的token后，需要根据用户信息来生成cookie

```javascript


```


当用户browser已经有cookie后，当用户再次访问Server时，将走下面的流程：

<img src="{{site.baseurl}}/assets/images/2011/07/cookie-1.png">



## JWT

JWT是JSON Web Token的缩写，它可以有效解决跨域问题，

## Resource

- [Intro to backend]()
- [Learn JSON Web Tokens](https://auth0.com/learn/json-web-tokens/)
