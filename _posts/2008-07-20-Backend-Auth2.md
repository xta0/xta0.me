---
updated: "2018-08-12"
layout: post
title: OAuth认证机制
list_title: Backend用户认证 | OAuth
---

## OAuth2

OAuth 2.0第三方认证流程图如下

{% include _partials/components/lightbox.html param='/assets/images/2008/07/oauth2-passport.png' param2='1' %}

如果使用Node.js，可以使用`passport`帮助我们完成绝大部分的工作，`passport`由两部分构成，一部分是`passport`核心库，用来处理登录逻辑，另一部分是`passpoart strategy`用来支持各类登录策略，比如JWT，OAuth等等。上述两个framework分别为

```shell
npm install --save passport passport-google-oauth20
```




### Resources

- [Passport]()