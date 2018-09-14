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

### Passport.js

Passport提供多种授权策略，这里以Google OAuth 2.0为例，首先初始化`passport`，配置请求策略

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
接下来在请求的中间件中中嵌入`passport`，假设请求的`url`为`/auth/google`

```javascript
app.get(
  '/auth/google',
  passport.authenticate('google', {
    scope: ['profile', 'email'] //该兴趣的用户信息
  })
);
```
这时我们测试请求的url，会得到下面的错误

```
The redirect URI in the request, http://127.0.0.1:5000/auth/google/callback, 
does not match the ones authorized for the OAuth client. To update the authorized redirect URIs
```




### Resources

- [Passport]()