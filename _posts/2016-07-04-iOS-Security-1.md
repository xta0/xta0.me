---
layout: post
list_title:  iOS的签名原理（一） | App Signing in iOS Part 1
title: App的签名原理（一）
categories: [iOS]
---

App签名属于iOS的安全控制范畴，根据iOS的安全白皮书的序言部分，整套iOS安全体系非常的复杂，分为下面几个主要部分：
- 系统安全，针对Apple的硬件平台，比如iPhone，iPad等
- 数据加密与数据安全
- App安全，主要针对应用签名，确保App内容不被篡改
- 网络安全，针对网络协议与传输数据的加密
- Apple Pay支付安全
- 互联网服务安全，针对Apple自己云服务的安全，包括消息的同步，备份等
- 密码管理
- 设备管控
- 隐私管控，包括用户数据或者位置信息数据等




## Resource

- [iOS Security Whitepaper](https://www.apple.com/business/site/docs/iOS_Security_Guide.pdf)
- [How iOS Security Really Works](https://developer.apple.com/videos/play/wwdc2016/705/)
- [iOS Code Signing Guide](https://developer.apple.com/library/archive/documentation/Security/Conceptual/CodeSigningGuide/Introduction/Introduction.html#//apple_ref/doc/uid/TP40005929-CH1-SW1)
- [Inside Code Signing](https://developer.apple.com/library/archive/documentation/Security/Conceptual/CodeSigningGuide/Introduction/Introduction.html#//apple_ref/doc/uid/TP40005929-CH1-SW1)
