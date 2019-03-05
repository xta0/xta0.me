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

> 如果有时间的话，这篇白皮书非常值得一读，它涵盖了Apple安全领域的各个方面，均是概要性质的介绍，读起来也不是很花时间

App签名属于App安全的范畴，它主要是为了防止恶意代码注入或者代码篡改，确保可执行程序一定是合规的。根据Apple在WWDC2016 Session 705的介绍，Apple早在10多年前就想到了这个问题，从而设计了这套App签名机制，不得不说这个设计确实很有先见之明

App签名对大多数iOS开发者来说都不陌生，但想搞清楚它的工作原理又不是很容易，加上众多冗余繁杂的证书，即使对于入行很久的iOS开发者来说，对代码加解密的过程也不一定能够说的清楚。纠其原因，大概有两点，一是绝大多数情况XCode将代码签名的工作自动化了，使开发人员感知不到其中的细节；二是代码签名所用到的知识本质上是密码学，需要理解非对称加密的原理，Hash函数，数字签名以及数字证书等一系列概念，这对于纯iOS开发者来说，多少有些门槛

但是我个人认为，作为iOS开发人员，这部分知识是一定要掌握的，它对于理解App的运行和系统底层的原理都有很大的好处，同时它也是Reverse Engineering的基础入门知识，在实际工程中，如果我们要配置CI，也需要用到这方面的知识。

接下来两篇文章我们将通过理论结合实践的方式来分析App代码签名的过程，这里假设你已经了解了非对称加密，公钥私钥等这些基本概念，如果这些概念还不清楚，可以参考之前介绍[HTTPs和SSL的文章](https://xta0.me/2011/07/10/Backend-HTTP.html)，另外，在下一篇文章中我们还将使用一个签名工具[mobdevim](https://github.com/DerekSelander/mobdevim)来分析开源的Wordpress App。

### 开发者证书

我们先从证书开始说，相信绝大多数的iOS开发者都有开发者证书，它是Apple颁发给你的代码签名的凭证，如下面这张图

<img src="{{site.baseurl}}/assets/images/2016/07/ios-app-sign-1.png" class="md-img-center">

如果了解SSL的工作原理，这个开发者证书可类比于(注意是类似而不是等同，因为开发者证书中还包含其它内容)数字证书，而Apple就是CA(Certifacate Authority)，我们拿到的开发证书实际上是Apple用自己的私钥加密过的，证书中有什么我们稍后讨论，参考《iOS Security WhitePaper》中可知，证书公钥保存在iPhone或者其它终端设备中。

> The Boot ROM code contains the Apple Root CA public key, which is used to verify that the iBoot bootloader is signed by Apple before allowing it to load

这也就是说当我们用其它的非法证书和App一起下发时，在App安装的过程中，iPhone会用系统中CA的公钥来校验随App下发的证书，如果校验失败则会不会进行安装。

接下来我们再看看这个证书里有什么，上图中可以看到证书中包含一把私钥，这个私钥是用来真正对代码签名的，那么对应的公钥在哪里呢？既然是非对称加密，那么公钥一定保存在Apple的Server端了，这点我们后面再解释。除了私钥，还有什么呢？我们可以通过`security`命令查看

```shell
security find-certificate -c "iPhone Developer: Tao Xu (Q7PV3L5FKY)" -p

-----BEGIN CERTIFICATE-----
MIIFizCCBHOgAwIBAgIIQbYxvc4mnecwDQYJKoZIhvcNAQELBQAwgZYxCzAJBgNV
BAYTAlVTMRMwEQYDVQQKDApBcHBsZSBJbmMuMSwwKgYDVQQLDCNBcHBsZSBXb3Js
...
fAcTLGucNU+mHD/9LGLlI/NJME2oW2QfCiy7XOUnjj/FG++Hirv026e07xIA2S3R
qkEDhYZScToVQlJNDVBCmgfQcuaDdt6lxVKW+awJIw==
-----END CERTIFICATE-----
```

上述命令是将证书按照x509标准的pem格式输出，begin和end之间是base64编码的字符串，这部分信息就是被Apple私钥加密过后的证书内容。

总结一下，关于开发者证书，我们需要明白下面两点：

1. 在App安装时证书本身会随着App下发，由于证书是要被验证真伪的，因此证书实际上是被Apple加密过的，解密的公钥就存放在iPhone或者iPad设备中，它在设备出厂的时候就被打到系统中了
2. 真正对我们代码进行签名或者说加密的是保存在开发者证书中的私钥，当App被上传到Appstore时，Apple会用对应 的公钥进行验证，看App内容是否被篡改

### Provisioning Profiles

上面我们曾提到当App被下载到手机时，证书也跟着下发，这里说的证书并不是开发者证书而是符合x509格式的数字证书，即begin和end中间那部分字符串，Provisioning Profile就是用来承载这部分内容的。Provisioning Profile除了包含数字证书，还包含entitlements文件和App支持的设备列表。Provisioning Profile文件默认的存放位置为

```shell
~/Library/MobileDevice/Provisioning Profiles

7693c059-56da-42c9-b38b-b8aee3c6ffdb.mobileprovision
7903d3a4-35b9-4def-81ed-42b6e036111b.mobileprovision
f8920973-a783-49ca-b4a1-cf455dbd0227.mobileprovision
```
我们可以通过下面命令查看`mobileprovision`文件的内容：

```shell
➜  Provisioning Profiles PP_FILE=$(ls ~/Library/MobileDevice/Provisioning\ Profiles/f8920973-a783-49ca-b4a1-cf455dbd0227.mobileprovision)
➜  Provisioning Profiles security cms -D -i "$PP_FILE"
```
结果为是一个XML格式的plist文件

```xml
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>AppIDName</key>
        ...
    <key>IsXcodeManaged</key>
    <key>DeveloperCertificates</key>
        <array>
            <data>MIIFjzCCBHegAwIBAgIICBy9Q3+oe/0wDQYJKoZIhvcNAQELBQAwgZYxCzAJBgNVBAYTAlVTMRMwEQYDVQQKDApBcHBsZSBJbmMuMSww...
            PPli1CckXUYjUENPj/h4vk0pD71LjWJ1HpoxGEPBO7uqxqGa5+g90wC9AAgjNSdt5PyjsRb6GZq7F+lAoN+1s+/uJ4WAAAbQQvvcdjaPlVWal/3JGIjvx4B8B3BrkMewMHQpEoVGiiM0=
            </data>
        </array>
    <key>Entitlements</key>
    <key>ExpirationDate</key>    
    <key>Name</key>
    <key>TeamIdentifier</key>
    <key>TeamName</key>
    <key>TimeToLive</key>
    <key>UUID</key>
    <key>Version</key>
</dict>
</plist>
```
可见这个文件包含的就是App元信息





### 签名的基本过程



### 小结

<img src="{{site.baseurl}}/assets/images/2016/07/ios-app-sign-2.png" class="md-img-center">

## Resource

- [iOS Security Whitepaper](https://www.apple.com/business/site/docs/iOS_Security_Guide.pdf)
- [WWDC2016 How iOS Security Really Works](https://developer.apple.com/videos/play/wwdc2016/705/)
- [iOS Code Signing Guide](https://developer.apple.com/library/archive/documentation/Security/Conceptual/CodeSigningGuide/Introduction/Introduction.html#//apple_ref/doc/uid/TP40005929-CH1-SW1)
- [Inside Code Signing](https://developer.apple.com/library/archive/documentation/Security/Conceptual/CodeSigningGuide/Introduction/Introduction.html#//apple_ref/doc/uid/TP40005929-CH1-SW1)
- [Advanced Apple Debugging and Reverse Engineering](https://store.raywenderlich.com/products/advanced-apple-debugging-and-reverse-engineering)
