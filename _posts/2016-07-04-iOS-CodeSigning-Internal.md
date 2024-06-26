---
layout: post
updated: "2019-06-20"
list_title:  iOS App的签名机制 | iOS Signing under the hood
title: iOS App的签名机制
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

App签名对大多数iOS开发者来说都不陌生，但想搞清楚它的工作原理又不是很容易，加上众多冗余繁杂的概念和各式各样的证书，即使对于入行很久的iOS开发者来说，对App签名的过程也不一定能够说的清楚。纠其原因，大概有两点，一是绝大多数情况XCode将代码签名的工作自动化了，使开发人员感知不到其中的细节；二是代码签名所用到的知识本质上是密码学，需要理解非对称加密的原理，Hash函数，数字签名以及数字证书等一系列概念，这对于纯iOS开发者来说，多少有些门槛

但是我个人认为，作为iOS开发人员，这部分知识是一定要掌握的，它对于理解App的运行和系统底层的原理都有很大的好处，同时它也是Reverse Engineering的基础入门知识，在实际工程中，如果我们要配置CI，也需要用到这方面的知识。

这里假设你已经了解了公钥私钥，数字签名，数字证书等非对称加密的基本概念，如果这些概念还不清楚，可以参考之前介绍[HTTPs和SSL的文章](https://xta0.me/2011/07/10/Backend-HTTPs.html)。为了帮助回忆这些概念，我们还是参考下面这张图

<img src="{{site.baseurl}}/assets/images/2016/07/ios-app-sign-2.png" class="md-img-center">

### 开发者证书

我们先从Apple的开发者证书开始说，相信绝大多数的iOS开发者都有开发者证书，它是Apple颁发给你的代码签名的凭证，如下面这张图

<img src="{{site.baseurl}}/assets/images/2016/07/ios-app-sign-1.png" class="md-img-center">

一个开发者证书中包含了一对本地的公钥，私钥，和一张数字证书，而Apple就是颁发证书的CA(Certifacate Authority)。我们知道数字证书实际上就是本地公钥被CA的私钥加密后的结果，因此<mark>我们拿到的开发证书实际上是Apple用自己的私钥加密过的<mark>，参考《iOS Security WhitePaper》中可知，CA的公钥保存在Apple的设备中，比如iPhone。

> The Boot ROM code contains the Apple Root CA public key, which is used to verify that the iBoot bootloader is signed by Apple before allowing it to load

这也就是说当我们用其它的非法证书和App一起下发时，在App安装的过程中，iPhone会用系统中CA的公钥来校验随App下发的证书，如果校验失败则会不会进行安装。注意这一步验证的是证书是否有效，而不是代码是否有效。如何验证代码是否被篡改，需要用到数字签名，我们接下来会提到。我们可以用下面命令来查看当前机器上有效的证书

```shell
> security find-identity -v -p codesigning
```

接下来我们再看看这个证书里有什么，上图中可以看到证书中包含一把私钥，这个私钥是用来真正对代码签名的，那么对应的公钥在哪里呢？既然是非对称加密，那么公钥一定保存在Apple的Server端了，这是非对称加密的基本策略。实际上在我们向Apple请求证书的时候，和私钥对应的公钥会被上传到Apple的后台，用来生成数字证书。除了私钥，还有什么呢？我们可以通过`security`命令查看

```shell
security find-certificate -c "iPhone Developer: xx (xxxx)" -p

-----BEGIN CERTIFICATE-----
MIIFizCCBHOgAwIBAgIIQbYxvc4mnecwDQYJKoZIhvcNAQELBQAwgZYxCzAJBgNV
BAYTAlVTMRMwEQYDVQQKDApBcHBsZSBJbmMuMSwwKgYDVQQLDCNBcHBsZSBXb3Js
...
fAcTLGucNU+mHD/9LGLlI/NJME2oW2QfCiy7XOUnjj/FG++Hirv026e07xIA2S3R
qkEDhYZScToVQlJNDVBCmgfQcuaDdt6lxVKW+awJIw==
-----END CERTIFICATE-----
```
上面begin和end之间是一段经过base64编码的字符串，这部分信息就是被Apple（CA）私钥加密过后的**数字证书**，其格式为x509标准的pem。想要提取证书中里的公钥，我们可以先用base64 decode，再用openssl解密

```shell
> CERT_DATA=MIIFizCCBHOgAwIBAgIIQbYx...
> echo "$CERT_DATA" | base64 -D > ./cert.cer
> openssl x509 -in ./cert.cer -inform DER -text -noout

Certificate:
Data:
Version: 3 (0x2)
Serial Number: 786948871528664923 (0xaebcdd447dc4f5b)
Signature Algorithm: sha256WithRSAEncryption
Issuer: C=US, O=Apple Inc., OU=Apple Worldwide Developer
Relations, CN=Apple Worldwide Developer Relations Certification Authority
Validity
Not Before: Jan 17 13:26:41 2016 GMT
Not After : Jan 17 13:26:41 2017 GMT
Subject: UID=PZYM8XX95Q, CN=iPhone Distribution: Automattic, Inc.
(PZYM8XX95Q), OU=PZYM8XX95Q,
...truncated...
```

总结一下，关于开发者证书，我们需要明白下面两点：

1. 在App安装时证书本身会随着App下发，并且该证书是被Apple加密过的，解密的公钥就存放在Apple的设备中，它在设备出厂的时候就被预置到文件系统中了，这一步是非对称加密，发生App安装时，目的是验证开发证书的真伪
2. 真正对我们代码进行签名或者说加密的是保存在开发者证书中的私钥，当App被上传到Appstore时，Apple会用对应的公钥进行验证，这一步也是非对称加密，发生在App Store，其目的是确保该程序是合法的，没有被篡改 

### Provisioning Profiles

上面我们曾提到当App被下载到手机时，证书也跟着下发，这里说的证书并不是开发者证书而是符合x509格式的数字证书，即begin和end中间那部分字符串，Provisioning Profile就是用来承载这部分内容的。Provisioning Profile除了包含数字证书，还包含entitlements文件和App支持的设备列表等一些App的元信息。Provisioning Profile文件默认的存放位置为

```shell
~/Library/MobileDevice/Provisioning Profiles

7693c059-56da-42c9-b38b-b8aee3c6ffdb.mobileprovision
7903d3a4-35b9-4def-81ed-42b6e036111b.mobileprovision
f8920973-a783-49ca-b4a1-cf455dbd0227.mobileprovision
```
我们可以通过下面命令查看`mobileprovision`文件的内容：

```shell
> PP_FILE=$(ls ~/Library/MobileDevice/Provisioning\ Profiles/f8920973-a783-49ca-b4a1-cf455dbd0227.mobileprovision)
> security cms -D -i "$PP_FILE"
```
结果是一个XML格式的plist文件

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
可见这个文件包含的就是App元信息，包括bundle Id， Entitlements中的内容，支持的device设备列表，以及最重要的**数字证书(DeveloperCertificate)**。

### 代码签名

在前面数字证书一节中曾提到对二进制进行签名的是证书中的私钥，我们可以使用XCode自带的签名工具手动的对某个App进行签名，比如使用下面命令

```shell
> codesign -f -s 'iPhone Developer: xx (xxxx)' Example.app
> codesign --verify Example.app
```
该命令首先会对app中所有的资源文件计算Hash值 - Digest，然后会我们本地的私钥对Digest进行加密，最后将得到的数字签名（signature）注入到Binary（MachO）中，因此Binary的结构会被改变。我们可以使用下面命令查看app的签名状态

```shell
> codesign -vv -d Example.app

Executable=/Example/Payload/Example.app/Example
Identifier=com.idsdk.demo
Format=app bundle with Mach-O thin (arm64)
CodeDirectory v=20200 size=1518 flags=0x0(none) hashes=40+5 location=embedded
Signature size=4705
Authority=iPhone Distribution: xx (xxxx)
Authority=Apple Worldwide Developer Relations Certification Authority
Authority=Apple Root CA
Signed Time=Jul 04, 2016 at 11:28:58 AM
Info.plist entries=26
TeamIdentifier=xxxx
Sealed Resources version=2 rules=10 files=19
Internal requirements count=1 size=172
```

上面已经提到，在签名时，app中的所有资源文件都需要进行加密，加密的结果会存在`_CodeSignature`目录下的`CodeResources`中，该文件是一个xml文件。

```shell
> cat "$WORDPRESS/_CodeSignature/CodeResources" | head -10

<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
	<key>files</key>
	<dict>
		<key>AboutViewController.nib</key>
		<data>
		rSZAWMReahogETtlwDpstztW6Ug=
		</data>
```
我们看到`AboutViewController.nib`文件的签名为`rSZAWMReahogETtlwDpstztW6Ug=`，这个值可以通过下面命令计算得到

```shell
openssl sha1 -binary "$WORDPRESS/AboutViewController.nib" | base64
```
`CodeResources`中保存了app中所有文件的签名值，而app的最终签名会由这些值共同生成。因此只要app中的内容发生一点变化，就会导致整个app的签名发生变化。

### 场景分析

了解上面的基本概念之后，将它们串起来我们便可以分析日常开发中会遇到的签名场景，包括提交AppStore，本地开发，InHouse发布。无论哪种场景我们都需要解决SSL的几条基本问题

1. 验证开发者证书的有效性
2. 验证数字签名
3. 验证代码是否被篡改

我们先从最简单的提交App Store的场景说起，这种情况我们用自己的私钥来加密binary，由于公钥在获取证书时已经上传给Apple，因此App Store可以顺利验证数字签名你和代码是否被篡改过，因此2，3条不是问题。此外，由于ipa是通过App Store分发，只要是Apple设备均可安装，因此第一条也没有问题。

> AppStore的加密过程实际上要比这个复杂，Apple会对我们的ipa进行二次加密，这会导致我们原来的证书信息会被抹去，但无论使用哪种证书，其基本思路是一样的

接下来，我们再来分析本地开发的情况，这种情况对数字签名的验证需要在ipa安装到设备上时完成，具体来说步骤如下

1. 安装时通过CA公钥验证开发证书是否有效
2. 如果有效，则通过CA的公钥提取存在`embedding.mobileprovision`中的签名公钥
3. 通过该公钥来验证数字签名

和AppStore情况不同的是，开发证书签名的ipa是不能随意分发的，不在Provisioning Profiles中的设备是无法安装的。那我们能不能在ipa编译完成后，手动修改里面的`embedding.mobileprovision`？显然这是不可行的，如前文所讲，如果修改app中的任何一个文件，整个app的签名都会发生变化，这样在app安装时，将无法通过验证。

最后我们来说InHouse发布，也就是企业发布。这种情况下所使用的证书是Apple的企业证书，对app的签名使用的也是Apple企业证的私钥。根据最基本的SSL通信规则，在安装企业ipa时，首先需要交换证书，和本地开发模式不同的是，该证书并不是由开发者request的，因此在安装到设备时需要开发者手动同意。同意就意味着该证书有效，那么就可以提取出证书中的公钥（Apple的设备上应该也内置了解密证书用的公钥），用于之后的数字签名验证。<mark>值得注意的是，对于企业证书，没有设备数量的限制</mark>。

## Resource

- [iOS Security Whitepaper](https://www.apple.com/business/site/docs/iOS_Security_Guide.pdf)
- [WWDC2016 How iOS Security Really Works](https://developer.apple.com/videos/play/wwdc2016/705/)
- [WWDC2016 What's New in Xcode App Signing](https://developer.apple.com/videos/play/wwdc2016/401/)
- [iOS Code Signing Guide](https://developer.apple.com/library/archive/documentation/Security/Conceptual/CodeSigningGuide/Introduction/Introduction.html#//apple_ref/doc/uid/TP40005929-CH1-SW1)
- [Inside Code Signing](https://www.objc.io/issues/17-security/inside-code-signing/)
- [Advanced Apple Debugging and Reverse Engineering](https://store.raywenderlich.com/products/advanced-apple-debugging-and-reverse-engineering)

## 附录 - App 重签名

对一个App进行重签名，可分为下面三个步骤

1. 用一个有效的provisioning profile来替换app中的`embedded.mobileprovision`
    - `cp xxx.mobileprovision ${APP}/embedded.mobileprovision`
2. 改变`Info.plist`中的`CFBundleIdentifier`和`ApplicationIdentifier`, 这一步不是必须的，如果pp是wildcard则只需要改写TeamIdentifier
    - `plutil -replace CFBundleIdentifier -string ${TeamID}.xxx Info.plist`
    - `plutil -replace ApplicationIdentifier -string ${TeamID}.xxx Info.plist`
3. 重签名，生成新的签名信息和entitlements
    - 创建一个`ent.xml`文件
    - 拷贝embedded.mobileprovision中的entitlement到dummy.plist中
        - `security cms -D -i "$PP_PATH" > /tmp/scratch`
        - `xpath /tmp/scratch '//*[text() = "Entitlements"]/following-sibling::dict'| pbcopy`
        - 编辑`ent.xml`，插入前一步得到的entitlement信息 - `<dict>...</dict>`
    - 签名所有的动态库 `codesign -fs "iPhone Developer: xx (xxxx)" ${App}/Framework/*.framework`
    - 用entitlements签名app
        - `codesign --entitlements ./ent.xml -f -s "iPhone Developer: xx (xxxx)" ${App}`
4. 使用`mobdevim`或者`ios-deploy`验证



