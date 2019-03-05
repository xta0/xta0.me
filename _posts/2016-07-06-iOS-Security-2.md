---
layout: post
list_title:  iOS的签名原理（二） | App Signing in iOS Part 2
title: 分析WordPress App
categories: [iOS]
---

[WordPress](https://github.com/wordpress-mobile/WordPress-iOS)是一个开源的App，[Advanced Apple Debugging and Reverse Engineering](https://store.raywenderlich.com/products/advanced-apple-debugging-and-reverse-engineering)这本书在Code Sign章节专门分析了这个App，本文将是这部分内容的一个小结

### The Provisioning Profile

使用上一节提到的命令查看`security cms -D -i "$WORDPRESS/embedded.mobileprovision"`，重点观察下面信息：

- TeamId

```
<key>TeamIdentifier</key>
	<array></array>
		<string>PZYM8XX95Q</string>
	</array>
<key>TeamName</key>
    <string>Automattic, Inc.</string>
```


