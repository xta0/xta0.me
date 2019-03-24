---
layout: post
list_title:  iOS的签名原理（二） | App Signing in iOS Part 2
title: 分析WordPress App
categories: [iOS]
---

[WordPress](https://github.com/wordpress-mobile/WordPress-iOS)是一个开源的App，[Advanced Apple Debugging and Reverse Engineering](https://store.raywenderlich.com/products/advanced-apple-debugging-and-reverse-engineering)这本书在Code Sign章节专门分析了这个App，这是一个不错的例子，本文也将使用这个例子进行分析。在阅读下面内容之前，我们还需要安装一个命令行工具[mobdevim](https://github.com/DerekSelander/mobdevim)，这个工具封装了一些常用的签名函数，是一个很好的辅助学习工具。

### The Provisioning Profile

我们先clone WordPress的代码，编译后，使用上一篇文章中提到的命令查看app包中的`.mobileprovision`文件，观察输出结果：

```shell
$`security cms -D -i "$WORDPRESS/embedded.mobileprovision"`
```

- TeamId

```
<key>TeamIdentifier</key>
	<array></array>
		<string>PZYM8XX95Q</string>
	</array>
<key>TeamName</key>
    <string>Automattic, Inc.</string>
```


