---
layout: post
list_title: Node.js中的Module（二）| How Module works in Node.js Part 2
title: More on Modules 
categories: [Javascript，nodejs]
---

### Native Modules

在前一篇文章中我们分析了`require`的内部实现，其中在`module.load`的函数中，我们发现Node.js可以load native的module，其源码为：

```javascript
Module._load = function(request, parent, isMain) {
  //...
  if (NativeModule.nonInternalExists(filename)) {
    debug('load native module %s', request);
    return NativeModule.require(filename);
  }
  ///....
}
```
因此我们可以尝试在Node.js中require Native的Module，可以到Node.js的文档中找到所有的Native Module，可以随便尝试一个

```javascript
var util = require('utl');
var name = "Tony";
//看起来像C的print
var greeting = util.format('Hello, %s',name);
util.log(greeting); //10 Mar 01:56:23 - Hello, Tony
```
需要注意的是，在`require` native module时，不需要指定路径和后缀名，如果有同名的js module，在require的时候需要加上路径，例如`require('path/util')`

### ES6 Syntax

在ES6中，`require`和`module.exports`变成了`export`和`import`，例如

```javascript
//module1.js
export function greet(){
    console.log("Greeting!");
}
//app.js
import * as greeter from 'greet';
greeter.greet();
```