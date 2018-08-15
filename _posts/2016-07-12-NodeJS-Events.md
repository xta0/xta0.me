---
layout: post
updated: "2017-10-11"
list_title: 理解Node.js | Events
title: Event
categories: [Javascript，nodejs]
---

### 两种Event

- System Events
    - C++ Core
        - libuv
    - File Operation
    - Network Opertaion

- Custom Events
    - Javascript Core
        - Event Emitter
    - Self-define event

### 观察者与监听者

我们可以快速手写一个简单的观察者与监听者模型

```javascript
//监听者，Emitter.js
function Emitter(){
	this.events = {}
}
Emitter.prototype.on = function(type,listener){
	this.events[type] = this.events[type] || []
	this.events[type].push(listener)
}
Emitter.prototype.emit = function(type){
	if(this.events[type]){
		this.events[type].forEach((listener)=>{
			listener()
		})
	}
}
module.exports = Emitter

//观察者, app.js
var Emitter = require('./emmiter')
var emtr = new Emitter()
emtr.on('greet',function(){
	console.log('#1 said hello.')
})
emtr.on('greet',function(){
	console.log('#2 said hello.')
})
emtr.emit('greet')
```
这是个最基本的监听-广播模式，如果翻看Node.js关于Emitter的源码实现（`event.js`），可以发现，其基本思路和上面的代码是一样的。将上述代码中的`require('./emmiter')`改为使用系统的`require('events')`，结果是一样的

### 自定义Emitter

我们可以自定义一个`EventEmitter`继承于Nod.js中的`EventEmitter`

```javascript
var EventEmitter = require ('events')
var util = require ('util')

function MyEmitter(){
	EventEmitter.call(this)
}
util.inherits(MyEmitter, EventEmitter)

MyEmitter.prototype.greet = function(key,data){
	this.emit(key,data)
}
var emitter = new MyEmitter()
emitter.on('Hello', function(data){
	console.log('event triggered! msg: '+data)
})
emitter.greet('Hello','Some_data')

```

由于ES5中没有继承的概念，想要实现继承需要使用lib库中的`util.js`，其原理是使用Prototype Chain来模拟继承，感兴趣的可以阅读其源码或者参考之前介绍JavaScript的文章

### ES6

最近学习了ES6,将上述代码用ES6改写如下

```javascript

//MyEmitter.js
'use strict';
var EventEmitter = require ('events')
module.exports = class MyEmitter extends EventEmitter{
	constructor(){
		super()
	}
	greet(key,data){
		this.emit(key,data)
	}
}
module.exports = MyEmitter

//app.js
var MyEmitter = require('./MyEmitter')

var emitter = new MyEmitter()
emitter.on('Hello', function(data){
	console.log('event triggered! msg: '+data)
})
emitter.greet('Hello','Some_data')
```

<p class="md-h-center">(全文完)</p>
