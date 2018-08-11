---
layout: post
list_title: Node.js中的Event(一)| Event in Node.js
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
如果有过系统编程经验的同学，不论哪种语言，那种平台，对上述代码一定不陌生，这是个最基本的监听-广播模式，当然，上述代码仅仅是个demo，真正使用的时候还需要考虑线程同步问题与资源共享等问题，这里就不展开了。

实际上，Node.js的Emitter的实现也是同样的思路，可以从源码`event.js`中看到其实现代码



### Node Events

