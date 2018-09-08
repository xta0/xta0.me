---
title: Javascript Basics
list_title: Web Dev Bootcamp Part 3-1 | DOM & JQuery
layout: post
categories: [Web]
---

### DOM 节点操作

- 创建DOM节点
	- `result = document.createElement("div");`
	- `result = document.createTextNode("Hello")`

- 添加DOM节点
	- `document.getElementById("id").appendChild(node)` 

- 删除节点：
	- `document.getElementId("id").parentNode.removeChild(node)` 
	- 三种方式：

	```javascript
	//by Id
	var node = document.getElementById("first"); 
	node.parentNode.removeChilde(node);
	//by tag
	var node = document.getElementsByTagName("p")[0]; 
	node.parentNode.removeChild(node)
	var parentNode = document.getElementById("body"); 
	parentNode.removeChild(parentNode.firstChild)
	```
- Clone DOM节点
	- Copying a node : `node.cloneNode()`
	- Copying a branch : `node.cloneNode(true)`
	- Adding nodes : `dest.appendChild()`


### DOM Event


- Timers
	- Timers are very useful for dynmic web page behavior
	
	```javascript
	var timer;
	timer = setTimeout(do_something, 1000);
	```
	- `do_something()` will be executed 1 second later
	- The value 1000 is in milliseconds, so 1000 = 1 second
	- stop timer:

	```javascript
	clearTimeout(the_timer)
	```

	- `setInterval()` repeatedly does something

	```javascript
	var the_timer;
	the_timer = setInterval(do_something,2000)
	```

	- `do_something()` will be executed every 2 seconds。 To stop it:

	```javascript
	clearInterval(the_timer);
	```

- Event Listener
	- Adding an event to `<body>`
	
		```html
		<html>
			<head>
				<script>
					function do_something(){ alert("Page has loaded");}
				</script>
			</head>
			<body onload="do_something()"></body>
		</html>
		```
	
	- Adding an event to `window`

	```html
	<html>
		<body id="thebody">
			<script>
				function do_something(){ alert("Page has loaded");}
				window.onload = do_something;
				window.addEventListener("load",do_something);
			</script>
		</body>
	</html>
	```
	
	-  Adding an event to `element`
		- API : `element.addEventListener(type,functionToCall)`
	
		```javascript
		var button = document.querySelector("button");
		button.addEventListener("click", function(){
			console.log("SOMEONE CLICKED THE BUTTON!")
		});
		```
	
- If you have more than one event handler
	- event handlers are stored in an array
	- when an event happens, all the handlers are executed one by one 
	- They are executed in the order they added

- To remove an event handler

```html
var theBody = document.getElementById("theBody");
theBody.removeEventListener("load",do_something);
```

- remove只能remove通过js添加的listener


