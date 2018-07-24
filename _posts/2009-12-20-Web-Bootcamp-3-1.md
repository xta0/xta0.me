---
title: Javascript Basics
list_title: Web Dev Bootcamp Part 3-1 | DOM & JQuery
layout: post
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
		- `var node = document.getElementById("first"); node.parentNode.removeChilde(node);`
		- `var node = document.getElementsByTagName("p")[0]; node.parentNode.removeChild(node)`
		- `var parentNode = document.getElementById("body"); parentNode.removeChild(parentNode.firstChild)` 

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

## JQuery

### 资源

- [API](api.jquery.com)
- [You Might Not Need JQuery](http://youmightnotneedjquery.com/)
- [What is the future of jQuery for 2017 and later?](https://www.quora.com/What-is-the-future-of-jQuery-for-2017-and-later)

### 定义

- jQuery: 一个Javascript函数库，封装了对不同浏览器适配，以及一些常用的JS API。
- jQuery库包括:
	- HTML元素选取
	- HTML元素操作
	- HTML事件函数
	- CSS操作
	- JavaScript特效和动画
	- HTML DOM遍历和修改
	- AJAX
	- 工具类

> jQuery的流行还是基于`HTML-first`的理念，现在的理念是`javascript-first`，主流的框架例如`React`都是以js为主构建页面，另外随着JS语言的进步，很多jQuery的封装已经有built-in的API了。总的来说jQuery在现在用不触大

### 引入

- 从CDN引入`<script src="https://code.jquery.com/jquery-3.2.1.min.js"></script>`
- JQuery Preview:

```javascript
//when a user clicks the button with id 'trigger'
$('#trigger').click(function(){

	//change the body's background to yellow
	$('body').css("background","yellow");
	
	//fade out all the img's over 3 seconds
	$('img').fadeout(3000, function(){
		
		//remove image from page
		$(this).remove();
	});
});
```
	
### 用法

- 表示DOM节点:
	- 使用`$`符号，jQuery可以将`HTML`标签转换成一个对象，这个对象有`hide`,`show`之类的方法，用来控制标签的行为
		- `$("body")`表示`<body></body>`对应的object
		- `$(".sale")` 所有class为sale的标签
		- `$("#bonus")` id为bouns的标签
		- `$("li a")`所有li下的a标签

- 修改节点属性
	- 使用`.css()`方法
		- 直接修改节点样式：
		
		```javscript
		$("div").css("background-color","purple");
		$("div.highlight").css("width","200px");
		$("#third").css("border","1px solid orange");
		$("div:first-of-type").css("color","pink");
		``` 
		- 使用样式表：
		
		```javascript
		var styles = {
			backgroundColor: "pink",
			fontWeight: "bold"
		};
		$("#special").css(styles);
		```

- 其它方法
	- `val()`：获取、修改`<input>`中的文字内容
		- `$("input").val()`
	- `text()`：获取、修改当前标签的文字（不包含HTML标签）
		- `$('li').text()` 
	- `attr()`：获取、修改标签的属性
		- `$('img').attr("src","https://xxx.jpg");` 
		- `$('img').last().attr("src","https://xxx.jpg");` 
		- `$('img').first().attr("src","https://xxx.jpg");` 
	- `html()`：获取、修改当前标签的HTML文本
		- `$('li').html()`
	- `Class`：
		- `$("h1").addClass("correct")`
		- `$("h1").removeClass("correct")`
		- `$("h1").toggleClass("correct")`  

### Event
	
- `click()`：点击响应

```javascript
$('#submit').click(function(){
	console.log("Another click");
});
$('button').click(function(){
	$(this).css("background", "pink");
});
```
	
- `keypress()`：按键响应

```javascript
$('input[type="text"]').keypress(function(){
	console.log($(this).val());
});
```

- `on()`：添加事件的通用API

```javascript
$('#submit').on('click', function(){
	console.log("click");
});
```

```javascript
$('button').on('dblclick', function(){
	console.log("double click");
});
```

```javascript
$('#ttt').on('dragstart', function(){
	console.log("Drag Started");
});
```

```javascript
$('h1').on('mouseenter', function(){
	console.log("Mouse Enter");
});
```

```javascript
$('h1').on('mouseleave', function(){
	console.log("Mouse Leave");
});
```

- Effect
	- `fadeOut` / `fadeIn`/`fadeToggle`
	
	```javascript
	$('#clickme').click(function(){
		$("#book").fadeOut("slow", function(){
			//animation complete
		});
	});
	``` 
	
	- `slideDown` / `slideUp` / `slideToggle`

	```javascript
	$('#clickme').click(function(){
		$("#book").slideDown(1000, function(){
			//animation complete
			$(this).remove
		});
	});
	```
