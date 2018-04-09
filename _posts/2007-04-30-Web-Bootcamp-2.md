---
title: Web Bootcamp-2
layout: post
categories: [javascript]
meta: Web Development Bootcamp. Basic Javascript usage
---

## Javascript

### 引入JS的三种方式

- 在`<head></head>`中引入
- 一般在`<body>`末尾插入外联的js文件:

```html
<body>
	
	<div class="container">
		<h1>Hello world</h1>
	</div>		

	<script src="scripts/script.js"></script>
	
</body>
```

### Debug

- chrome: `cmd+alt+j` 

### 数据类型

- 5种基本数据类型
	- number
	- string
	- boolean
	- null
	- undefined

- **Variable**
	- Syntax	
		- `var score = 0;`
	- 数字，字母，下划线，`$`符号开头
	- 类型检查
		- `typeof`关键字
		- `typeof(111); //"number"`
		- `typeof({name:'abc'}); //"object"` 
		- `typeof(undefined); //"undefined"`


- **Numbers**
	- Integer，Float
	- 科学计数法:`9e-6`,`9e+6`;
	- 数学库:`Math`
		- 四舍五入:`Math.round(4.5);` 
		- 向下round:`Math.floor(4.5); => 4`
		- 向上round:`Math.ceil(4.5); => 5`
		- 随机数:`Math.random();` 返回[0-1)的随机小数
			- [x,y]之间的数：`var a =Math.floor(Math.random()*(y-x+1))+y`; 
	
	-  Number转String:`String(x);`

- **Stings**
	- JS中的String是immutable的，无法修改里面的值
	
	```js
	str = "abc"
	str[0] = 'b' #not working
	``` 
	- 获得字符
		- `"hello[0]" //"h"` 
	
	- Concatentation:
		- `one string` + `another string`
		- `var name = "Dave"; var message = "Hello "+name` 
	
	- Length: 
		- `a.length` 
		
	- uppercase / lowercase : 
		- `a.toUpperCase(); a.toLowerCase();` 
	
	- 字符转数字:
		- `parseInt('10');`	
		- `parseFloat('3.14');`
		- `parseInt('100 badges'); => 100`
		- `parseInt('badges100'); => Nan`
		- `Number(string)`
	
	- 检查字串，返回position
		- `string.indexof("text")`  

	- 字符串替换
		- `string.replace(\regex\g, otherString)` 返回一个新的string，不修改原string 
	
	- Escape Character
		- `\`开头
		- `"Singing\"Merry Chrismas\""`
		- `"This is a backslash \\"` 


- **Boolean**
	- A Boolean value can only be `true` or `false`


- **Null/Undefined**
	- `Undefined`: 变量定义未初始化
		- `var name`
	- `null`: 明确变量值为空

### 数据结构

- **Arrays**
	- 创建数组:一个Array可以存放不同类型的object

	```javascript
	var scores = [100, 99, 99, 72, 60];
	var names = ['Joan', 'Amit', 'Sarah', 'Ricardo', 'Piers'];
	var values = [1, -100, true, false, 'JavaScript'];
	```

	- 访问数组:使用下标
		- `var t = scores[0] //100 `
	- `length/index`
		- `scores.length(); //5` 
		- `scores.indexOf(100); //9`
		- `scores.indexof(999); //-1 不存在的元素index返回-1`
	- `push/pop`: 尾部操作
		- `scores.push('a','b'); //尾部追加`
		- `scores.pop(); //尾部移除b`
	- `shift/unshift`:头部操作 
		- `scores.unshift('a') //头部插入，[a,a,b];`
		- `scores.shift(); //头部删除，[a,b]`
	- `splice`
		- 删除操作
			- API:	`array.splice(index,length)` 原数组被修改，返回一个新数组
				- `scores.splice(1,1); => [99]`
				- `scores.splice(1,0); => [  ]`
		- 插入操作
			-  API:`array.splice(position,0,element)`，原数组修改，返回一个空数组
				- `scores.splice(1,0,66); => [100, 66, 99, 99, 72, 60]`
				- `scores.splice(100,0,66); => [100, 66, 99, 99, 72, 60,66]` 越界默认插入到最后
		- 替换操作
			- API: `array.splice(position,length,element)`
				- `names.splice(1,1,'Jayson') =>[ 'Joan', 'Jayson', 'Sarah', 'Ricardo', 'Piers' ]`
			
	- `slice`：返回子数组的copy
		- `names.slice(1) => ['Amit', 'Sarah', 'Ricardo', 'Piers']`
		- `names.slice(); //返回整个数组的copy`
		- `names.slice(1,3);  => ["Amit", "Sarah"] //1,返回3-1=2个元素的数组 2，数组的第一个元素是names[1]`

	- `join`：链接元素，返回string 
		- `names.join(','); => "Joan,Amit,Sarah,Ricardo,Piers"`
		- `scores.join(','); => "100,99,99,72,60"`
	- `concat`:链接两个数组
		- `var list = list1.concat(list2);`
	- 二维数组:
		- `var list = [[12,3],[2,3,4],[11]]`
	- 排序:
		- `scores.sort()`
	- 翻转:
		- `scores.reverse()`

	- `foreach`遍历数组
		- API: `array.foreach(function(ele,idx,array){...})`
		
		```javascript
		// 参数个数不固定，按照参数顺序推断参数类型
		scores.forEach(function(ele){ 
			console.log(ele); //100,66,...
		})
		scores.forEach(function(ele,idx){ 
			console.log(idx); //0,1,2... 
		})
		```
		- 自定义`foreach`
		
		```javascript
		Array.prototype.myForEach = function(func){
			for(var i=0; i<this.length; i++){
				func(this[i]);
			}
		}
		```
	- Map
		- `map`:`array.map(function)` map产生一个新的array，并不会修改原来的array

- **Object**
	- 创建Object
		- `var student = { name: "Dave", grades: [1,23,4] };`
		- `var student={}; student.name="Dave"; student.grades:[1,23,4]`
		- `var student = new Object(); student.name="Dave"; student.grades:[1,23,4]`
	- 成员变量:
		- 访问：
			- `student['name']`
			- `student.name` 语法糖
		- 更新：
			- `student.name = "Tim"` 
		- 遍历：
		
		```javascript
		for(var key in object)
		{
			var value = object[key];
			//wrong: var value = object.key 
		}
		``` 
	- 成员方法
		- `student.method = function(var1,var2){...}` 

- More on Javascript
	- [Javascript Notes]() 
	
### 表达式

- **比较运算**
	- `>`,`>=`,`<`,`<=`
	- `==``!=`：非严格类型比较，转成相同类型比较值
		- `("3" == 3) => true` 
		- `("" == 0) => true`
		- `var y=null; y == undefined //true`
		- 特别的例子
			- `true == "1" //true`
			- `0 == "false" //true`
			- `null == undefined //true`
			- `NaN == NaN //false, not comparable` 
	- `===``!==`: 严格类型比较，比较值和类型
		- `("3" === 3) => false` 
		- `var y=null; y===undefined //false`

- **逻辑运算**
	- `&&, ||, !`
	- 所有值可比较，自带`true`或`false`的属性
		- `!"Hello World" //false`
		- `!"" //true`
		- `!null //true`
		- `!0 //true`
		- `!-1 //false`
		- `!NaN //true`
	- 自身错误属性的值：`false, 0, "", null, undefined, NaN` 

- **条件控制**
	- **使用if**:
	
	```javascript
	if()
	{...}
	else if()
	{...} 
	else{...}`
	```
		
	- **使用switch**
	
	```javascript
	switch(var)
		case:"option_!":
			do_something();
			break;
		default:
		break;
	```


- **循环语句**
	- **使用while**
	
	```javascript
	while()
	{
		...
		break;
	}
	```
	
	- **使用do-while**
	
	```javascript 
	do{
		...
	}
	while( ... break;)
	```
	
	- **使用for**
		- 使用index和list：
		
		```javascript
		for(var i=0;i<list.length;i++){
			...
			break;
		}
		```
		- `for-in`
		
		```javascript
		for (var obj in list){ 
			... 
		}
		```


### Function

- 定义
	- `function myFunc() {...} //function as declaration`
	- `var myFunc = function(){...} //function as first class object`

- 返回值
	- return function from function

	```javascript
	function counter(){
		var count = 0;
		return function(){
			count ++;
			alert(count);
		}
	}
	var count = counter();
	count();
	count();
	count();
	```
	
- HOF
	- Function as argument
	
	```javascript
	setInterval(function(){
		console.log("fired");
	},2000);	
	```

## DOM(Document Object Model)

### Overview

![](/assets/images/2007/04/dom.png)

- 访问`document`对象
	- `console.dir(document)`
	- `document.URL`
	- `document.head`
	- `document.body`

- 修改DOM属性

```javascript
var h1 = document.querySelector("h1")
h1.style.color = "orange";
```

### 访问DOM节点

- 几种方式
	- `document.childNodes[i].childNodes[i].childNodes[i]...`
	- `document.getElementsByTagName()`
		- `var tags = document.getElementByTagName("li"); //返回array` 
	- `document.getElementById()`
		- `var tag = document.getElementById("some_id"); //返回唯一Id标签object` 
	- `document.getElementByClassName()`
		- `var tags = document.getElementByClassName("some_class"); //返回某个class的数组`
	- `document.querySelector()`
		- `selector`的值为CSS的标签，可以为`tag`，`id`和`class` 
		- `var tag = document.querySelector('#highlight');` 
		- `var tag = document.querySelector(".bolded");  //返回第一个匹配的tag`
		- `var tag = document.querySelector("li a.special")`
	- `document.querySelectorAll()`
		- `var lis = document.querySelectorAll("li"); //返回所有li标签` 

### 改变节点Style

- 直接修改节点属性
	
```javascript
var node = getElementById("id")
node.style.color = "blue";
node.style.border = "10px solid red";
node.style.fontSize = "70px";
node.style.background = "yellow";
node.style.marginTop = 20px
node.setAttribute("style":"color:red")
``` 
- 使用样式表替换当前节点属性
	
```javascript
.some-class{
	color: blue;
	border: 10px solid red;
}
var tag = document.getElementById("highlight");
//ADD THE NEW CLASS TO THE SELECTED ELEMENT
tag.classList.add('some-class');
	
//REMOVE A CLASS
tag.classList.remove('some-class');
	
//TOGGLE A CLASS
tag.classList.toggle("some-class");
```
	
- 改变文字
	- 使用`textContent`，返回的文字中不包含HTML标签

	```javascript
	<p>This is an <strong>awesome</strong> paragraph </p>
	var p = document.querySelector("p");
	p.textContent;  //"This is an awesome paragraph ";
	```	
	
	- 使用`innerHTML`会保留文字的HTML标签
	
	```javascript
	<p>This is an <strong>awesome</strong> paragraph </p>
	var p = document.querySelector("p");
	p.innerHTML;  //"This is an <strong>awesome</strong> paragraph "	```	
	
- 改变**Attributes**

使用`getAttribute()`和`setAttribute()`来读写属性

```HTML
 <a href="http://www.google.com">I am a link</a>
 <img src="logo.png" >
```

```javascript
var link = document.querySelector("a");
link.getAttribute("href");
link.setAttribute("href","www.baidu.com");

var img = document.querySelector("img");
img.setAttribute("src","./dog.png");

```
	
	
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

### 参考资料

- [MDN](https://developer.mozilla.org)