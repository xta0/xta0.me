---
title: Javascript Basics
list_title: Web Dev Bootcamp Part 3 | Javascript Basics
layout: post
categories: [JavaScript]
---

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

### Debug Tools

- Chrome Dev Tools
	- shortcut: `cmd+alt+j` 

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
	- <mark>JS中的String是immutable的，无法修改里面的值</mark>
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

	- 访问数组		
	```javascript
	// 使用下标
	var t = scores[0] //100 
		```
	- `length/index`
	```js
	scores.length(); //5`
	scores.indexOf(100); //9
	scores.indexof(999); //-1 不存在的元素index返回-1`
	```
	- `push/pop`
	```js
	//尾部操作
	scores.push('a','b'); //尾部追加
	scores.pop(); //尾部移除b
	```
	- `shift/unshift`
	```js
	//头部操作 
	scores.unshift('a') //头部插入，[a,a,b];
	scores.shift(); //头部删除，[a,b]
	```
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
			
	- `slice`
		- 返回子数组的copy
		- `names.slice(1) => ['Amit', 'Sarah', 'Ricardo', 'Piers']`
		- `names.slice(); //返回整个数组的copy`
		- `names.slice(1,3);  => ["Amit", "Sarah"] //1,返回3-1=2个元素的数组 2，数组的第一个元素是names[1]`
	- `join`
		- 链接元素，返回string 

		```Javascript
		names.join(','); //=> "Joan,Amit,Sarah,Ricardo,Piers"
		scores.join(','); //=> "100,99,99,72,60"
		```
	- `concat`:链接两个数组
		- `var list = list1.concat(list2);`
	- `map`
		- `array.map(function)` map产生一个新的array，并不会修改原来的array
	- 二维数组:
		- `var list = [[12,3],[2,3,4],[11]]`
	- 字符串算法:
		- `scores.sort()`
		- `scores.reverse()`

	- 遍历数组

	```javascript
	// 参数个数不固定，按照参数顺序推断参数类型
	scores.forEach(function(ele){ 
		console.log(ele); //100,66,...
	})
	scores.forEach(function(ele,idx){ 
		console.log(idx); //0,1,2... 
	})
	//自定义`foreach`
	Array.prototype.myForEach = function(func){
		for(var i=0; i<this.length; i++){
			func(this[i]);
		}
	}
	```


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
	
```js
//function as declaration
function myFunc() {
	//
} 

//function as first class object
var myFunc = function(){
	//...
} 
```

- 函数可做返回值

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
	
- 函数作为First-Class Object参数传递
	
```javascript
setInterval(function(){
	console.log("fired");
},2000);	
```
### OOP

- **构造函数**

使用`Function`定义构造函数，使用`new`构造对象:

```javascript
function House(bedrooms, bathrooms, numSqrt){
    this.bedrooms = bedrooms;
    this.bathrooms = bathrooms;
    this.numSqrt = numSqrt;
}

var firstHouse = new House(3,2,1000);
console.log (firstHouse.bedrooms);
console.log (firstHouse.bathrooms);
console.log (firstHouse.numSqrt);
```

- **new的作用**

1. 创建一个空object
2. 创建`this`指向这个空object
3. 在构造函数最后增加一行`return this` 
4. 为空object增加一个成员`__proto__`

使用`new`构造对象时，对象内部的`this`指向对象本身

```javascript
var Person = function(firstName,lastName){
	this.firstName = firstName //this refers to the Person object
	this.lastName = lastName
}

var tom = new Person("Tom","Xu") 
console.log ( tom.firstName ); 
console.log ( tom.lastName );
```

共享构造函数:可以使用`apply`,`call`传递`this`实现共享构造函数

```javascript
function Car(make, model, year){
    this.make  = make;
    this.model = model;
    this.year  = year;
    this.numWheels = 4;
}

function MotoCycle(make, model, year){
    //using call
    //Car.call(this, make, model, year)
    //using apply
    //Car.apply(this, [make,model,year]);
    Car.apply(this,arguments)
    this.numWheels = 2;
}

var motocycle = new MotoCycle("a","b","2011")
console.log (motocycle.numWheels) //2
```

> 这个设计太奇怪了

- **Prototypes**

![](/assets/images/2010/04/proto-1.png)

```javascript
/* ProtoType */
function Person(name){
    this.name = name;
} 

var elie = new Person("elie");
var colt = new  Person("colt");
```

1. 每个构造函对象数有一个成员叫做`prototype`，类型是`object`
2. 这个`prototype`对象有一个成员叫做`constructor`， 指回构造函数对象

```javasript
Person.prototype.constructor === Person; // true
```

3. 当一个新对象通过`new`创建时，会为其增加一个`__proto__`的成员，指向构造函数对象的`prototype`成员

```javascript
elie.__proto__ === Person.prototype; // true
colt.__proto__ === Person.prototype; //true 
```

- **Prototype Chain**

通过构造函数创建的对象都享有`prototype`的成员

```javascript
Person.prototype.isInstructor = true;
elie.isInstructor = true;
colt.isInstructor = true;
```

```javascript
var arr = new Array;
console.dir(arr); //push方法是__proto__对象的成员
arr.__proto__ === Array.prototype; //true

```

`Prototype Chain`类似继承链，当object调用一个方法，首先在自己的成员找，找不到则在`__proto__`对象中找，这个`__proto__`对象的类型是object的类型，也是构造函数的函数名；当在这个`__proto__`对象仍找不到方法时，会在`__proto__`对象的`__proto__`成员，这个`__proto__`对象的成员类型为`object`（类似基类）中继续寻找。

![](/assets/images/2010/04/proto-2.png)

```javascript
arr. hasOwnProperty('length') //true
dir(arr)
//hasOwnProperty这个方法在 Array.__proto__.__proto__中
```

> 可以看到通过`Prototype Chain`是Javascript在API层面具备了OOP的特性，但实际上，在实现层面是截然不同的，本质原因是JS没有继承的机制

- **抽象方法**

可以将构造函数中一些通用方法提取到构造函数对象的`prototype`成员中

```javascript
Person.prototype.sayHi = function (){
    return "Hi "+this.name;
}
var kate = new Person("kate");
kate.sayHi();
```

- 私有成员

JS没有OOP中的私有成员，如果想要某个property不被外部修改，则需要使用closure制作类似getter的方法：

```javascript
/* hide private variable */
function counter(){
    var count = 0;
    return function (){
        return ++count;
    }
}

var countFn = counter();
countFn(); //countFn函数会隐藏变量`count`

function classRoom(){
    var instructors = ['Colt', 'Elie'];
    return {
        getInstructors: function(){
            return instructors;
        },
        addInstructor: function(instructor){
            instructors.push(instructor);
            return instructors;
        }
    }
}
course1 = classRoom();
course1.getInstructors(); //['Colt', 'Elie']
course1.addInstructor("Ian") //[ 'Colt', 'Elie', 'Ian' ]
```

### This指针

- **Global Context**

```javascript
console.log(this); //window
function whatIsThis(){
	return this; 
}
whatIsThis(); //window
function variableInThis(){
	//sincce the value of this is the window
	//all we are doing here is creating a global variable
	this.person = "Elie" 
}
console.log(person) //Elie
```
- **Strick Mode**

当使用strict模式时，禁止修改全局的`this`，防止污染

```javascript
"use strict"
console.log(this); //window
function whatIsThis(){
	return this; 
}
whatIsThis(); //undefined
function variableInThis(){
	//sincce the value of this is the window
	//all we are doing here is creating a global variable
	this.person = "Elie" //TypeError, can't set person on undefined!
}
```

- **Implicit Context**

`object`中的`this`指向和它最近的`object`
	
```javascript
var person = {
	firstName: "Elie",
	sayHi: function () { 
		return "Hi " + this.firstName 
	},
	determineContext: function () {
		return this === person;
	},
	dog: {
		sayHello: function(){
			return "Hello "+this.firstName;
		},
		determineContext: function () {
			return this === person;
		}
	}
}
erson.sayHi(); //"Hi Elie"
person.determineContext() //true
person.dog.sayHello(); //"Hello undefined"
person.dog.determineContext(); //false
```

- **Explicit Context**

显式指定`this`的值，使用`call`,`apply` or`bind`

| Name | Parameters | Invoke Immediately |
|:----:|:---------------:| :----:|
| Call | thisArg,a,b,c,d... 		 | yes | 
| Apply| thisArg,[a,b,c,d,...] 	 |Yes 	|
| Bind | thisArg,a,b,c,d,...		 |No 	|

- **call**

使用`call` , 接受`this`+`可变参数`, 立即执行

```javascript
var colt = {
	firstName: "Colt",
	sayHi: function () { 
		return "Hi " + this.firstName 
	},
	addNumber: function(a,b,c,d){
		return this.firstName + " just calculated "+ (a+b+c+d);
	},
	sayHiLater: function(){
		setTimeout(function(){
			console.log( "Hi Later " + this.firstName );
		}.bind(this),1000)
	}
}

var elie = {
	firstName: "Elie"
}
colt.sayHi(); //Hi Colt
colt.sayHi.call(elie); //Hi Elie
colt.addNumbers(1,2,3,4); //Colt just calculated 10
colt.addNumbers.call(elie, 1,2,3,4); //Elie just calculated 10
```

- **apply**

使用`apply`, 接受`this`+`[可变参数]`, 立即执行

```javascript
colt.addNumbers.apply(elie, [1,2,3,4]); //Elie just calculated 10
```

- **bind**

`bind`接受`this`+`可变参数`, 返回一个函数对象

```javascript
var elicAddNumber = colt.addNumber.bind(elie);
elicAddNumber(1,2,3,4);//'Elie just calculated 10'
colt.sayHiLater(); //Hi Later Colt
```





## DOM Operation

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