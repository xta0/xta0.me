---
layout: post
list_title: Javascript Part 1  | This 
title: Javascript中的This
sub_title: This in Javascript
---

> update @2016/4/10

## 理解This 

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

### Implicit Context

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

### Explicit Context

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

### New Context

使用**`New`**构造对象时，对象内部的`this`指向对象本身

```javascript
var Person = function(firstName,lastName){
	this.firstName = firstName //this refers to the Person object
	this.lastName = lastName
}

var tom = new Person("Tom","Xu") 
console.log ( tom.firstName ); 
console.log ( tom.lastName );
```


### Closures

```javascript
function outer(){
    var data = "closures are ";
    return function inner(){
        var innerData = "awesome";
        return data + innerData;
    }
}

//outer()();
var inner = outer();
inner();
```

JS里所谓的`closure`指的是一个匿名函数，并且这个函数使用了外部变量
