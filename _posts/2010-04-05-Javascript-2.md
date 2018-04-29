
---
layout: post
title: Javascript Part 1 
categories: PL
tag: Javascript
---


## OOP

- 构造函数

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

`new`的作用

1. 创建一个空object
2. 创建`this`指向这个空object
3. 在构造函数最后增加一行`return this` 
4. 为空object增加一个成员`__proto__`

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

### Prototypes

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

- Prototype Chain

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

- 抽象方法

可以将构造函数中一些通用方法提取到构造函数对象的`prototype`成员中

```javascript
Person.prototype.sayHi = function (){
    return "Hi "+this.name;
}
var kate = new Person("kate");
kate.sayHi();
```


- 私有成员

JS没有OOP中的私有成员，如果想要某个property不被外部修改，则需要使用closure：

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
```

```javascript

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
