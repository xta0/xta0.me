---
layout: post
updated: "2018-08-10"
list_title:  ES6快速参考 | ES6 Quick Reference Guide
title: ES6快速参考
categories: [JavaScript]
---

### Motavition

最近在学习React，持续更新对ES6各种feature的理解

### Array

- `forEach` 循环

```javascript
var colors = ['red','blue','green'];
colors.forEach(function(color){
  console.log(color);
});
```

### Sets & Maps

Sets和Maps默认会强引用集合内的object，引用计数+1。 WeakSet和WeakMap不会强引用object，当集合内的某个object被delete或被指为`null`时，GC会进行回收，则该对象会自动从集合中被删除。WeakSet/WeakMap有如下几个特性

1. WeakSet/WeakMap只包含objects
2. WeakSet/WeakMap不可被枚举
3. WeakSet/WeakMap没有`clear()`方法

以WeakSet为例，WeakMap同理

```javascript
let student1 = { name: 'James', age: 26, gender: 'male' };
let student2 = { name: 'Julia', age: 27, gender: 'female' };
let student3 = { name: 'Richard', age: 31, gender: 'male' };

const roster = new WeakSet([student1, student2, student3]);
console.log(roster);
/*
WeakSet {
  Object {name: 'Julia', age: 27, gender: 'female'}, 
  Object {name: 'James', age: 26, gender: 'male'},
  Object {name: 'James', age: 26, gender: 'male'}
}
*/
student3 = null;
console.log(roster);
/*
WeakSet {
  Object {name: 'Julia', age: 27, gender: 'female'}, 
  Object {name: 'James', age: 26, gender: 'male'}
}
*/
```

### Let and Const

到目前为止，在JS中声明变量只能用`var`，为了理解`let`和`const`需要先搞清楚`var`的一些弊端，看下面例子

```javascript
function getClothing(isCold) {
  if (isCold) {
    var freezing = 'Grab a jacket!';
  } else {
    var hot = 'It’s a shorts kind of day.';
    console.log(freezing);//undefined
  }
}
getClothing(false)
```
JavaScript的执行过程包含两部分，一部分是解释，一部分是执行，因此需要两个context，一个是scope context用来解释，一个是excution context用来执行。上函数在执行前JS的解释器会对函数栈中所有用到的变量先扫一遍并注册到Scope Context中（这是我个人的叫法，想不出特别好的词来形容，实际上应该是属于这个函数scope的一片内存或者一个对象，用于保存用到的符号），这个过程会让`freezing`被初始化为`undefined`。接着在执行阶段，会被打印输出。

显然这是个非常糟糕的设计，应该说是错误的设计，原因就不展开了。为了解决这个问题ES6引入了`let`和`const`，通过这两个变量声明的对象，其生命周期只局限在最近的scope内，这个其实就符合了一般编程语言对变量作用域的规则。

```javascript
function getClothing(isCold) {
  if (isCold) {
    const freezing = 'Grab a jacket!';
  } else {
    const hot = 'It’s a shorts kind of day.';
    console.log(freezing); //ReferenceError: freezing is not defined
  }
}
```

除了限定变量的作用域外，`let`和`const`还有下面两条规则

1. 用`let`声明的变量可以被重新赋值，但同一符号在同一作用域内不能被重复声明
2. 用`const`声明的变量必须赋初值，在同一作用域内既不能被重新赋值，也不能被重复声明，`const`比`let`更严格

因此在以后声明变量时，只用`let`即可，如果不希望它被修改，则加上`const`

### Destructing

引入Destructing应该是借鉴了Python或者Perl，它允许等号左边的变量自动绑定到等号右边对象的某个key的值，实际上是是一种语法糖，通过patter matching来减少冗余代码

- Destructuring values from an array

```javascript
const point = [10, 25, -34];
const [x, y, z] = point;
console.log(x, y, z); //10 25 -34
const [x, , z] = point; //10 -34
```

- Destructuring values from an object

```javascript
const gemstone = {
  type: 'quartz',
  color: 'rose',
  carat: 21.29
};
const {type, color, carat} = gemstone;
console.log(type, color, carat); //quartz, rose, 21.29
```
这个特性在object传参的时候特别好用

```javascript
const savedFile = {
    extension: 'jpg',
    name: 'repost',
    size: 100
};
//参数destructing
function fileSummary({name,extension,size}){
    return `The file ${name}.${extension} is of size ${size}`
}
fileSummary(savedFile)
```
如果`object`中有函数作为成员的话，destructing要格外小心，例如

```javascript
const circle = {
  radius: 10,
  color: 'orange',
  getArea: function() {
    return Math.PI * this.radius * this.radius;
  },
  getCircumference: function() {
    return 2 * Math.PI * this.radius;
  }
};

let {radius, getArea, getCircumference} = circle;

//call function
getArea() //NaN
```
此时当调用该函数时，由于对象并未创建，this指针并不存在，因此输出`NaN`

### Spread & Reset Operator

ES6引入了`...`符号做集合类对象的展开（spread）与合并(reset)操作

- Spread

```javascript
const books = ["Don Quixote", "The Hobbit", "Alice in Wonderland", "Tale of Two Cities"];
console.log(books);
/*
[ 'Don Quixote',
  'The Hobbit',
  'Alice in Wonderland',
  'Tale of Two Cities' ]
*/
console.log(...books); 
//Don Quixote The Hobbit Alice in Wonderland Tale of Two Cities
```

上述代码中，相当于取出了`books`中的所有object。再看一个例子，如果要连接两个Array，以前的做法为

```javascript
const fruits = ["apples", "bananas", "pears"];
const vegetables = ["corn", "potatoes", "carrots"];
const produce = fruits.concat(vegetables);
console.log(produce);
```
使用Spread操作则可以将`fruits`和`vegetables`两个Array依次展开后再拼接

```javascript
const produce = [...fruits,...vegetables];
console.log(produce);
 //[ 'apples', 'bananas', 'pears', 'corn', 'potatoes',` 'carrots' ]
```
- Reset Operator

合并操作用来合并object到一个Array，下面代码将后四个对象合并到一个`items`中

```javascript
const order = [20.17, 18.67, 1.50, "cheese", "eggs", "milk", "bread"];
const [total, subtotal, tax, ...items] = order;
console.log(total, subtotal, tax, items);
//20.17 18.67 1.5 [ 'cheese', 'eggs', 'milk', 'bread' ]
```

- 使用Reset做可变参数

例如定义一个`sum()`函数接受可变参数求和，ES6之前可以使用`arguments`

```javascript
function sum() {
  let total = 0;  
  for(const argument of arguments) {
    total += argument;
  }
  return total;
}
console.log(sum(1, 23, 3, 4)) //31
```
使用Reset Operator

```javascript
function sum(...nums) {
  let total = 0;  
  for(const num of nums) {
    total += num;
  }
  return total;
}
```
这种方式比使用`arguments`更容易理解，并且在遍历的时候可以使用ES6的`for...of`循环

### Arrow Functions

Arrow Function规则：

- 如果函数只有一个参数，则参数可省略括号
- 如果函数没有参数或者有多个参数，则不可省略括号

```javascript
var greet = name => { 
  return "Hello: " + name
};
var greet = () =>{
  console.log("Greet!")
}
```
- 如果函数体只有一条语句
  - 不需要大括号
  - 不需要`return`语句

```javascript
var greet = name => "Hello: " + name;
```

### `this`

Arrow function修正了`this`的作用域，参考下面代码

```javascript
// constructor
function IceCream() {
  this.scoops = 0;
}

// adds scoop to ice cream
IceCream.prototype.addScoop = function() {
  setTimeout(function() {
    this.scoops++; // references the `cone` variable
    console.log('scoop added!');
  }, 0.5);
};

const dessert = new IceCream();
dessert.addScoop();
```

上述代码中，`addScoop`执行时，调用了`timer`，由于`timer`是异步执行，`this`未绑定到任何object，因此指向了全局`object`。一种补救方法是显式的利用closure绑定`this`，或者使用`bind`函数将`this`传到timer函数

```javascript
//capture this outside the callback function
IceCream.prototype.addScoop = function() {
  var self = this;
  setTimeout(function() {
    self.scoops++; 
    console.log('scoop added!');
  }, 0.5);
};
//bind this
IceCream.prototype.addScoop = function() {
  setTimeout(function() {
    self.scoops++; 
    console.log('scoop added!');
  }.bind(this), 0.5);
};
```

ES6的Arrow funciton修正了这个缺陷，在Arrow function中`this`的指向不会根据位置的变化而变化，当`this`值在初始化被确定后，任何位置的`this`的都指向原对象

```javascript
IceCream.prototype.addScoop = function() {
  setTimeout(() => { 
    this.scoops++;
    console.log('scoop added!');
  }, 0.5);
}
//another example
var person = {
  name: "Elie",
  func: function(){
    var setname = name => this.name = name;
    setname("Tom");
    return this;
  }
}
person.func().name;// Tom
```

### Classes

ES6设计了一大把语法糖，是JavaScript看起来更像大家都熟悉的面向对象语言，但实际上底层实现还是基于Prototype的，并没有本质的变化

```javascript
class Tree {
  constructor(size = '10', leaves = {spring: 'green', summer: 'green', fall: 'orange', winter: null}) {
    this.size = size;
    this.leaves = leaves;
    this.leafColor = null;
  }
  changeSeason(season) {
    this.leafColor = this.leaves[season];
    if (season === 'spring') {
      this.size += 1;
    }
  }
}

class Maple extends Tree {
  constructor(syrupQty = 15, size, leaves) {
    super(size, leaves);
    this.syrupQty = syrupQty;
  }

  changeSeason(season) {
    super.changeSeason(season);
    if (season === 'spring') {
      this.syrupQty += 1;
    }
  }
  gatherSyrup() {
    this.syrupQty -= 3;
  }
}

const myMaple = new Maple(15, 5);
myMaple.changeSeason('fall');
myMaple.gatherSyrup();
myMaple.changeSeason('spring');
```

和ES5对比

```javascript
function Tree(size, leaves) {
  this.size = (typeof size === "undefined")? 10 : size;
  const defaultLeaves = {spring: 'green', summer: 'green', fall: 'orange', winter: null};
  this.leaves = (typeof leaves === "undefined")?  defaultLeaves : leaves;
  this.leafColor;
}

Tree.prototype.changeSeason = function(season) {
  this.leafColor = this.leaves[season];
  if (season === 'spring') {
    this.size += 1;
  }
}

function Maple (syrupQty, size, leaves) {
  Tree.call(this, size, leaves);
  this.syrupQty = (typeof syrupQty === "undefined")? 15 : syrupQty;
}

Maple.prototype = Object.create(Tree.prototype);
Maple.prototype.constructor = Maple;

Maple.prototype.changeSeason = function(season) {
  Tree.prototype.changeSeason.call(this, season);
  if (season === 'spring') {
    this.syrupQty += 1;
  }
}

Maple.prototype.gatherSyrup = function() {
  this.syrupQty -= 3;
}

const myMaple = new Maple(15, 5);
myMaple.changeSeason('fall');
myMaple.gatherSyrup();
myMaple.changeSeason('spring');
```

另外需要注意的一点是，在同一个class内，不允许同名函数的存在，即是它们签名不同，这点和ES5是相同的。

### Symbol

Symbol 是 ES6 中新加入的原生数据类型。一个 Symbol 类型的对象可以通过`Symbol`函数生成。`Symbol`函数也可以传入字符串，作为 Symbol 的名称，但是相同名字的 Symbol 彼此也是不同的对象

```javascript
//create a symbol
let s = Symbol();
console.log(typeof s); // "symbol"

//create a symbo with name
const sym1 = Symbol("banana");
const sym2 = Symbol("banana");
console.log(sym1 === sym2); //false
```

Symbol 的这个特性可以用来取代硬编码的字符后者数字常量

```javascript
const COLOR_RED = Symbol();
const COLOR_GREEN = Symbol();

function getComplement(color) {
  switch (color) {
    case COLOR_RED:
      return COLOR_GREEN;
    case COLOR_GREEN:
      return COLOR_RED;
    default:
      throw new Error("Undefined color");
  }
}
```

我们还可以使用 Symbol 消除同名属性

```javascript
const key1 = Symbol("greet");
const key2 = Symbol("greet");
let person = {
  name: "James"
};
person[key1] = function() {
  return this.name;
};
person[key2] = function(msg) {
  return msg;
};
console.log(person[key1]()); //James
console.log(person[key2]("Some Message"));
```

上述代码为`person`添加两个同名的`greet`方法，由于使用`Symbol`方式添加，因此并不会造成同名函数的覆盖。注意，Symbol 需要使用`[]`符号进行赋值或访问。

使用 Symbol 添加的属性不会出现在`Object.keys(person)`、`Object.getOwnPropertyNames()`以及`JSON.stringify()`之中。但是，Symbol 也不是私有属性，有一个`Object.getOwnPropertySymbols`方法，可以获取指定对象的所有 Symbol 属性名。

```javascript
console.log(Object.getOwnPropertySymbols(person));
//[ Symbol(greet), Symbol(greet) ]
```

### 迭代器

ES6 中的`Iterable Protocols`代表一个 interface，实现了该 interface（`Symbol.iterator`）的对象具备枚举能力。例如，`Array`:

```javascript
const digits = [0, 1, 2];
const arrayIterator = digits[Symbol.iterator](); //object
```

`digits[Symbol.iterator]()`返回一个类型为`object`的 iterator,它从`Arrary Iterator`继承下来，有一个`next()`方法，该方法返回一个`object`包含两个属性`value`和`done`，分别用来表示此次枚举的结果以及是否枚举到数组的末端。对于 Map 和 Set 也可获取各自的迭代器

```javascript
console.log(arrayIterator.next()); //{ value: 0, done: false }
console.log(arrayIterator.next()); //{ value: 1, done: false }
console.log(arrayIterator.next()); //{ value: 2, done: false }
console.log(arrayIterator.next()); //{ value: undefined, done: true }
```

### Promise

Promise 是一种异步回调模型，ES6 中集成了 Promise，用法和之前的 Promise Framework 基本相同

```javascript
const task = new Promise((resolve, reject) => {
  //do some work
  if (err) {
    //failed
    reject(err);
  } else {
    //succeed
    resolve();
  }
});
```

### Proxy

TBD

### Generator

Generator 是 ES6 引入的一种被动（惰性）求值技术

```javascript
//generator function
function* getEmployee() {
  console.log("the function has started");
  const names = [
    "Amanda",
    "Diego",
    "Farrin",
    "James",
    "Kagure",
    "Kavita",
    "Orit",
    "Richard"
  ];
  for (const name of names) {
    console.log(name);
    yield; //new keyword
  }
  console.log("the function has ended");
}
```
未完待续。。

### Resource

- [Udacity ES6 Tutorial](https://classroom.udacity.com/courses/ud356)
- [Udemy ES6 Tutorial](https://www.udemy.com/javascript-es6-tutorial)
- [ECMAScript 6 入门](http://es6.ruanyifeng.com/)



