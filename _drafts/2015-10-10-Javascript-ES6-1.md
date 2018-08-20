---
layout: post
list_title: Javascript ES6-1
title: Javascript ES6 Part 1
---

### Motavition

补习一下ES6的知识，升级一下JavaScript的技能包。

> 这篇及后面的文章不会覆盖所有ES6的特性

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





