---
layout: post
list_title: Javascript ES6 Quick Reference Guide Part 2
title: Javascript ES6 Part 2
categories: [JavaScript]
---

### Symbol

Symbol是ES6中新加入的原生数据类型。一个Symbol类型的对象可以通过`Symbol`函数生成。`Symbol`函数也可以传入字符串，作为Symbol的名称，但是相同名字的Symbol彼此也是不同的对象

```javascript
//create a symbol
let s = Symbol();
console.log(typeof s) // "symbol"

//create a symbo with name
const sym1 = Symbol('banana');
const sym2 = Symbol('banana');
console.log(sym1 === sym2); //false
```
Symbol的这个特性可以用来取代硬编码的字符后者数字常量

```javascript
const COLOR_RED    = Symbol();
const COLOR_GREEN  = Symbol();

function getComplement(color) {
  switch (color) {
    case COLOR_RED:
      return COLOR_GREEN;
    case COLOR_GREEN:
      return COLOR_RED;
    default:
      throw new Error('Undefined color');
    }
}
```

我们还可以使用Symbol消除同名属性

```javascript
const key1 = Symbol('greet');
const key2 = Symbol('greet');
let person = {
    name: "James"
}
person[key1] = function(){
    return this.name
}
person[key2] = function(msg){
    return msg
}
console.log(person[key1]()) //James
console.log(person[key2]("Some Message"))
```
上述代码为`person`添加两个同名的`greet`方法，由于使用`Symbol`方式添加，因此并不会造成同名函数的覆盖。注意，Symbol需要使用`[]`符号进行赋值或访问。

使用Symbol添加的属性不会出现在`Object.keys(person)`、`Object.getOwnPropertyNames()`以及`JSON.stringify()`之中。但是，Symbol也不是私有属性，有一个`Object.getOwnPropertySymbols`方法，可以获取指定对象的所有 Symbol 属性名。

```javascript
console.log(Object.getOwnPropertySymbols(person));
//[ Symbol(greet), Symbol(greet) ]
```


### 迭代器

ES6中的`Iterable Protocols`代表一个interface，实现了该interface（`Symbol.iterator`）的对象具备枚举能力。例如，`Array`:

```javascript
const digits = [0, 1, 2];
const arrayIterator = digits[Symbol.iterator](); //object
```
`digits[Symbol.iterator]()`返回一个类型为`object`的iterator,它从`Arrary Iterator`继承下来，有一个`next()`方法，该方法返回一个`object`包含两个属性`value`和`done`，分别用来表示此次枚举的结果以及是否枚举到数组的末端。对于Map和Set也可获取各自的迭代器

```javascript
console.log(arrayIterator.next()); //{ value: 0, done: false }
console.log(arrayIterator.next()); //{ value: 1, done: false }
console.log(arrayIterator.next()); //{ value: 2, done: false }
console.log(arrayIterator.next()); //{ value: undefined, done: true }
```

### Promise

Promise是一种异步回调模型，ES6中集成了Promise，用法和之前的Promise Framework基本相同

```javascript
const task = new Promise((resolve, reject)=>{
    //do some work
    if(err){
        //failed
        reject(err)
    }else{
        //succeed
        resolve()
    }
})
```

### Proxy


### Generator

Generator是ES6引入的一种被动求值技术

```javascript
//generator function
function* getEmployee() {
    console.log('the function has started');

    const names = ['Amanda', 'Diego', 'Farrin', 'James', 'Kagure', 'Kavita', 'Orit', 'Richard'];

    for (const name of names) {
        console.log(name);
        yield; //new keyword
    }

    console.log('the function has ended');
}
```

Generator和`for...of`

### Resource

- [Udacity ES6 Tutorial](https://classroom.udacity.com/courses/ud356)
- [Udemy ES6 Tutorial](https://www.udemy.com/javascript-es6-tutorial)
- [ECMAScript 6 入门](http://es6.ruanyifeng.com/)

