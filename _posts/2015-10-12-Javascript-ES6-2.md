---
layout: post
list_title: Javascript ES6 Quick Reference Guide Part 2
title: Javascript ES6 Part 2
categories: [JavaScript]
---

### Symbol

Symbol是新加入的原生数据类型，用来解决属性名冲突的问题，Symbol 值通过Symbol函数生成。这就是说，对象的属性名现在可以有两种类型，一种是原来就有的字符串，另一种就是新增的 Symbol 类型。凡是属性名属于 Symbol 类型，就都是独一无二的，可以保证不会与其他属性名产生冲突。

```javascript
const sym1 = Symbol('banana');
const sym2 = Symbol('banana');
console.log(sym1 === sym2); //false
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



### Resource

- [Udacity ES6 Tutorial](https://classroom.udacity.com/courses/ud356)
- [Udemy ES6 Tutorial](https://www.udemy.com/javascript-es6-tutorial)
- [ECMAScript 6 入门](http://es6.ruanyifeng.com/)

