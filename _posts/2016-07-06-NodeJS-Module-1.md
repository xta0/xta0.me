---
layout: post
list_title: Node.js中的Module（一）| How Module works in Node.js
title: require & module.exports
categories: [Javascript，nodejs]
---

这一篇我们来深入研究下，Node.js中的中的Module是如何实现的

### require syntax

Node.js中使用`require`和`export`来处理module，其调用形式为

```js
//module1.js
function greet(){
    consolo.log('greet!')
}
//app.js
require('./module1.js');
greet(); //wrong
```
按照经验，在`require`了`greet.js`之后，就应该可以访问到这个函数，结果却是不可以，如果想要使用module中的代码需要显式的将其`exports`出来，用的时候需要将其作为函数对象来使用

```js
//module1.js
function greet(){
    consolo.log('greet!')
}
module.exports = greet;

//app.js
var greet = require('./module1.js');
greet(); //greet!
```

这个例子可以看出来，`module.exports`返回的是一个函数对象，而Javascript中的函数对象是多种多样的，比如下面代码返回一个"构造函数"

```js
//module2.js
function House(bedrooms, bathrooms, numSqrt){
    this.bedrooms = bedrooms;
    this.bathrooms = bathrooms;
    this.numSqrt = numSqrt;
    this.log = function(){
        console.log("#bedreooms: "+this.bedrooms + " " +
                    "#bathrooms: "+this.bathrooms + " "+
                    "#size: "+ this.numSqrt)
    }
}
module.exports = House;

//app.js
var House = require('./module2')
var house = new House(2,2,100)
house.log()
```
看到这里有些人或许会有些疑问，这个`require`和`module`是哪里来的呢？有经验的程序员大概会猜到这个是Node中的全局函数，但是`require`是如何工作的呢，`module1.js`中的`module.exports`对象是如何传给`app.js`中的`greet`对象的呢？接下来我们就来回答这个问题。

### IIFE

IIFE是(Immediately Invoked Function Expressions)的缩写，形式如下

```javascript
(function(){
    //function body
}());
```
这种函数被定以后立刻自己执行自己，很长一段时间JavaScript以这种形式来封装Module，其思路大概是import这个函数，传入一些参数，这个函数立刻执行并且只局限在自身的scope，例如我们可以用下面这种方式来封装Module中的函数

```javascript
var object = (function(){
    var average = function(myGrades) {
        var total = myGrades.reduce(function(accumulator, item) {
        return accumulator + item;
        }, 0);
        return'Your average grade is ' + total / myGrades.length + '.';
    };
    return {
        name:"Kevin",
        average: average
    }
})();
console.log(object.average([93, 95, 88, 0, 55, 91]))
```
### Require Behind Sence

有了前面的铺垫，下面可以来回答前面的问题，在node.js中`require`是怎么工作的。还是以上面的代码为例

```js
//module1.js
function greet(){
    consolo.log('greet!')
}
module.exports = greet;

//app.js
var greet = require('./module1.js');
greet(); 
```

接下来从`app.js`的第一句开始，注意JavaScript是门解释型语言，它的`require`并非是一个编译器的预处理符号，而是一个函数，需要被解释器解释执行，可以在`require`处打一个断点，观察Nodejs的执行过程。

```javascript
> var greet = require('./module1');
  greet();

```
断点后，首先进入内部的`require`函数，传入path为当前文件的路径

```javascript
  function require(path) {
    try {
      exports.requireDepth += 1;
      return mod.require(path); //返回mod.require,path是当前文件路径
    } finally {
      exports.requireDepth -= 1;
    }
  }
```

我们最终得到的`var greet`就是`mod.require(path);`这个函数的返回值，继续看看这个函数干了什么事情

```javascript
// Loads a module at the given file path. Returns that module's
// `exports` property.
Module.prototype.require = function(path) {
  assert(path, 'missing path');
  assert(typeof path === 'string', 'path must be a string');
  return Module._load(path, this, /* isMain */ false);
};
```

从这个函数的注释中可知，该函数会加载`path`文件中所有的module，并返回它们的`exports`成员，因此`var greet`的值间接来自`Module._load`的返回值。除此之外，这个函数并未做其它事情，而是直接调用了`Module._load`函数

```javascript
// Check the cache for the requested file.
// 1. If a module already exists in the cache: return its exports object.
// 2. If the module is native: call `NativeModule.require()` with the
//    filename and return the result.
// 3. Otherwise, create a new module for the file and save it to the cache.
//    Then have it load  the file contents before returning its exports
//    object.
Module._load = function(request, parent, isMain) {
    //1. check cache
    //2. check native
    //3. create a new module
    //filename : "xxxx/module1.js"
    //parent: "xxxx/app.js"
    var module = new Module(filename, parent);
    //...
    Module._cache[filename] = module;
    tryModuleLoad(module, filename);
    return module.exports;
}
```
省略掉一些代码，注释上看，这个函数只做三件事，检查缓存中是否有这个module，检查该module是否是NativeModule，所谓NativeModule是Node.js中V8部分的C++代码，显然，这里的`module1.js`不是NativeModule，于是走到了第三步，创建一个新的JS Module，两个参数分别为`module1.js`的路径和当前`app.js`文件的路径。

<mark>这个函数很关键，从这里开始，便有了module对象</mark>，而在创建完module后，将其放入了全局的`Module._cache`中，但此时创建的`module`对象是个空壳，如果访问`module.exports`返回的是一个空对象，因此接下来需要执行load module的操作即`tryModuleLoad`函数，最后将制作好的`module.exports`对象返回给`var greet`对象。

到这里基本流程就走完了，但实际上`tryModuleLoad`的实现非常复杂，其大概思路为读取`module1.js`中的代码，调用JavasScript Core的Native代码(V8)进行求值后返回。对此感兴趣的同学可继续阅读附录中的内容，不想了解细节的同学，看到这里也就足够了。

### More on Require

如果被require的文件过多，可以将他们放到文件夹中，并创建一个`index.js`文件用来索引其它被`require`的文件。假设文件结构如下

```
├── app.js
├── greet
│   ├── English.js
│   ├── Spanish.js
│   ├── config.json
│   └── index.js
```

在`app.js`中可以通过`require './greet'`来间接应用`English.js`和`Spanish.js`两个Module，`require './greet'`会调用默认`require './greet/index.js'`：

```javascript
//index.js
var e = require ('./English')
var s = require ('./Spanish')
module.exports = {
    english:e,
    spanish:s
}
//app.js
var greet = require('./greet')
greet.english();
greet.spanish();
```

### 附录: tryLoadModule的实现

`tryLoadModule`首先会执行`module.load`函数，如下

```javascript
Module.prototype.load = function(filename) {
  //....
  var extension = path.extname(filename) || '.js';
  if (!Module._extensions[extension]) extension = '.js';
  Module._extensions[extension](this, filename);
  this.loaded = true;
  //...
};
```

`extension`为Module文件类型，如果没有指定，则默认为`.js`，另外运行时可以看到`Module._extensions`可支持的文件类型有三种，`.js/.json/.node`。每种类型的文件，对应一个解析的API，接下来会执行`Module.extensions[.js](this,filename)`

```javascript
// Native extension for .js
Module._extensions['.js'] = function(module, filename) {
  var content = fs.readFileSync(filename, 'utf8');
  module._compile(internalModule.stripBOM(content), filename);
};
```

这个函数首先读取了`module1.js`中的代码，接下来调用`module._compile`对这部分代码做了一些预处理，具体为

```javascript
NativeModule.wrap = function(script) {
    return NativeModule.wrapper[0] + script + NativeModule.wrapper[1];
};

NativeModule.wrapper = [
'(function (exports, require, module, __filename, __dirname) { ',
'\n});'
];
```

这里很有意思，所谓的`module._compile`实际上是将module中的代码包装了一层变为了IIFE的形式

```javascript
(function (exports, require, module, __filename, __dirname){
    function greet(){
        consolo.log('greet!')
    }
    module.exports = greet;
})
```
接着将包装后的代码丢给 `vm.js`中的`runInThisContext`来执行，该函数用来和V8通信，返回一个`compilerWrapper`的对象，该对象会来执行`module1.js`中的代码, 即`module.exports = greet`

```javascript
//参数this.exports即为module.exports
result = compiledWrapper.call(this.exports, this.exports, require, this,filename, dirname);

//module1.js
function greet(){
    consolo.log('greet!')
}
> module.exports = greet; //执行这一句
```

（完）

