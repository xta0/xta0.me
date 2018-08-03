---
layout: post
list_title: Some wired design of Javascript
---

### Global Context

```javascript
hello();
console.log(a);

var a = 30;
function hello(){
	console.log("hello!");
}
```
上述代码在Browser中可以正常执行，输出结果为`hello`和`undefined`。如果Javascript是编译执行的话，上述代码肯定会报错，因为在第二句种中，编译器无法找到`a`这个符号；如果是解释执行，则有运行时的
