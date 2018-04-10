---
layout: post
title: Web Bootcamp Part 2
---


## CSS

- [MDN](https://developer.mozilla.org/zh-CN/)
- [WebPlatform.org](https://docs.webplatform.org/wiki/Main_Page)
- [W3C](http://www.w3.org/)
- [Can I Use](http://caniuse.com/)
- [CSS Zen Garden](http://www.csszengarden.com/)
- 1996年第一个版本
- 1998年5月第二个版本
- 2007年第三个版本

### 引入CSS

- 行内样式(inline)
	- 在HTML标签内直接引入style:`<body style="background-color: orange;">` 
	- 优先级最高，会覆盖`Internal`和`External`指定的css样式

- 内部样式(Internal)
	- 定义在`<head>`中:

```html	
<style type="text/css">
	header{ background-color: orange;}
	p {
		font-size: 20px;
		font-weight: bold;
	}
	h1{
		font-size: 90px;
		color: red;
	}
</style>
```

- 外部样式(External)
	- 在`<head>`中加入:`	<link rel="stylesheet" type="text/css" href="css/style.css">`
	- 这时浏览器会去请求`css/style.css`这个资源，**一个网页可以引入多个css文件，但一般不超过3个**

- 导入样式（不建议用）
	- 另一种引入CSS文件的方式是`@import`, 同样每次获取的时候都要发送一次请求

### Selector定义

- 定义:

```css
selector{
	property: value;
	property: value;
	property: value;
	property: value;
	...
}
```


- **Element Selector**，为指定标签类型应用特定样式

```html
<div>
	<p>You say yes</p>
</div>
```
	
```css
div{
	background: purple;
}
	
p{
	color: yellow;
}
```

- **ID Selector** 为指定标签ID应用特定样式。格式为：`# + 名字`
 - 一个元素只能有一个ID
 - 一个页面中可以有多个ID，ID不能重复 

```html
<div>
	<p id="special">You say yes</p>
</div>
```
	
```css
div{
	background: purple;
}
	
#special {
	color: yellow;
}
```

- **Class Selector**: 为具有同一种类型的多个标签应用特定样式, 格式为`. + 类名`:

```html
<div>
	<p class="special">You say yes</p>
</div>
```
	
```css
div{
	background: purple;
}
	
.special {
	color: yellow;
}
```

- 其它使用selector的方式

```css
/*Star*/
* {
	border: 1px solid lightgrey;
}

/*Descendant Selector*/
/* li下所有a标签 */
li a {
	color: red;
}

/*Adjacent Selector*/
/* 和h4相邻的标签 */
h4 + ul {
	border: 4;
}

/*Attribute Selector*/
/*所有checkbox的背景是蓝色的*/
input[type="text"]{
    background-color: gray;
}

/*n-th-type*/
/*所有li的第三项*/
li:nth-of-type(3){
	background-color: red;
}

/*所有li的奇数项*/
li:nth-of-type(even){
	background-color: red;
}

/*使用Pseudo-Class*/
/*Make the <h1> element's color change to blue when hovered over */
h1:hover{
    color: blue;
}

/*Make the <a> element's that have been visited gray */
a:visited{
    color: grey;
}
```

- **优先级**: 
	1. @import  
	2. inline selector
	3. ID selector  
	4. Class selector, Attribute Selector, Pseudo-Class Selector
		- `.hello {}`
		- `input[type="text"]{}`
		- `input:email{}` 
	5. Type Selector
		- `li + a{}`
		- `li a{}`
		- `li {}` 
	6. 伪对象
	7. 父类  
	8.  通配符 
	


### 颜色

- 使用内置关键字 `h1{ color: red; }`
- 使用十六RGB进制`h1{ color: #123456; }`
- 使用十RGB进制`h1{ color: rgb(100,0,100); }`
- 使用十RGBA进制`h1{ color: rgba(100,0,100,0.6); }`
	

### 背景和边框

```css
h2 {
    /* 颜色 */
	background: orange;
	/* 背景图片 */
	background: url(http://xxx.jpg);
	background-repeat: no-repeat;
    background-size: cover;
    /* 边框 */
    border-color: #111111;
    border-width: 5px;
    border-style: solid;
    border: 5px dashed #444444;
}
```
### 字体和文本

- 文本:
	- 字体：`font-family`
	- 字号：`font-size`
		- `px`:像素值
		- `em`: 相对于当前字号的倍数,`em: 0.5`表示被修饰的字号是当前字号的0.5倍  
	- 粗体：`font-weight`
		- `normal`,`bold`,`100-800` 
	- 行高
		- `line-height: 2` 表示行高是字体高度的两倍
		- `line-height: 200%`代表 2*字体高度
	- 对齐：`text-align：right` 
	- 缩进：`text-indent`，单位为em，使用的时候要注意em的大小
	- 修饰：`text-decoration`
		- `line-through`
		- `underline`  
	- 折行：
		- `word-wrap`：
			- `normal`：允许内容顶开或溢出指定的容器边界
			- `break-word`：内容在边界换行，如果需要，内部允许断行 
		- `white-space`:
			- `nowrap` 不让换行，直到br
			- `pre` 用等宽字体显示预先格式化的文本，不合并文字间的空白距离，当文字超出边界时不换行
			- `pre-wrap`：用等宽字体显示预先格式化的文本，不合并文字间的空白距离，当文字碰到边界时换行
			- `pre-line`：保持文本换行，不保留文字间的空白距离，当文字碰到边界时发生换行
		- `overflow`
			- `hidden`：当前对象内文本超出的部分隐藏 
- 自定义字体
	- 在`<header>`里面引入：`<link href="https://fonts.googleapis.com/css?family=Roboto" rel="stylesheet">`
	- [google font](https://fonts.google.com/) 

### 列表和表格
	
- 列表 	
	- `list-style-type`通常用默认的

- 表格
	- `border-collapse` 
		- `separate`：边框独立
		- `collapse`：相邻边被合并 



### 盒子模型

![](/assets/images/2007/04/box-model.png)

- `margin`
	- `margin: 20px 40px 500px 100px`：top,right,bottom,right
	- 在auto方向上将剩余空间两边均分，即在auto方向上居中
		- `margin: 0 auto 0 auto`：垂直方向margin为0，左右方向居中
		- `margin: 0 auto` => `top-bottom=0`, `left-right=auto`


- 块级元素和行内元素的转换
	- **块级元素转行内元素**，使用`inline`。如果使用了`inline`会忽略元素的`width`和`height`属性
	
	```html
	<div style="display:inline;">123</div>
	<div style="display:inline;">abc</div>
	```
	- **行内元素转块级元素**，使用`block`。如果使用了`block`会忽略元素的`width`和`height`属性
	
	```html
	<span style="display: block;">123</span>
	<span style="display: block;">abc</span>
	```

- **使用float**
	-  实现块级元素转成行内元素，保留元素的宽，高度自适应
	- `float:left`：向左对齐
	- `float:right`：向右对齐

- **使用position**
	- `relative`:相对于自己的位置，类似layer的anchorPoint
	- `absolute`:相对于父容器的位置，如果父容器没有指定`position`属性，则以浏览器为父容器
	- 如果有元素的`position`设为`inline-block`，那么所有inline的元素都底部对齐


### H5新增布局

```
<div id = "header">
<div id = "nav">
<div class="article"> <div class="sidebar">
<div class="section"> <div class="address">
<div id="footer">

```