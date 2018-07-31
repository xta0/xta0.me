---
layout: post
title: CSS Basics
list_title: Web Dev Bootcamp Part 2 | CSS Basics
---

### 历史

- 1996年第一个版本
- 1998年5月第二个版本
- 2007年第三个版本

### Browser work flow 

<img class="md-img-center" src="/assets/images/2007/08/work-flow.png">

### 引入CSS

- **行内样式(inline)**
	- 在HTML标签内直接引入style:`<body style="background-color: orange;">` 
	- 优先级最高，会覆盖`Internal`和`External`指定的css样式

- **内部样式(Internal)**
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

- **外部样式(External)**
	
	- 在`<head>`中加入:

	```HTML
	<link rel="stylesheet" type="text/css" href="css/style.css">
	```
	- 这时浏览器会去请求`css/style.css`这个资源
		- **一个网页可以引入多个css文件，但一般不超过3个**

- 导入样式（不建议用）
	- 另一种引入CSS文件的方式是`@import`, 同样每次获取的时候都要发送一次请求

### Selector定义

```css
selector{
	property: value;
	property: value;
	property: value;
	property: value;
	...
}
```

- **Element Selector**

为指定标签类型应用特定样式

```css
div{
	background: purple;
}	
p{
	color: yellow;
}
```

- **ID Selector** 

为指定标签ID应用特定样式。格式为：`# + 名字`, 一个元素只能有一个ID, 一个页面中可以有多个ID，ID不能重复 

```css
#special {
	color: yellow;
}
```

- **Class Selector**

为具有同一种类型的多个标签应用特定样式, 格式为`. + 类名`

```css
.special {
	color: yellow;
}
```

- **Selector应用**

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
.btn:focus{
	//focus状态下应用某种样式
}
.btn:hover{
	//hover状态下应用某种样式
}
```
### 冲突解决

- 三个来源
	- Author: 开发者指定的属性
	- User: 用户通过浏览器设定的属性（比如字体等）
	- Browser: 浏览器自身的属性(比如`a`标签是蓝色的)  
- 冲突解决顺序
	1. **Importance**（权重）
		- User `!important` declarations
		- Author `!important` declarations
		- Author declarations
		- User declarations
		- Default browser declarations
	2. **Specificity** 
		- Inline styles
		- IDs
		- Classes, pseudo-classes, attribute
		- Elements, pseduo-elements
	3. **Source Order**
		- 最后的声明覆盖之前的声明 

- 小结
	- CSS declarations marked with `!important`有最高优先级
		- 尽量少使用`!important`，不利于代码维护和扩展
	- **Inline Style**比样式表优先级高，同样不利于代码扩展
	- **ID > classes > elements**
	- <strong>*</strong>通配符优先级最低，CSS值为(0,0,0,0)，可被任何样式覆盖
	- **spcificity** 比 书写顺序重要
	- 对于第三方引入的样式表，自己的样式表要放在最后
	
### 属性的继承

- 如果元素某个属性的值没有指定，如果该属性可以被继承，则寻找其父容器中的值
	- 文字的属性可以被继承 
		- `font-family`, `font-size`,`color`,etc.
	- 需要计算的属性可以被继承，声明的固定的值不被继承
- 使用`inherit`关键字可以强制继承某属性

## Box Model

元素layout规则

<img class="md-img-center" src="/assets/images/2007/08/box-model.png">

- 涉及到
	- **Dimensions of boxes**: the box model
	- **Box type**: `inline`, `block` and `inline-block` 
	- **Positioning scheme**: `floats` and `positioning`
	- **Stacking contexts**
	- Other elements in the render tree.
	- Viewport size, dimensions of images, etc.

- Box-Sizing
	- border-box
		- 元素宽高计算不包括padding和border
		- **total width** = specified width 
		- **tottal height** = specified height

- Box model
	- **total width** = right border + right padding + specified width + left padding + left border
	- **tottal height** = top border + top padding + specified height + bottom padding + bottom border
	- **margin**
		- 参数顺序
			- top,left,bottom,right
		- auto
			- 在auto方向上将剩余空间两边均分，即在auto方向上居中
			- `margin: 0 auto 0 auto`：垂直方向margin为0，水平方向居中
			- `margin: 0 auto`：垂直方向margin为0，水平方向为`auto`

- Box Type
	- 行内元素(Inline)
		- `display: inline` 
		- 元素行在内显示，自适应宽高
		- 无法应用`height`和`width`
		- `padding`和`margin`只在水平方向有用
	- 块级元素(Block-Level)  
		- `display: block`
		- 默认撑满父级元素的宽度(100% parent's width)
		- 元素间垂直排列
		- 盒模型算法适用，可指定`width`,`height`
	- Inline-Block
		- 可以在行内使用的块级元素，兼具上述两种元素的特征
		- 元素行在内显示，自适应宽高
		- 盒模型算法适用，可指定`width`,`height`
	
- Positions
	- **Relateive**
		- `position: relative`
		- 元素默认的Position方式，相对于自己的位置，类似layer的anchorPoint
		- **NOT** floated
		- **NOT** absolutely positioned
		- Elements laid out according to their source order
	- **Absolute**
		- `position: absolute`,`postion: fixed`
		- 相对于父容器的位置，使用`top`,`bottom`,`left`,`right` 指定相对父容器的offset
		- 父容器需显式指定`position`的值，如果父容器没有指定`position`属性，则以浏览器为父容器
	- **Float**
		- 元素脱离Normal Flow，用来块级元素转成行内元素
		- 保留元素的宽，高度自适应，`height`属性失效
		- `float:left`,`float:right` //左上角对齐，右上角对齐
		- [All About Floats](https://css-tricks.com/all-about-floats/)

### 单位换算

![](/assets/images/2007/08/value-processing.png)

- CSS每种属性都有一个inital value
- 浏览器会指定一个默认的**root font size** （通常是16px）
- 百分比和相对单位会被转化成像素单位
- 百分比
	- 如果修饰`font-size`，是相对于父类`font-size`的值
	- 如果修饰`length`上，是相对于父类`width`的值
- `em`
	- 如果修饰`font-size`，是相对于父类`font-size`的值
	- 如果修饰`length`，是相对于当前的`font-size`的值	
	- 相对于`document's root`的`font-size` 
- `vh`和`vw`
	- 相对于viewport's `height`和`width` 


## Resources

- [MDN](https://developer.mozilla.org/zh-CN/)
- [WebPlatform.org](https://docs.webplatform.org/wiki/Main_Page)
- [W3C](http://www.w3.org/)
- [Can I Use](http://caniuse.com/)
- [CSS Zen Garden](http://www.csszengarden.com/)
- [FlexBox Guide](https://css-tricks.com/snippets/css/a-guide-to-flexbox/)
- [Style Guide](http://codeguide.co)
- [BEM](http://getbem.com/naming/)
- [7-1 Pattern](https://gist.github.com/rveitch/84cea9650092119527bc) 
- [All About Floats](https://css-tricks.com/all-about-floats/)


## 附|常用元素的属性

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
