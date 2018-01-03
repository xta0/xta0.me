---
title: Web Dev 
layout: post
categories: [web,html,css,javascript]
meta: Web Development Bootcamp. Basic HTML/CSS/Javascript usage
---

> 转载请注明出处

## HTML

### 概述

- <!DOCTYPE html> : HTML
	- DTD文档模型：
		- 过渡的(Transitional): 要求非常宽松，使用HTML4.0.1标识
		- 严格的(Strict)：不能使用任何表现层的标识。如`<br>`
		- 框架的(Frameset): 页面中包含框架，需要采用这种DTD

	- 一般来说，后两种很少见，第一种主要应用于PC时代
	- 对于HTML5来说，DTD文档模型为`<!DOCTYPE html>`即可  
	
- 基础标签: 

	- `<head>`包含meta信息，不可见
		- head中的`<title>`出现在浏览器的标签位置和搜索结果
	- `<meta>`:
		- 设置文档编码
			- PC WEB: `<meta http-equiv="Content-Type" content="text.html;charset=UTF-8">` 
			- HTML5:`<meta charset="UTF-8">`
		- 设置关键字和描述方便SEO
			- `<meta name="keywords" content="关键字">`
			- `<meta name="description" content="描述">`
		- 设置移动开发所发的比例
			- `<meta content="width=device-width,initial-scale=1.0,maximum-scale=1.0,user-scalable=0;"name="viewport"/>` 
	- `<body>`:可见内容

- 标签的属性
	- `class`:元素类名： `<h2 class="h2ID"> haha </h2>`
	- `id`:元素唯一ID：- `<h2 id="h2ID"> haha </h2>`
	- `style`:规定元素样式, 例如指定css的样式表
	- `title`:规定元素额外信息

### HTML的布局

- HTML行内元素和块级元素
	- 块级元素（换行）:
		- `div, dl/dt/dd, form`
		- `h1-h6标题, hr-水平分割线`
		- `ol - 排序表单`
		- `ul - 非排序表单`
		- `p - 划分段落`：p标签上下会有空行  
		- `table - 表格`

	- 行内元素（非换行）:
		- `a,br,img，b,i,u,em`
		- `input,label,select,span`
		- `特殊符号(&nbsp等)`

- `<div>`和`<span>`
	- `<div>`是通用的块级元素
	- `<span>`是通用的行内元素 

### HTML常用标签

- 标题`<h>`标签
	- 块级(block level)元素，自动换行 
	- 从`h1 ~ h6` 

- 段落`<p>`标签
	- 块级元素，自动换行

- 列表`<ol>`,`<ul>`标签
	- 块级元素，自动换行
	- 子元素标签为`<li>`，可互相嵌套

```html
<ol>
    <li>Red</li>
    <li>Orange</li>
    <li>Yellow</li>
        <ul>
            <li>sunflowers</li>
            <li>bananas</li>
        </ul>
</ol>
```

- 链接`<a>`标签
	- 行内元素，不换行 
	- 内容是文字：`<a href="https://www.gmail.com/">gmail</a>`
	- 内容是图片：`<a href="https://twitter.com"><img src="facebook_icon.png"></a>`
	- 锚点：
		- body内跳转:`<a href="#here">Go here</a>`
		- 跳转到另一个webpage:`<a href="web_page.html#here">Go here</a>`


- 自闭合标签
	- `<meta name="author" content="">`
	- `<a>`
	- `<img>`
	- `<input>`
	- `<br><wbr><hr>`
- 文字格式化标签:	
	-  行内元素(inline elements),不换行
	- `<b>` <b>粗体</b>
	- `<big>`<big>字体加大</big>
	- `<em>`<em>定义着重文字</em>
	- `<i>`<i>定义斜体</i>
	- `<small>`<small>小号字</small>
	- `<strong>`<strong>加重语气</strong>
	- `<sub>`test<sub>定义下标</sub>test
	- `<sup>`test<sup>定义上标</sup>test
	- `<ins>`<ins>插入字符</ins>
	- `<del>`<del>删除字符</del>



- Image: `<img src="ac.png" width="30" height = "100"/>`
- Audio: `<audio src="a.mp3" autoplay controls loop> </audio>`
- Video: `<video src="a.mp4" autoplay controls loop></video>`

### HTML标签属性

属性是为标签增加额外信息的方式。格式为`<tag name="value"></tag>`

### HTML表格

- 表格`<table`>标签:
	- structure: `<table>, <thead>, <tbody>`
	- 表头: `<th>`
	- body: `<tr><td>`
- 表格样式：
	- Table borders: `border`
	- Table width: `width`
	- Table height: `height`
	- Vertical alignment: `vertical- align`
	- Table padding: `padding`

- 表格结构

```html
<table>
	<thead>
		<tr>
			<th> ... </th> //column
			<th> ... </th>
			<th> ... </th>
		</tr>
	</thead>
	<tbody>
		<tr> <td>...</td> <td>...</td> <td>...</td> </tr>
		<tr> <td>...</td> <td>...</td> <td>...</td> </tr>
	</tbody>
</table>
```


### HTML表单创建

表单用于获取不同类型的输入

- `<form></form>`标签是一个不可见的表单容器，里面包含其它元素
	- action : the URL to send from data to
	- method: the type of HTTP request
		
```html
<form action="/my-form-submiiting-page" method="post">
	<!-- All our inputs will go in here -->
</form>
```
	
- `<input>`标签用来和用户进行交互,“type”属性用来表示具体类型
	- 文本：`<input type="text">`
		- `placeholder`属性，默认文案
		- `name`属性，发送给表单的data 
	- 密码：`<input type="password">`
	- 复选框：`<input type="checkbox">` 
	- 单选框：`<input type="radio" name="sex">`
	- 按钮：`<input type＝"button" value="tap">`
	- 提交：`<input type＝"button" value="confrim">`

- `<label>`标签用来提示`<input>`标签的文字
	- 直接嵌套：`<label>Username:<input type="text"></label>` 
	- 使用`id`，`for`属性互相绑定 :`<input id="abc" type="text"> <label for="abc">username</label>`


- 表单校验：使用`require`关键字
	- `<input type="email" required>`校验输入的是email格式
	- `<input type="password" pattern=".{5,10}" required title="pwd must be between 5 and 10 characters">` 校验密码在5到10个字符


- 使用radio button

```html
<!-- 相同name保证同一个form内radio button互斥 -->
<!-- button元素位于form最后一个，click默认为提交表单操作 -->
<!-- value属性值为提交表单选中的data值，key为name属性的值{name:value} -->

<form action="">
    <label for="dogs">Dogs</label>
    <input type="radio" name="petChoice" id="dogs" value="DOGS">

    <label for="cats">Cats</label>
    <input type="radio" name="petChoice" id="cats" value="CATS">
    
    <button>Go!</button>
</form>

``` 

- 下拉组件`<select>`标签(块级元素)
	
```html
<!-- value属性值为提交表单选中的data值，key为name属性的值{name:value} -->
<form>
	<select name="platform">
		<option value="ios">iOS</option>
		<option value="android">Android</option>
		<option value="wp">Windows Phone</option>
	</select>
</form>
```

- 文本输入组件`<textarea>`(行内元素 )
	
```html
<!-- value属性值为提交表单选中的data值，key为name属性的值{name:value} -->
<textarea name="comment" cols="30" rows="30">
	fill info
</textarea>
```

- 常用表单标签

表格  | 输入
-----| ----
`<form>`    | 表单
`<input>`  | 输入域
`<textarea>`  | 文本域
`<label>`  | 控制标签
`<fieldset>`  | 定义域
`<legend>`  | 域标题
`<select>`  | 选择列表
`<optgroup>`  | 选项组
`<option>`  | 下拉列表
`<button>`  | 按钮
`<che>`  | 按钮


- 简单例子

```html
<form action="/sign-in-url" method="post">
    <input type="text" name="username" >
    <input type="password" name="password">
    <input type="submit" >
</form>

```


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


## Bootstrap

- [Bootstrap Instroduction](https://getbootstrap.com/docs/4.0/getting-started/introduction/)
	- One single CSS file + One single Javascript file

### Grid System

- Bootstrap将每个容器分成12个column，每个显示区域可以指定占有column的数量。

```html
<!-- 使用boostrap container -->
<div class="container">
   <!-- div横向展示 -->
    <div class="row">
        <!-- column-large-2 占两个格子 -->
        <div class="col-lg-2 pink">COL LG #2</div>
        <div class="col-lg-8 pink">COL LG #10</div>
        <div class="col-lg-2 pink">COL LG #2</div>
    </div>
    <div class="row">
        <div class="col-lg-4 pink">COL LG #4</div>
        <div class="col-lg-4 pink">COL LG #4</div>
        <div class="col-lg-4 pink">COL LG #4</div>
    </div>
</div>
```

- 四种不同尺寸

<table>
    <thead>
        <tr>
            <th></th>
            <th> 手机屏幕(<768px)</th>
  			  <th> 平板屏幕(≥768px)</th>
  			  <th> 中等屏幕PC (≥992px)</th>
  			  <th> 大屏幕PC(≥1200px)</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <th class="text-nowrap" scope="row">Grid behavior</th>
            <td>Horizontal at all times</td>
            <td colspan="3">Collapsed to start, horizontal above breakpoints</td>
        </tr>
        <tr>
            <th class="text-nowrap" scope="row">Container width</th>
            <td>None (auto)</td>
            <td>750px</td>
            <td>970px</td>
            <td>1170px</td>
        </tr>
        <tr>
            <th class="text-nowrap" scope="row">Class prefix</th>
            <td><code>.col-xs-</code></td>
            <td><code>.col-sm-</code></td>
            <td><code>.col-md-</code></td>
            <td><code>.col-lg-</code></td>
        </tr>
        <tr>
            <th class="text-nowrap" scope="row"># of columns</th>
            <td colspan="4">12</td>
        </tr>
        <tr>
            <th class="text-nowrap" scope="row">Column width</th>
            <td class="text-muted">Auto</td>
            <td>~62px</td>
            <td>~81px</td>
            <td>~97px</td>
        </tr>
        <tr>
            <th class="text-nowrap" scope="row">Gutter width</th>
            <td colspan="4">30px (15px on each side of a column)</td>
        </tr>
        <tr>
            <th class="text-nowrap" scope="row">Nestable</th>
            <td colspan="4">Yes</td>
        </tr>
        <tr>
            <th class="text-nowrap" scope="row">Offsets</th>
            <td colspan="4">Yes</td>
        </tr>
        <tr>
            <th class="text-nowrap" scope="row">Column ordering</th>
            <td colspan="4">Yes</td>
        </tr>
    </tbody>
</table>

- 不同比例间缩放

```html
<div class="container">
    <div class="row">
        <div class="col-lg-3 col-md-3 col-sm-6 pink">VIZLAB</div>
        <div class="col-lg-3 col-md-3 col-sm-6 pink">VIZLAB</div>
        <div class="col-lg-3 col-md-3 col-sm-6 pink">VIZLAB</div>
        <div class="col-lg-3 col-md-3 col-sm-6 pink">VIZLAB</div>
    </div>
</div>
```

1. 第一层`div`类型为bootstrap容器
2. 第二层`div`类型为`row`，子元素横向布局
3. 第三层`div`有三个类型，分别为
	- 大屏下和中屏下，每个元素占屏幕的3个格子，1/4
	- 小屏下，每个元素占6个格子

- 嵌套

```html
<div class="col-lg-3 col-md-3 col-sm-6 pink">VIZLAB
    <div class="row">
        <div class="col-lg-6 orange">FIRST HALF</div>
        <div class="col-lg-6 orange">FIRST HALF</div>
    </div>
</div>
```
1. 嵌套父容器遵从12格子等分
2. 嵌套子元素按12各自划分

  

## Javascript Basics

### 资源

- [MDN](https://developer.mozilla.org)

### 引入JS的三种方式

- 在`<head></head>`中引入

- 一般在`<body>`末尾插入外联的js文件:

```html
<body>
	
	<div class="container">
		<h1>Hello world</h1>
	</div>		

	<script src="scripts/script.js"></script>
	
</body>
```

### Debug

- chrome: `cmd+alt+j` 

### 数据类型

- 5种基本数据类型
	- number
	- string
	- boolean
	- null
	- undefined

- **Variable**
	- Syntax	
		- `var score = 0;`
	- 数字，字母，下划线，`$`符号开头
	- 类型检查
		- `typeof`关键字
		- `typeof(111); //"number"`
		- `typeof({name:'abc'}); //"object"` 
		- `typeof(undefined); //"undefined"`


- **Numbers**
	- Integer，Float
	- 科学计数法:`9e-6`,`9e+6`;
	- 数学库:`Math`
		- 四舍五入:`Math.round(4.5);` 
		- 向下round:`Math.floor(4.5); => 4`
		- 向上round:`Math.ceil(4.5); => 5`
		- 随机数:`Math.random();` 返回[0-1)的随机小数
			- [x,y]之间的数：`var a =Math.floor(Math.random()*(y-x+1))+y`; 
	
	-  Number转String:`String(x);`

- **Stings**
	- 获得字符
		- `"hello[0]" //"h"` 
	
	- Concatentation:
		- `one string` + `another string`
		- `var name = "Dave"; var message = "Hello "+name` 
	
	- Length: 
		- `a.length` 
		
	- uppercase / lowercase : 
		- `a.toUpperCase(); a.toLowerCase();` 
	
	- 字符转数字:
		- `parseInt('10');`	
		- `parseFloat('3.14');`
		- `parseInt('100 badges'); => 100`
		- `parseInt('badges100'); => Nan`
		- `Number(string)`
	
	- 检查字串，返回position
		- `string.indexof("text")`  

	- 字符串替换
		- `string.replace(\regex\g, otherString)` 返回一个新的string，不修改原string 
	
	- Escape Character
		- `\`开头
		- `"Singing\"Merry Chrismas\""`
		- `"This is a backslash \\"` 


- **Boolean**
	- A Boolean value can only be `true` or `false`


- **Null/Undefined**
	- `Undefined`: 变量定义未初始化
		- `var name`
	- `null`: 明确变量值为空

### 数据结构

- **Arrays**
	- 创建数组:一个Array可以存放不同类型的object

	```javascript
	var scores = [100, 99, 99, 72, 60];
	var names = ['Joan', 'Amit', 'Sarah', 'Ricardo', 'Piers'];
	var values = [1, -100, true, false, 'JavaScript'];
	```

	- 访问数组:使用下标
		- `var t = scores[0] //100 `
	- `length/index`
		- `scores.length(); //5` 
		- `scores.indexOf(100); //9`
		- `scores.indexof(999); //-1 不存在的元素index返回-1`
	- `push/pop`: 尾部操作
		- `scores.push('a','b'); //尾部追加`
		- `scores.pop(); //尾部移除b`
	- `shift/unshift`:头部操作 
		- `scores.unshift('a') //头部插入，[a,a,b];`
		- `scores.shift(); //头部删除，[a,b]`
	- `splice`
		- 删除操作
			- API:	`array.splice(index,length)` 原数组被修改，返回一个新数组
				- `scores.splice(1,1); => [99]`
				- `scores.splice(1,0); => [  ]`
		- 插入操作
			-  API:`array.splice(position,0,element)`，原数组修改，返回一个空数组
				- `scores.splice(1,0,66); => [100, 66, 99, 99, 72, 60]`
				- `scores.splice(100,0,66); => [100, 66, 99, 99, 72, 60,66]` 越界默认插入到最后
		- 替换操作
			- API: `array.splice(position,length,element)`
				- `names.splice(1,1,'Jayson') =>[ 'Joan', 'Jayson', 'Sarah', 'Ricardo', 'Piers' ]`
			
	- `slice`：返回子数组的copy
		- `names.slice(1) => ['Amit', 'Sarah', 'Ricardo', 'Piers']`
		- `names.slice(); //返回整个数组的copy`
		- `names.slice(1,3);  => ["Amit", "Sarah"] //1,返回3-1=2个元素的数组 2，数组的第一个元素是names[1]`

	- `join`：链接元素，返回string 
		- `names.join(','); => "Joan,Amit,Sarah,Ricardo,Piers"`
		- `scores.join(','); => "100,99,99,72,60"`
	- `concat`:链接两个数组
		- `var list = list1.concat(list2);`
	- 二维数组:
		- `var list = [[12,3],[2,3,4],[11]]`
	- 排序:
		- `scores.sort()`
	- 翻转:
		- `scores.reverse()`

	- `foreach`遍历数组
		- API: `array.foreach(function(ele,idx,array){...})`
		
		```javascript
		// 参数个数不固定，按照参数顺序推断参数类型
		scores.forEach(function(ele){ 
			console.log(ele); //100,66,...
		})
		scores.forEach(function(ele,idx){ 
			console.log(idx); //0,1,2... 
		})
		```
		- 自定义`foreach`
		
		```javascript
		Array.prototype.myForEach = function(func){
			for(var i=0; i<this.length; i++){
				func(this[i]);
			}
		}
		```
	- Map
		- `map`:`array.map(function)` map产生一个新的array，并不会修改原来的array

- **Object**
	- 创建Object
		- `var student = { name: "Dave", grades: [1,23,4] };`
		- `var student={}; student.name="Dave"; student.grades:[1,23,4]`
		- `var student = new Object(); student.name="Dave"; student.grades:[1,23,4]`
	- 成员变量:
		- 访问：
			- `student['name']`
			- `student.name` 语法糖
		- 更新：
			- `student.name = "Tim"` 
		- 遍历：
		
		```javascript
		for(var key in object)
		{
			var value = object[key];
			//wrong: var value = object.key 
		}
		``` 
	- 成员方法
		- `student.method = function(var1,var2){...}` 

- More on Javascript
	- [Javascript Notes]() 
	


### 表达式

- **比较运算**
	- `>`,`>=`,`<`,`<=`
	- `==``!=`：非严格类型比较，转成相同类型比较值
		- `("3" == 3) => true` 
		- `("" == 0) => true`
		- `var y=null; y == undefined //true`
		- 特别的例子
			- `true == "1" //true`
			- `0 == "false" //true`
			- `null == undefined //true`
			- `NaN == NaN //false, not comparable` 
	- `===``!==`: 严格类型比较，比较值和类型
		- `("3" === 3) => false` 
		- `var y=null; y===undefined //false`

- **逻辑运算**
	- `&&, ||, !`
	- 所有值可比较，自带`true`或`false`的属性
		- `!"Hello World" //false`
		- `!"" //true`
		- `!null //true`
		- `!0 //true`
		- `!-1 //false`
		- `!NaN //true`
	- 自身错误属性的值：`false, 0, "", null, undefined, NaN` 

- **条件控制**
	- **使用if**:
	
	```javascript
	if()
	{...}
	else if()
	{...} 
	else{...}`
	```
		
	- **使用switch**
	
	```javascript
	switch(var)
		case:"option_!":
			do_something();
			break;
		default:
		break;
	```


- **循环语句**
	- **使用while**
	
	```javascript
	while()
	{
		...
		break;
	}
	```
	
	- **使用do-while**
	
	```javascript 
	do{
		...
	}
	while( ... break;)
	```
	
	- **使用for**
		- 使用index和list：
		
		```javascript
			for(var i=0;i<list.length;i++){
				...
				break;
			}
		```
		
		 - 循环index：
	
		```javascript
			for(var index in list){
				...
			}
		```
		
		- 循环value
	
		```javascript
			for (var value in list){ 
				... 
			}
		```


### Function

- 定义
	- `function myFunc() {...} //function as declaration`
	- `var myFunc = function(){...} //function as first class object`

- 返回值
	- return function from function

	```javascript
	function counter(){
		var count = 0;
		return function(){
			count ++;
			alert(count);
		}
	}
	var count = counter();
	count();
	count();
	count();
	```
	
- HOF
	- Function as argument
	
	```javascript
	setInterval(function(){
		console.log("fired");
	},2000);	
	```

## DOM(Document Object Model)

### Overview

![](/assets/images/2007/04/dom.png)

- 访问`document`对象
	- `console.dir(document)`
	- `document.URL`
	- `document.head`
	- `document.body`

- 修改DOM属性

```javascript
var h1 = document.querySelector("h1")
h1.style.color = "orange";
```

### 访问DOM节点

- 几种方式
	- `document.childNodes[i].childNodes[i].childNodes[i]...`
	- `document.getElementsByTagName()`
		- `var tags = document.getElementByTagName("li"); //返回array` 
	- `document.getElementById()`
		- `var tag = document.getElementById("some_id"); //返回唯一Id标签object` 
	- `document.getElementByClassName()`
		- `var tags = document.getElementByClassName("some_class"); //返回某个class的数组`
	- `document.querySelector()`
		- `selector`的值为CSS的标签，可以为`tag`，`id`和`class` 
		- `var tag = document.querySelector('#highlight');` 
		- `var tag = document.querySelector(".bolded");  //返回第一个匹配的tag`
		- `var tag = document.querySelector("li a.special")`
	- `document.querySelectorAll()`
		- `var lis = document.querySelectorAll("li"); //返回所有li标签` 

### 改变节点Style

- 直接修改节点属性
	
```javascript
var node = getElementById("id")
node.style.color = "blue";
node.style.border = "10px solid red";
node.style.fontSize = "70px";
node.style.background = "yellow";
node.style.marginTop = 20px
node.setAttribute("style":"color:red")
``` 
- 使用样式表替换当前节点属性
	
```javascript
.some-class{
	color: blue;
	border: 10px solid red;
}
var tag = document.getElementById("highlight");
//ADD THE NEW CLASS TO THE SELECTED ELEMENT
tag.classList.add('some-class');
	
//REMOVE A CLASS
tag.classList.remove('some-class');
	
//TOGGLE A CLASS
tag.classList.toggle("some-class");
```
	
- 改变文字
	- 使用`textContent`，返回的文字中不包含HTML标签

	```javascript
	<p>This is an <strong>awesome</strong> paragraph </p>
	var p = document.querySelector("p");
	p.textContent;  //"This is an awesome paragraph ";
	```	
	
	- 使用`innerHTML`会保留文字的HTML标签
	
	```javascript
	<p>This is an <strong>awesome</strong> paragraph </p>
	var p = document.querySelector("p");
	p.innerHTML;  //"This is an <strong>awesome</strong> paragraph "	```	
	
- 改变**Attributes**

使用`getAttribute()`和`setAttribute()`来读写属性

```HTML
 <a href="http://www.google.com">I am a link</a>
 <img src="logo.png" >
```

```javascript
var link = document.querySelector("a");
link.getAttribute("href");
link.setAttribute("href","www.baidu.com");

var img = document.querySelector("img");
img.setAttribute("src","./dog.png");

```
	
	
### DOM 节点操作

- 创建DOM节点
	- `result = document.createElement("div");`
	- `result = document.createTextNode("Hello")`

- 添加DOM节点
	- `document.getElementById("id").appendChild(node)` 

- 删除节点：
	- `document.getElementId("id").parentNode.removeChild(node)` 
	- 三种方式：
		- `var node = document.getElementById("first"); node.parentNode.removeChilde(node);`
		- `var node = document.getElementsByTagName("p")[0]; node.parentNode.removeChild(node)`
		- `var parentNode = document.getElementById("body"); parentNode.removeChild(parentNode.firstChild)` 

- Clone DOM节点
	- Copying a node : `node.cloneNode()`
	- Copying a branch : `node.cloneNode(true)`
	- Adding nodes : `dest.appendChild()`


### DOM Event


- Timers
	- Timers are very useful for dynmic web page behavior
	
	```javascript
	var timer;
	timer = setTimeout(do_something, 1000);
	```
	- `do_something()` will be executed 1 second later
	- The value 1000 is in milliseconds, so 1000 = 1 second
	- stop timer:

	```javascript
	clearTimeout(the_timer)
	```

	- `setInterval()` repeatedly does something

	```javascript
	var the_timer;
	the_timer = setInterval(do_something,2000)
	```

	- `do_something()` will be executed every 2 seconds。 To stop it:

	```javascript
	clearInterval(the_timer);
	```

- Event Listener
	- Adding an event to `<body>`
	
		```html
		<html>
			<head>
				<script>
					function do_something(){ alert("Page has loaded");}
				</script>
			</head>
			<body onload="do_something()"></body>
		</html>
		```
	
	- Adding an event to `window`

	```html
	<html>
		<body id="thebody">
			<script>
				function do_something(){ alert("Page has loaded");}
				window.onload = do_something;
				window.addEventListener("load",do_something);
			</script>
		</body>
	</html>
	```
	
	-  Adding an event to `element`
		- API : `element.addEventListener(type,functionToCall)`
	
		```javascript
		var button = document.querySelector("button");
		button.addEventListener("click", function(){
			console.log("SOMEONE CLICKED THE BUTTON!")
		});
		```
	
- If you have more than one event handler
	- event handlers are stored in an array
	- when an event happens, all the handlers are executed one by one 
	- They are executed in the order they added

- To remove an event handler

```html
var theBody = document.getElementById("theBody");
theBody.removeEventListener("load",do_something);
```

- remove只能remove通过js添加的listener

## JQuery

### 资源

- [API](api.jquery.com)
- [You Might Not Need JQuery](http://youmightnotneedjquery.com/)
- [What is the future of jQuery for 2017 and later?](https://www.quora.com/What-is-the-future-of-jQuery-for-2017-and-later)

### 定义

- jQuery: 一个Javascript函数库，封装了对不同浏览器适配，以及一些常用的JS API。
- jQuery库包括:
	- HTML元素选取
	- HTML元素操作
	- HTML事件函数
	- CSS操作
	- JavaScript特效和动画
	- HTML DOM遍历和修改
	- AJAX
	- 工具类

> jQuery的流行还是基于`HTML-first`的理念，现在的理念是`javascript-first`，主流的框架例如`React`都是以js为主构建页面，另外随着JS语言的进步，很多jQuery的封装已经有built-in的API了。总的来说jQuery在现在用不触大

### 引入

- 从CDN引入`<script src="https://code.jquery.com/jquery-3.2.1.min.js"></script>`
- JQuery Preview:

```javascript
//when a user clicks the button with id 'trigger'
$('#trigger').click(function(){

	//change the body's background to yellow
	$('body').css("background","yellow");
	
	//fade out all the img's over 3 seconds
	$('img').fadeout(3000, function(){
		
		//remove image from page
		$(this).remove();
	});
});
```
	
### 用法

- 表示DOM节点:
	- 使用`$`符号，jQuery可以将`HTML`标签转换成一个对象，这个对象有`hide`,`show`之类的方法，用来控制标签的行为
		- `$("body")`表示`<body></body>`对应的object
		- `$(".sale")` 所有class为sale的标签
		- `$("#bonus")` id为bouns的标签
		- `$("li a")`所有li下的a标签

- 修改节点属性
	- 使用`.css()`方法
		- 直接修改节点样式：
		
		```javscript
		$("div").css("background-color","purple");
		$("div.highlight").css("width","200px");
		$("#third").css("border","1px solid orange");
		$("div:first-of-type").css("color","pink");
		``` 
		- 使用样式表：
		
		```javascript
		var styles = {
			backgroundColor: "pink",
			fontWeight: "bold"
		};
		$("#special").css(styles);
		```

- 其它方法
	- `val()`：获取、修改`<input>`中的文字内容
		- `$("input").val()`
	- `text()`：获取、修改当前标签的文字（不包含HTML标签）
		- `$('li').text()` 
	- `attr()`：获取、修改标签的属性
		- `$('img').attr("src","https://xxx.jpg");` 
		- `$('img').last().attr("src","https://xxx.jpg");` 
		- `$('img').first().attr("src","https://xxx.jpg");` 
	- `html()`：获取、修改当前标签的HTML文本
		- `$('li').html()`
	- `Class`：
		- `$("h1").addClass("correct")`
		- `$("h1").removeClass("correct")`
		- `$("h1").toggleClass("correct")`  

### Event
	
- `click()`：点击响应

```javascript
$('#submit').click(function(){
	console.log("Another click");
});
$('button').click(function(){
	$(this).css("background", "pink");
});
```
	
- `keypress()`：按键响应

```javascript
$('input[type="text"]').keypress(function(){
	console.log($(this).val());
});
```

- `on()`：添加事件的通用API

```javascript
$('#submit').on('click', function(){
	console.log("click");
});
```

```javascript
$('button').on('dblclick', function(){
	console.log("double click");
});
```

```javascript
$('#ttt').on('dragstart', function(){
	console.log("Drag Started");
});
```

```javascript
$('h1').on('mouseenter', function(){
	console.log("Mouse Enter");
});
```

```javascript
$('h1').on('mouseleave', function(){
	console.log("Mouse Leave");
});
```

- Effect
	- `fadeOut` / `fadeIn`/`fadeToggle`
	
	```javascript
	$('#clickme').click(function(){
		$("#book").fadeOut("slow", function(){
			//animation complete
		});
	});
	``` 
	
	- `slideDown` / `slideUp` / `slideToggle`

	```javascript
	$('#clickme').click(function(){
		$("#book").slideDown(1000, function(){
			//animation complete
			$(this).remove
		});
	});
	``` 