---
list_title: Web Bootcamp Part 1
layout: post
categories: [web,html,css,javascript]
meta: Web Development Bootcamp. Basic HTML/CSS usage
---

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
