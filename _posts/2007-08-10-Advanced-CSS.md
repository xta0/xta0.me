---

layout: post
title: Advanced CSS
---

# Advanced CSS

### Best Practice

- Responsive Design
	- Fluid layouts
	- Media Queries
	- Responsive images
	- Correct units
	- Desktop-first vs mobile-first

- Maintainable and Scalable Code
	- Clean
	- Easy-to-understand
	- Growth
	- Reusable
	- How to organize files
	- How to name Classes
	- How to structure HTML

- Web Performance
	- Less HTTP requests
	- Less code
	- Compress code 
	- Use a CSS preprocessor
	- Less images
	- Compress images

- BEM naming convention
	- Block: 
		- standalone component that is **meaningful** on its own 
		- `.block {}`
	- Element:
		- part of a block that has **no standalone meaning**
		- `.block__element {}` 
	- Modifier:
		- a different version of a block or an element. 元素的某种状态
		- `.block--modifier {}` 
		- `.block__element--modifier {}`
- 7-1 Pattern
	- `base/`
	- `components/`
	- `layout/`
	- `pages/`
	- `themes/` 
	- `abstracts/`
	- `vendors/`

## Behind Scenes

### Browser work flow chart

![](/assets/images/2007/08/work-flow.png)

### **CASCADE** 

解决某个标签应用多个CSS属性冲突问题 

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

### Specificity work flow 
	
![](/assets/images/2007/08/Specificity.png)
	
### Value Processing work flow 

![](/assets/images/2007/08/value-processing.png)

### Visual Formatting Model

- Definition

Algothrithm that calculates boxes and determines the layout of these boxes, for each element in the render tree, in order to determine the final layout of the page. 浏览器的layout算法

- 涉及到
	- **Dimensions of boxes**: the box model
	- **Box type**: `inline`, `block` and `inline-block` 
	- **Positioning scheme**: `floats` and `positioning`
	- **Stacking contexts**
	- Other elements in the render tree.
	- Viewport size, dimensions of images, etc.


- Box model
	- **total width** = right border + right padding + specified width + left padding + left border
	- **tottal height** = top border + top padding + specified height + bottom padding + bottom border
	
	![](/assets/images/2007/08/box-model.png)

- Box-Sizing
	- border-box
		- **total width** = specified width 
		- **tottal height** = specified height
- Box Type
	- Inline
		- `display: inline` 
		- Content is distributed in lines 
		- **Occupies only content's space**
		- **No line-breaks**
		- No heights and widths
		- Paddings and margins only horizontal
	- Block-Level  
		- `display: block`
		- **100% parent's width** 
		- vertically, one after another
		- **Box-model applies**
	- Inline-Block
		- A mix of block and inline
		- **Occupies only content's space**
		- **No line-breaks**
		- **Box-model applies**

- Positions
	- Normal flow 
		- Default positioning scheme `positon: relative` 
		- **NOT** floated
		- **NOT** absolutely positioned
		- Elements laid out according to their source order
	- Floats
		- Element is removed from the normal flow
		- Text and inline elements will wrap around the floated element
		- The container will not adjust its height to element
		- `float:left`,`float:right` //左上角对齐，右上角对齐
		- [All About Floats](https://css-tricks.com/all-about-floats/)
	- Absolute Positioning 
		-  Element is removed from the normal flow
		-  No impact on surrounding content or elements
		-  Use `top`,`bottom`,`left`,`right` to offset the element from its relative positioned container
		-  `position: absolute`,`postion: fixed`

- Stacking Contexts

![](/assets/images/2007/08/stacking-context.png)

### 单位换算

- CSS每种属性都有一个inital value
- 浏览器会指定一个默认的**root font size** （通常是16px）
- 百分比和相对单位会被转化成像素单位
- 百分比
	- 如果修饰`font-size`，是相对于父类`font-size`的值
	- 如果修饰`length`上，是相对于父类`width`的值
- `em`
	- 如果修饰`font-size`，是相对于父类`font-size`的值
	- 如果修饰`length`，是相对于当前的`font-size`的值	- `rem`
	- 相对于`document's root`的`font-size` 
- `vh`和`vw`
	- 相对于viewport's `height`和`width` 

![](/assets/images/2007/08/unit.png)

### 继承关系

- Inheritance passes the values for some specific properties from parents to children
- 关于文字的属性可以被集成: `font-family`,`font-size`,`color`,etc.
- 需要计算的属性可以被集成，声明的固定的值不被继承
- 使用`inherit`关键字可以强制继承某属性
- 使用`initial`关键字可以强制重置某属性

![](/assets/images/2007/08/inheritance.png)


## Sass

> Sass is a CSS **preprocessor**, an extension of CSS that adds power and elegance to the basic language

### Features

- **Variables**: for reusable values such as colors, font-sizes, spacing, etc;
- **Nesting**: to nest selectors inside of one another, allowing us to write less code.
- **Operators**: for mathematical operations right inside of CSS;
- **Partials and imports**: to write CSS in different files and importing them all into one single file;
- **Mixins**: to write reusable pieces of CSS code;
- **Functions**: Similar to mixins, with the difference that they produce a value that can than be used;
- **Extends**: to make different selectors inherit declarations that are common to all of them;
- **Control directives**

### Sass and SCSS

![](/assets/images/2007/08/sass-vs-scss-syntax.png)

### Variable

- SCSS允许在CSS文件中定义变量后使用：

```css
$color-primary: #f9ed69; //yellow

nav{
  background-color: $color-primary;
 }
```

- 使用嵌套结构, Pseudo Class状态前用`&`代替

```css

.navigation{
  list-style: none;
  float: left;
  
  li {
    
    display: inline-block;  
    margin-left: 30px;
    
    &:first-child{
      margin: 0;
    }
     
    a{
      text-decoration: none;
      text-transform: uppercase;
    }
  }
}

```

### Mixin

提取公共样式组件,可传参

```css
@mixin style-link-text($color){
  text-declaration: none;
  text-transform: uppercase;
  color: $color
}
```

### Function

- 使用内置function

```css
&:hover{
background-color: lighten($color-tertiary,15%);
}
```

- 使用自定义function

```css
@function divide($a, $b){
  @return $a/$b;
}

```


### NPM

- 配置Saas开发环境
	- `npm install node-sass --save-dev`	
	- package.json:

	```json
	"scripts": {
		"compile:sass": "node-sass sass/main.scss css/style.css"
	},
	```
	- `npm install -g liver-server`


## Best Practice

- 将`px`转换为`rem`。
	- 改变`root font-size`其它属性都会自动变化
	- 将`root font-size`设为`62.5%`
	- IE9以下不支持`rem`
- `box-sizing`

```css

* {
    box-sizing: inherit;
  }
  
body {
	box-sizing: border-box;
}

``` 


### Seudo Class

`seudo class`用来指定某个元素的某种状态

```css
.btn:focus{
	//focus状态下应用某种样式
}
.btn:hover{
	//hover状态下应用某种样式
}

```

### Animations

- 使用`transform`

和`transition-duration`一起使用，`transition-duration`描述时间

```css
.btn:hover::after{
    transform: scaleX(1.4) scaleY(1.6);   
    opacity: 0;
    transition: .4s;
}
```

- 使用`key-frame`

`key-frame`帧动画和`transform`一起用

```css

.btn {
    animation: moveFromBottom .5s ease-in-out;
}

@keyframes moveFromBottom {
    0% {
        opacity: 0;	
        transform: translateY(30px); 
    }
    
    100% {
        opacity: 1;
        transform: translateY(0);
    }
}



```


### Tricks

- 样式表先* 初始化`*`和`body`

```css
* {
  margin: 0;
  padding: 0;
  box-sizing: border-box;
}

body {
  font-family: "Lato", sans-serif;
  font-weight: 400;
  font-size: 16px;
  line-height: 1.7;
  color: #777;
  padding: 30px;
}
```

- 使用`backgroud-image`
	- 增加渐变：
  	
  	```
  	cssbackground-image: linear-gradient(
      to right bottom,
      rgba(126, 213, 111, 0.8),
      rgba(40, 180, 131, 0.8)
    ), url(../img/hero.jpg);
    ```
	- 增加`clip-path`

  ```css
  clip-path: polygon(
    0 0,
    100% 0,
    100% 50%,
    0 100%
  ); //(X,Y) => 左上角，右上角，右下角，左下角
  ```

- 对于`inline`的元素，外层套`div`

## FlexBox Properties

* Container Properties

  * `flex-direction`
  * `justify-content`
  * `flex-wrap`
  * `aligin-items`
  * `align-content`

* Item Properties
  * `order`
  * `flex`
  * `flex-grow`
  * `flex-shrink`
  * `align-self`

## 资料

- [FlexBox Guide](https://css-tricks.com/snippets/css/a-guide-to-flexbox/)
- [Style Guide](http://codeguide.co)
- [BEM](http://getbem.com/naming/)
- [7-1 Pattern](https://gist.github.com/rveitch/84cea9650092119527bc) 
- [All About Floats](https://css-tricks.com/all-about-floats/)