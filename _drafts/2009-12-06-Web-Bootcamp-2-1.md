---
layout: post
title: SASS
list_title: Web Dev Bootcamp Part 2 - 1 | SASS 
categories: [Web]
---

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

### Variable

- SCSS允许在CSS文件中定义变量后使用：

```css
$color-primary: #f9ed69; //yellow

nav{
  background-color: $color-primary;
 }
```

- 支持嵌套结构

```css
.navigation{
  list-style: none;
  float: left;
  li {
    display: inline-block;  
    margin-left: 30px;
    /* & 代表li */
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