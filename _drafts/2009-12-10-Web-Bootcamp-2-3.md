---
layout: post
title: Bootstrap
list_title: Web Dev Bootcamp Part 2 - 3 | CSS Best Practice
---

### Best Practice Rules

- Responsive Design
	- Fluid layouts
	- Media Queries
	- Responsive images
	- Correct units
	- Mobile-first
		- Font
			- `@media`
		- Responsive Image
			- `picture`
		
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

### Organize CSS Files

- **7-1 Pattern**
	- `base/`
	- `components/`
	- `layout/`
	- `pages/`
	- `themes/` 
	- `abstracts/`
	- `vendors/`

### BEM Rule

- **Block**
	- 组件
	- standalone component that is meaningful on its own
	- `.block {}`
- **Element**
	- 组件中的元素
	- part of a block that has no standalone meaning
	- `.block__element {}`
- **Modifier**
	- 不同的组件版本
	- a different version of a block or an element.
	- `.block--modifier {}`


## Tips & Tricks

### Center Element

- 水平居中
	- block element
		- `text-align: center`
			- 用于文本相关标签
			- `p,h1,...`
		- `margin-left:auto; margin-right:auto`
			- 用在container上，比如`div`套一个`img`
			- parent和child需要指定宽度或者有默认宽度

```html
<body>
    <div class="text-wrap">
        <div class="center">
            <h1>Center Heading</h1>
            <p>All my text in this section would be aligned left as default</p>
        </div>
    </div>
</body>
```
```css
.text-wrap {
  width: 100%;
}

.center {
  width: 50%;
  margin-left: auto;
  margin-right: auto;
}

h1 {
  text-align: center;
}
```