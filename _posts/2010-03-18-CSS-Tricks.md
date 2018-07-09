---
layout: post
list_title: CSS Tricks
---


## Best Practice

- 使用Emmet

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

