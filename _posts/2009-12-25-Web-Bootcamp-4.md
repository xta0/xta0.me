---
layout: post
title: Tools for Web Work Flow
list_title: Web Dev Bootcamp Part 4 | Dev Tools
---

> Update @ 2016/07

### 及时预览

- live server

```
npm install -g live-server
```

### Compile Sass

- 配置Saas开发环境
	- `npm install node-sass --save-dev`	
	- package.json:

	```json
	"scripts": {
		"compile:sass": "node-sass sass/main.scss css/style.css"
	},
	```
	- `npm run compile:sass`

### Gulp

Gulp是一个JS语言的Task Runner，功能类似Ruby的Rake。可以通过自定义task来实现一些辅助功能，比如

1. compile Sass
2. Watch dog


- install

```
//全局环境
npm install -g gulp-cli

//项目环境
npm install gulp --save-dev
```
- gulpfile

Gulp自身是一个空的容器，需要在gulpfile中配置任务，gulpfile是js文件，支持js语法


### 浏览器适配

- browserSync
    - 多浏览器互动
    - 支持mobile预览





