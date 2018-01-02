---
title: PL-Ruby-2
layout: post
tag: Ruby Tool Stack
categories: PL
tags: Ruby
---

> 所有文章均为作者原创，转载请注明出处

## Ruby中的正则表达式

### 工具

- [negexr.com](negexr.com)


### 创建正则表达式

- 通过`//`标识
- 通过`%r{regexp}`
- 通过`Regexp.new("regexp")`

```ruby

2.2.1 :006 > a = /hello/
 => /hello/ 
2.2.1 :007 > a.class
 => Regexp 
 
2.2.1 :008 > a = %r{hello}
 => /hello/ 
2.2.1 :009 > Regexp.new("hello")
 => /hello/ 

```

### 匹配字符

- 使用`match`或`=~`

```ruby

2.2.1 :004 > /hello/=~'hello world'
 => 0 
2.2.1 :005 > /o/.match 'hello world'
 => #<MatchData "o"> 
 
```
使用`=~`
使用`match`方法会返回`MatchData`

- 使用修饰符
	- `/hello/i`:忽略大小写
	- `/hello/m`:匹配多行
	- `/hello/x`:多行编辑+注释
	- `u,e,s,n`:控制编码
 

## Debuging Ruby Code

### 使用Ruby自带库"debug"来调试Ruby代码

- ruby debugger是ruby内置的debug工具
- `ruby -rdebug xxx.rb`
- 在需要debug的代码前面插入`require debug`

### 使用byebug来调试Ruby代码

- byebug是开源的ruby debugger，调试ruby 2.0以上的代码
- byebug可以作为gem包安装
- 功能上是ruby debug的超集
- [Github](https://github.com/deivid-rodriguez/byebug)

### 使用pry来内省

- pry是`irb`的代替
- pry比`irb`有更多功能
- 在需要debug的地方加入`binding.pry`,拥有更好的REPL的调试方式

```ruby

debug = Debug.new()
binding.pry
debug.func(100)

```


### 使用Ruby自带的benchmark来检查代码执行

- `benchmark`:

```ruby


# test benchmark
arr = (1..10**8).to_a

Benchmark.bm(7) do | b |

	b.report("each"){ arr.each do |i|; i; end }
	b.report("for"){ for i in arr; i; end }
	b.report("upto"){ 0.upto(arr.length()-1) do |i|; i; end }
	
end

```

- `measure`方法用来获取某个方法的执行时间

## 制作Ruby命令行工具

### 使用Thor

在制作VZScaffold的时候，对于命令行工具的制作，经历了各种尝试, 包括使用optparser，使用Slop

## Make Ruby Gems


## 使用Grape设计API

### Grape简介

- Grape，为API设计而生的DSL
	- 描述性的API构建方式，代码可读性强
	- 提供了构建Restful API的一套工具，如参数约束，路径约束，版本管理等
	- 提供了JSON/XML/TXT格式的渲染工具 
