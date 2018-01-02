---
title: Ruby Programming Language
layout: post
tag: Ruby
categories: [pl,cheatsheet]
---

<em>所有文章均为作者原创，转载请注明出处</em>

> ###Ruby 2.0 简明语法整理


##目录

- [变量/常量](#var)
- [类型转换](#type)
- [表达式](#expression)
- [字符串](#string)
- [控制流](#control-flow)
- [符号](#symbol)
- [数组](#array)
- [Hash](#hash)
- [迭代器](#itor)
- [Block](#block)
- [Proc](#proc)
- [Lambda](#lambda) 
- [类](#class)
- [命名空间](#namespace)
- [Module](#module)
- [Include](#include)
- [异常](#exception)
- [文件操作](#file)
- [ENV](#env)
- [Kernel](#kernel)
- [Object](#object)
- [Date](#date)
- [YAML](#yaml)
- [ERB](#erb)




<h3 id="var">变量/常量</h3>

- 全局变量$标识

```ruby

def basic_method
  puts $x
end

$x = 10
basic_method()

```

- 全局常量大写开头

```ruby

CONST_V = "abc"

```


<h3 id="type"> 类型 </h3>

- 整型，浮点型，字符串

```ruby
a = 10;
print a.to_f #10.0

a = 3.3
print a.to_i

a = 100
print a.to_s

```

- 浮点型

```ruby

d = 7.3-7.2
puts d 
=> 0.09999999999999964

require "bigdecimal"

a = BigDecimal.new("7.3")
b = BigDecimal.new("7.2")
puts a-b 

=> 0.1E0

```

<h3 id="expression"> 表达式 </h3>

- !

```ruby

a,b = 10,20
c = 0
if a==b
  c = 1

elsif a != b
  c = -1 
end

```

- eql? :检测类型和值

```ruby

c = 10
r = a.eql?(c)
puts r #true
d = 10.0
r = a.eql?(d)
puts r #false


```


- <=>：

```
x<=>y
```
x>y 则返回1 ， x<y则返回-1 ， x=y则返回0


<h3 id="string"> 字符串 </h3>

- 拼接:

```ruby

 #使用+号
x = "a"
y = "b"
z = x+y

 #使用<<号
c = "a"<<" "<<"b"

```

- inspect：查看字符串

```ruby

string = "  Hello world"
d = string.inspect
puts d
=> "\"  Hello world\""

```

- chop:砍掉末尾字符

```ruby

string = "  Hello world"
e = string.chop
puts e
=> "  Hello worl"

```

- include? :包含substring

```ruby

b = string.include?("Hello")
puts b

```

- 转义符号:

\n, \t：tab , \b : 删除 , \v 换行+tab


```ruby

 ##字符串中的引号，双引号可被#{}解析
puts "Hello \" world \" " 
=> Hello " world " 

 ##使用单引号，不会被#{}解析
puts 'Hello "world"'
=> Hello "world" 

 ##字符串中的\
puts "Hello \\ world"
=>Hello \ world

```


- 使用标记来约定字符串的开始和结束：<<ID xxxxx ID

```ruby

x = <<JAYSON_
this is some
thing
else,haha
JAYSON_

puts x

```

- 重复字符串

```ruby

x = "b"*5
puts x

```

- 格式化：#{}

```ruby

x = 10
y = 20
puts "#{x} + #{y} = #{x+y}"

```

- 格式化：使用sprintf

```ruby

r = rand(100) #0-100内的随机数
str = sprintf("%.1f",r)
puts str.inspect

```


- %q{}不会被#{}解析

```ruby

x = %q{ this is some
thing 
else,haha}

=> " this is some\nthing \nelse,haha"

```

- %{} 会被#{}解析

```ruby

name = %{Jason}

```

- %w{}将字符串转换为字符串数组

```ruby

strlist2 = %w{dog monkey pigg cock} ##将元素转换为字符串数组
puts strlist2.inspect

```


- 替换substring

```ruby
x = "foobar"

 #sub方法只将第一个bar变成boo
y = x.sub('bar','boo')
puts y

x = "foobarxoobar,ioo="

 #gsub方法将所有bar都变成boo
y = x.gsub('bar','boo')
puts y  
 
```
- 正则表达式以/xxx/表示

```ruby

 #将前两个字符替换成Hello
y = x.sub(/^../,'Hello')
puts y

 #将后两个字符替换为Hello
y = x.sub(/..$/,'Hello')
puts y

 #x.each{|i| puts i} #--wrong!
 #需要使用scan
x.scan(/./){|i| puts i} #--Right!

 #scan的参数为正则表达式
x = "this is something I will never do"
x.scan(/\w\w/){|i| puts i}

 #\w的意思是匹配数字,字母下,划线，对每个字符进行匹配 
 # \w\w 是对两个字符进行匹配
x.scan(/\w+/){|i| puts i}

 ###.scan方法实际上返回的时数组
p = x.scan(/\w+/)
puts "-----"
puts p.class  #array
puts "-----"
puts p
puts "-----"
 #\w+ ：+的意思是一直向后匹配，直到遇见非数字，字母，下划线

x = "I spent 100 dollars on this machine which is now 200 dollars"
x.scan(/\d+/){|i| puts i}
 #匹配出数字

 #匹配字符集[....]匹配[]中的字符集
x = "this is a test"
x.scan(/[aeiou]/) {|i| puts i}
x.scan(/[a-m]/){|i| puts i}

```

- 访问char

```ruby

x = "abc"
puts x[0]

```

- char的index

```ruby

string.index("a") ##1

```

- 字符串匹配

```ruby

 #（1）根据正则匹配：
 #检测字符串中是否包含某字符: 使用=~
 #检测字符串中不包含某字符: 使用!~
puts "get string" if "jayson loves basketball" =~ /[basketball]/
puts "String contains no digits" unless "jayson loves basketball" =~ /[0-9]/


 #（2）使用match方法
puts "String has matched" if "jayson loves basketball".match("loves")

```

- 其它

```ruby

 #reverse
string = string.reverse()
puts string

 #upercase
string = string.upcase()
puts string

 #lowercase
string = string.downcase()
puts string

 #swap
string = string.swapcase()
puts string

 #length
l = string.length
puts l

 #size == length
s = string.size
puts s

 #split
string = "hello world"
list = string.split(" ")
puts list.inspect

 #concat
string.concat("another string")
puts string

```


<h3 id="control-flow">控制流</h3>

- if

```ruby

 ##单行判断
puts "a>10" if a>10

 #if-elsif-else
if a<10
  puts "a<10"
elsif
  puts "a=10"
else
  puts "a>10"
end

```

- unless

```ruby
unless a>10 
  puts "a>10"
end

```

- 三元操作

```ruby
type = a > 10 ? "1" : "2"
puts type

```

- case-when

```ruby

fruit = "orange"

case fruit
when "orange"
  color = "orange"
when "apple"
  color = "red"
when "banana"
  color = "yellow"
else
  color = "unknown"
end

```

- while

```ruby

x = 10

while(x>=1)
  x = x/2
  puts x
end

 #单行
x=10
x = x/2 while x >= 1 
puts x

```

- until

```ruby


x = 10

until x <= 1
  x = x/2
  puts x
end

 #单行
x = 10
x = x/2 until x<1 
puts x

```

- loops

```ruby

array = [1,2,3,4]
for i in array do
  puts "i is #{i}"
end

strlist1 = ["a","b","c"]
puts strlist1.inspect

strlist2 = %w{dog monkey pigg cock} ##将元素转换为字符串数组
puts strlist2.inspect

for animal in strlist2 do
  next if animal == "monkey" ##continue
  puts "#{animal}"
end


```


<h3 id="symbol">Symbol</h3>

symbol是无值常量，全局唯一，Ruby特性

```ruby

hello = :key
puts "equal" if hello == :key #equal
a = {:key => "hh"}
puts a[:key] ## hh

 #symbol:在内存中只创建一次
treehouse = {'name'=>'Treehouse','location' => 'Treehourse Island'}
 
 #如果再创建一个treehouse，那么所有的key都要重新创建一遍
 #如果使用symbol，key不会重新创建
treehouse = {:name => 'Treehouse', :location => 'Treehouse Island'}

```

<h3 id="array">Array</h3>

- 数组个数

```ruby

a = p.count

```

- 清空

```ruby

a.clear

```


- 数组索引

```ruby

puts x.first
puts x.last

puts x[0]
puts x[-1] ##最后一个

puts x.first(2) ##前两个
puts x.at(0)


```
- 数组中每个元素的类型可以不同

```ruby
x = [1,"2",3.0]
```
- 反转

```ruby

x.reverse()

```

- 数组遍历

```ruby

 #使用do-each
q.each do |i|
  puts i.to_s
end

 #使用block
q.each { |i| 
  puts i.to_s + "x"
}

 #使用collect
[1,2,3,4].collect{|element| element*2}

```

- 数组包含某个元素

```ruby

x = [1,2,3]
puts x.include?("x")
puts x.include?(3)

```


- 数组的push操作

```ruby
x << "wrod"
x.push("word")

x += [1,2]

```
- 数组的pop操作

```ruby
x.pop

```

- 数组的join操作:

```ruby

 #如果一个数组全是字符串，可以使用.join方法，将数组中的元素连接起来

x = ["jj","ss","gg"]
puts x.join()

=> jjssgg

puts x.join(', ')

=> jj, ss, gg

```

- 字符串按符号分割:

```ruby

x = "short sentence; Another; no more"
q = x.split(';')
puts q.class 
=> array

puts q
=> short sentence Another no more
```

使用inspect方法使输出格式更易读，所有对象都有inspect方法:

```ruby

x = "short sentence; Another; no more"
p = x.split(';').inspect 

puts p  
=>["short sentence", " Another", " no more"]

```

- 数组相加:

```ruby
p = [1,2,3]
q = ["1",2,"33"]  
z = p+q
print z

=> [1, 2, 3, "1", 2, "33"]

```

- 数组相减:

```ruby

 #两个数组相减
p = [1,2,3]
q = [4,5,6]
z = p-q
print z 

=> 1,2,3

```

- slice: 取数组的某些值，组成一个新数组, 参数为index值，不修改原array

```ruby

list1 = [1,2,3]
ele = list1.slice(1); puts ele.inspect #[2]
slice = list1.slice(1,2) ; puts slice.inspect #[2,3]

```

- slice! :取数组的某些值，组成一个新数组, 参数为index值，修改原array

```ruby

slice = list1.slice!(1,2) ; puts list1.inspect #[1]

```

- unshift :在头部插入元素

```ruby

a = [1,3,4]
a.unshift 2

=>  [2, 1, 3, 4]

```

- rindex :返回数组元素的index

```ruby
	
a = [1,2,3,"a","s"]
a.rindex "a"

=> 3

```

- 多维数组

```ruby

array = [1,2,3]
array = array.push([2,3,4])
puts array.inspect 
=> [1, 2, 3, [2, 3, 4]]

 #展成1维数组
array = array.flatten
=> [1, 2, 3, 2, 3, 4]

```

- sort & compare

```ruby

array = [5,1,3,5,4,9]
array1 = array.sort ; puts array1.inspect()
array1 = array.sort{|a,b| a<=>b} ; puts array1.inspect()
array1 = array.sort{|b,a| a<=>b}.reverse ; puts array1.inspect()
array1 = array.sort{|b,a| a<=>b}.reverse.uniq ; puts array1.inspect() ##uniq去重

```


- any? all?

```ruby

array = [5,1,3,5,4,9]
ret = array.any?{|e| e>3} ; puts ret.inspect()
ret = array.all?{|e| e>3} ; puts ret.inspect()

```

- select :选出符合条件的元素

```ruby
newArray = array.select{ |e| e>3}; puts newArray.inspect()
```

- reject :去除符合条件的元素

```ruby
newArray = array.reject{ |e| e>3}; puts newArray.inspect()
```

- map：

```ruby
newArray = array.map{|e| e*2 }; puts newArray.inspect()
```

<h3 id="hash">Hash</h3>

- 创建

```ruby

h = Hash.new() ; puts h.inspect()
h = {"hello" => "world"}
h = Hash.new{|hash,key| hash[key] = "Default value for #{key}"}

```

- 清空

```ruby

h.clear;puts h.empty?

```


- 个数

```ruby

map.size

```

- 访问

```ruby

map['cat'] = 'boss'

```

- 遍历

```ruby

map.each{ |k,v|
	
	puts "#{k} => #{v}"
}


h.each_pair {|k,v| 
	
	puts "the key at #{k} is #{v}"
	
}

h.each_key{|k| 
	puts "the key is #{k}"
	
}

h.each_value{|v| 
	puts "the value is #{v}"

}

h.select{|k,v| k == "501"};puts h.inspect

h.keep_if{|k,v| k == "502"}; puts h.inspect


```

- 删除

```ruby

map.delete('cat')
map.delete(0)
map.delete_if {|key,value| key =="some key"}

```

- 所有keys/values

```ruby

map.keys
map.values

```

- key/value存在?

```ruby

puts h.has_key?("candy")
puts h.has_value?("candy")

```

- to array

```ruby
h = {"501" => "cctv1", "502" => "cctv2", "503" => "cctv3"}
puts h.to_a.inspect

=>[["501", "cctv1"], ["502", "cctv2"], ["503", "cctv3"]]

```

- find

```ruby

h = {"501" => "cctv1", "502" => "cctv2", "503" => "cctv3"}
puts h.find{|k,v| k == "503"}.inspect;  ##["503","cctv3"]返回数组
puts h.find_all{|k,v| v.match("cc")}.inspect 
puts h.all?{|k,v| v.match("cctv1")}
puts h.any?{|k,v| v.match("cctv1")}

```

- map

```
puts h.map{|k,v| v += "ss"}.inspect #将value的值映射为#["cctv1ss","cctv2ss","cctv3ss"]

```

- reject

```
h.reject{|k,v| k == "cctv1"}.inspect

```

<h3 id="itor">迭代器</h3>

```ruby

 #例如upto：从1到5
1.upto(5) {puts "upto"}

 #downto:从10递减到5
10.downto(5) {puts "downto"}

 #从0开始，到50，每次步进5
0.step(50,5) do
  puts "step"
end

 #如果要打印出循环的index，怎么办？

 #使用block
1.upto(5) {|i| puts i}

 #使用do-end
10.downto(5) do |i|
  puts i
end

```

<h3 id="block">Block</h3>

- block:是匿名函数，可以当参数传递,当参数时需要用&符号声明参数是block类型
 
```ruby

def say_hello(&block)
  puts "Hello world"
  if block_given?
    block.call
  else
    puts "no block given"
  end
end

say_hello {puts "call from block"}

say_hello do
  puts "call from block"
end

 #block的返回值
def say_name(&block)
  name = block.call
  puts name
end

say_name{"Jayson"}


```

- yield 

```ruby

def test_yield(&block)
  for i in 1..5 do
    yield i ##will call block, pass i‘s copy to the block
  end
end

test_yield {|i| puts "#{i*2}"} #2,4,6,8,10

```

class SimpleBenchmarker
  
  def self.go(times, &block)
    
    puts "----------Benchmarking started------------"
    start_time = Time.now
    puts "start time : #{start_time}\n\n"
    times.times do |t|
      print "."
      block.call
    end
    print "\n\n"
    end_time = Time.now
    puts "End Time:\t #{end_time}\n"
    puts "----------Benchmarking finished-----------"
    puts "Result :\t\t #{end_time - start_time} seconds"
    
  end
end

SimpleBenchmarker.go 5  do
	time = rand(0.1..1.0)  
	sleep time
end



<h3 id="proc">Proc</h3>

- proc :用来创建匿名函数 - block

```ruby

proc_b = Proc.new {puts "hello b"}
proc_a = proc {puts "hello a"}
proc_b.call
proc_a.call

```


<h3 id="lambda">Lambda</h3>

- lambda: 一种proc

```ruby

my_lambda_1 = lambda {}
my_lambda_2 = -> lambda {}

```

proc和lambda的区别是，lambda必须要传入参数

```ruby

my_proc = proc {|name| puts "hello #{name}"}
my_proc.call ("jayson")

my_lambda = lambda {|name| puts "hello #{name}"}
my_lambda.call ("candy")

def return_from_proc

  p = proc {return "return from proc"}
  p.call
  return "return from function"
  
end

def return_from_lambda

  l = lambda {return "return from lambda"}
  l.call
  return "return from function"
end

puts return_from_proc()
puts return_from_lambda()


```


<h3 id="class">Class</h3>

- 成员变量用@定义

```ruby

class Square
  
  ##默认构造
  def initialize(side_length)
    
    ##定义成员变量用@
    @side_length = side_length
  
  end
  
  def area
  
    priFunc() ##调用private方法
    self.pubFunc() ##调用public方法
    @side_length * @side_length
    
  end
  
private
  def priFunc()
    puts "private func called"
  end

public
  def pubFunc()
    puts "public func called"
  end

end


```

- 继承,类方法,类成员

```ruby

class Test < Object ##继承：默认都是从object继承

  ##定义类成员的getter方法
  def self.clz_counter
    @@clz_counter ##计数器
  end
  
  def initialize
    
    ##类成员
    if defined? (@@clz_counter)
      @@clz_counter += 1
    else
      @@clz_counter = 1
    end
  end
  
  ##类方法
  def self.test_method1
    puts "test self_method1"
  end
  
  def Test.test_method2
    puts "test self_method2"
  end

end

```

- override

```ruby

class ParentClz
  
  #父类的pFunc方法返回a
  def pFunc (a)
     a
  end
  
end

class ChildClz <ParentClz
  
  ##重写p方法，返回ax
  def pFunc (a)
     a.to_s + "x"
  end
  
end

```

- 修改或增加现有类的方法

```ruby

class ChildClz
  
  def pFunc(a)
    a.to_s+"x"+"x"
  end
  
  def qFunc
    puts "this method is added"
  end
  
end


```

- attr

```ruby

class AttrClz
  
  ##自动支持.预发的getter和setter
  attr_accessor :name, :age
  
  ##私有方法
  private
  def private_method
  end
  
  ##共有方法
  public
  def public_method
  end
  
  def age
    @age
  end
  
  ##protect方法
  protected :age
  
end

```

- 内部类

```ruby

class InsideClz
  
  def InsideClz.giveMeDrawingClz
    Drawing.new
  end
  
  ##内部类
  class Drawing
    def print
      puts "this is a print method inside Drawing"
    end
  end
  
end

a = InsideClz.new
InsideClz.giveMeDrawingClz.print

 ##返回Drawing 对象
a = InsideClz::Drawing.new
a.print


```

<h3 id="namespace">命名空间</h3>

```ruby

module NameSpace1

  def NameSpace1.random
    rand(100)
  end

end

module NameSpace2
  
  def NameSpace2.random
    rand (10)
  end
end

puts NameSpace1.random
puts NameSpace2.random

```


<h3 id="module">Module</h3>


```ruby

module ToolBox
  
  class Ruler
    attr_accessor :length
  end
end

module Country
    
  class Ruler
    attr_accessor :name
  end
end

a = ToolBox::Ruler.new
a.length = 50

b = Country::Ruler.new
b.name = "kate"

puts a,b

```

<h3 id="Include">include</h3>

在某个作用域中引入其它namespace中的方法或者类

```ruby
module UsefulFeatures
  
  def class_name
    
    self.class.to_s
    
  end
  
end


class SomeClass
  
  ##包含了其它类
  include UsefulFeatures
  
end

x = SomeClass.new
puts x.class_name


```


系统对array默认include的两个类：
Enumerable

- each方法：

```ruby

[1,2,3].each{|i| puts i}

```

除了each方法外，还提供了20多个有用的放法:
collect,detect,find,find_all,include?,max,min,select,sort,to_a

```ruby
list = %w{this is the longest word check}
puts list.inspect

```

- collect

```ruby
[1,2,3].collect{|i| puts i.to_s+"x"}
```
- detect

```ruby
a = [1,2,3].detect{|i| i.between?(2,3)}
puts a #2
```

- select

```ruby
a = [1,2,3,4,5].select{|i| i.between?(2,3)}
puts a

```

- sort

```ruby
[4,3,2,1].sort #[1,2,3,4]
```

- min/max

```ruby
[1,2,3].max 
[1,2,3].min
```

<h3 id="exception">异常</h3>

- begin-rescue

```ruby

begin 
  puts 10/0
rescue
  puts "crash"
end

```

- catch-throw

```ruby

catch(:finish) do
  1000.times do
    x = rand(1000)
    throw  :finish if x == 123
  end
  
  puts "the random number is 123"
end

```

- rescue-ensure

```ruby

 #ensure : 类似finally，保证会执行
 #rescue : 

def header(&block)
  puts "this is our header"
  block.call
  puts "this is our footer"
  
rescue
  puts "this is where rescue an error"  
ensure
  puts "this one is ensure to be called"
end

header {puts "this block will crash"; raise "this is an error"}

```


<h3 id="file">文件操作</h3>

- 创建文件

```ruby

 ##如果用new打开文件，需要调用close
a = File.new("./input.json","r")
puts a.class ## File

 ##默认以换行符进行分割
a.each{|line| puts line}
a.close


a = File.new("./input.json","r")
 ##如果用其他符号分割:
a.each(':'){|line| puts line}
a.close


a = File.new("./input.json","r")
 ##逐个字节读取I/O流
a.each_byte{|byte| puts byte}
a.close

a = File.new("./input.json","r")
puts a.readlines.join("--")
a.close

```

- 读文件

```ruby

 ##更简单的读文件的方法,a为字符串
a = File.read("./input.json")
puts a.class ##string, not File
puts a

 ##文件指针
f = File.open("./input.json")
f.pos = 8 ##从第8个字节后开始读取
puts f.gets
puts f.pos


```

- 写文件

```ruby

 ##创建文件，“w”为只写，创建新文件
File.open("text.txt","w") do |f|
  f.puts "This is a test\nthis is another test"
end

 ##创建文件，可以持续写入"a",用来输出日志
f = File.open("logFile.txt","a")
f.puts Time.now
f.close

 ##创建文件，可读写"r+"
f = File.open("text.txt","r+")
f.write "12345"
f.close

f = File.open("text.txt","r")
puts f.gets ##输出一行
f.close

```
- 删除

```ruby

File.delete("file1.txt")

```

- 重命名

```ruby

File.rename("text.txt","test.txt")

```

- 文件是否存在

```ruby

b = File.exist?("file1.txt")

```

- 文件是否相同

```ruby

b = File.identical?("file1.txt","file2.txt")

```

- 文件是否读取到末尾

```ruby

f = File.open("test.txt","r")
f.each{ |line| puts line}
puts "end of file" if f.eof?
f.close

```

- 文件夹路径

```ruby

puts Dir.exist?
puts Dir.pwd

```

- 改变当前目录

```ruby

Dir.chdir("~/ruby/")
Dir.chdir("~/rails/")

```

- 列出当前路径下的所有文件

```ruby

puts Dir.entries("~/ruby/").join("\n")

```

- FileUtil

```ruby

require 'fileutils'
here = File.dirname(__FILE__) #__FILE__:built-in ruby value : 表示当前文件所在目录
puts here
puts File.expand_path(here)

puts Dir.entries(here) #列出here目录下所有文件和文件夹

 #fileutil创建目录
FileUtils.mkdir_p(here+'/stuff')
puts Dir.entries(here)

 #创建文件
FileUtils.touch('file1')
puts Dir.entries(here)

```




<h3 id="env">ENV</h3>

```ruby

puts RUBY_PLATFORM ##universal.x86_64-darwin13

ENV.each{ |e| puts e.join(': ')}

puts ARGV.join('-')

```

<h3 id="kernel">Kernel</h3>

kernel:ruby的kernel是一个公共库，里面定义了一些全局API

```ruby
puts "hello world"
```
等同于:

```ruby
Kernel.puts "Hello world"
```

<h3 id="object">Object</h3>

Object：所有全局函数都是Object对象的methods

全局函数：

```ruby
def func(a)
  puts a
end
```
调用全局函数:

```
func(10)
```
等同于：

```
Object.func(10)
```

<h3 id="date"> Date </h3>

```ruby


 #date
require 'date'
require 'time'

 #格式化时间：
date = Date.new(2014,4,1)
puts date.to_s
puts date.strftime("%m/%d/%Y")
puts date.strftime("%b/%d/%Y")
puts date.mday
puts date.day
puts date.year
 
 #date 可以解析
date2 = Date.parse("2014/4/3")
puts date2-date #2/1
puts (date2-date).to_i

 #time
time = Time.new
puts time

```

<h3 id="yaml"> YAML </h3>


```ruby

require 'yaml'
array = %w(dog cat frog)
puts array.to_yaml
hash = {:name => "jayson", :location =>"TB"}
puts hash[:name]
puts hash.to_yaml

 #yaml用于写配置文件
File.open('./config.yml','w+') {|f|
  
  f.puts hash.to_yaml
  
}

config = YAML.load(File.read('./config.yml'))
puts config.inspect

class Frog
  attr_accessor :name
end

frog = Frog.new
frog.name = "jayson"
yaml = frog.to_yaml
puts yaml.inspect

same_frog = YAML::load(yaml)
puts same_frog.inspect


```

<h3 id="erb"> ERB </h3>


```ruby

require 'erb'

hash = {:name => "jayson", :location =>"TB"}

 #以TEMPLATE开始到TEMPLATE结束之间的这段字符串会被复制到template这个变量中
 #ruby代码需要包含在<%= ... %>中执行
 
template = <<-TEMPLATE

From the desk of <%= hash[:name] %>
-----------------------------------
Welcome to <%= treehouse[:location] %>.

we hope you enjoy stay.
-----------------------------------

<% hash.keys.each{|k| %>
  key:<%= k %>
<% } %>

TEMPLATE

erb = ERB.new(template)
puts erb.result


```