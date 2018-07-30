---
list_title:  PL-Haskell-1
layout: post
tag: Haskell
categories: PL

---

<em> </em>

>本文为"Learn You a Haskell for Great Good"读书笔记

##Chap1:概述

everything in haskell is function

```haskell
f (x) = x+x
doubleMe::(Num a)=>a->a
doubleMe x = x+x;

f(x) = x+1
doubleSmallNumber x = (if x > 100 then x else x*2)+1

f(x,y) = 2*x + 2*y
doubleXY x y = 2*x + 2*y
```

- 字符常量:

```haskell
conanO'Brien = "It's a-me, Conan O'Brien!"  
```

- 数组

```haskell
numbers = [1,2,2,3,4,5,6]
strings = ["ask","hell"]
```

- 数组拼接

```haskell
addNumbers = [1,2,3] ++ [3,2,1]
addStrings = ["haskell","java"] ++ ["lua","python"]
```

- 入队

```haskell
anotherStrings = "shit":["ask","hell"]
```

- 索引函数，x为索引值

```haskell
index x = [1,2,3,4,5,5] !! x 
```

- list比较，如果位数相同，则从第一个值开始向后比较

```haskell
compareResult1 = [3,2,1] > [1,2,3] true
compareResult2 = [3,4,2] > [99,88] false
```
- list方法

```haskell
head ： 5
listHead = head[5,3,2,1]

tail ： 2，3，1
listTail = tail[4,2,3,1]

last : 1
listLast = last [3,2,1]

init: 1,2,3
listInit = init [1,2,3,4]

length : 4
listLength = length [123,2,3,4]

```

take返回前几个:1,2,3

```haskell
listTake = take 3 [1,2,3,4,5,5]
```

null检测数组是否为空

```haskell
listNull = null[12,3]  false
listNull2 = null[]true
```

maximum/minimum数组最大值

```haskell
listMax = maximum [1,2,3]
listMin = minimum [3,2,1]
```

sum数组求和

```haskell
listSum = sum [1,2,3]
```

elem数组是否包含某个元素

```haskell
listEle = 4 `elem` [3,3,4,5,5]  true
```

range

```haskell
listRangeInteger = [1..20] 1,2,3,4,5,6...,20
listRangeString = ['a'..'z'] a,b,c,d..,z
listRangeEventInteger = [2,4..20]  2,4,6,8...,20
```

产生无限长的list，取前10个

```haskell
listCycle10Integer = take 10 (cycle[1,2,3]) 1,2,3,1,2,3,1,2,3,1
listCycle10String = take 10 (cycle"LOL ") LOL LOL LO
```

产生重复的list,取前10个

```haskell
listRepeat10Integer = take 10 (repeat(4)) 4444444444 
```

- 集合


例如：
```haskell
S = {2*x | x -> N, x <=10}
```
用Haskell表示为：1<=x<=10，2x的集合

```haskell
collectioA = [x*2 | x <- [1..10]]
```
1<=x<=10，2x>12，2x的集合

```haskell
collectionB = [x*2 | x <- [1..10],2*x >= 12]
```

50<=x<=100 && x % 7 == 3的集合

```haskell
collectionC = [x | x <- [50..100],x `mod` 7 == 3]
```

从一个 List 中筛选出符合特定限制条件的操作也可以称为过滤 (flitering)。
即取一组数并且按照一定的限制条件过滤它们。再举个例子吧
假如我们想要一个 comprehension，它能够使 List 中所有大于 10 的奇数变为 "BANG"，小于 10 的奇数变为 "BOOM"

```haskell
boomBangs list = [if x<10 then "BOOM" else "BANG" | x<-list,odd x]
```

从多个 List 中取元素也是可以的。
这样的话 comprehension 会把所有的元素组合交付给我们的输出函数。
在不过滤的前提 下，取自两个长度为 4 的集合的 comprehension 会产生一个长度为 16 的 List。
假设有两个 List，[2,5,10] 和 [8,10,11]， 要取它们所有组合的积，可以这样：

```haskell
multiplyList = [x*y | x<-[2,5,10],y<-[3,6,9]]
```

让我们编写自己的 length 函数吧！就叫做 length'!

```haskell
length' list = sum[1 | _<-list]
```
_ 表示我们并不关心从 List 中取什么值，与其弄个永远不用的变量，不如直接一个 _。
这个函数将一个 List 中所有元素置换为 1，并且使其相加求和

去除字符串中的小写字母

```haskell
removeNonUppercase list = [c | c<-list, c `elem` ['A'..'Z']]
```

- Tuple 元组

从某种意义上讲，Tuple (元组)很像 List 都是将多个值存入一个个体的容器。
但它们却有着本质的不同，一组数字的 List 就是一组数字，它们的类型相 同，且不关心其中包含元素的数量。
而 Tuple 则要求你对需要组合的数据的数目非常的明确，它的类型取决于其中项的数目与其各自的类型。
Tuple 中的项 由括号括起，并由逗号隔开。

Tuple类似hash map，但是有多个value


```haskell
tupleA = ("jayson","code",28)
```

tuple的长度是固定的，不能动态增减
fst 返回tuple的首项。
tupleHead = fst(8,11)

snd返回尾项

```haskell
tupleTail = snd("jsyon",False)
```

zip方法将k-v关联起来

```haskell
zipValue1 = zip[1..3]["a","b","c"]
```
zip不固定长度

```haskell
zipValue2 = zip[1,2,4,5]["a","b"] (1,"a")(2,"b")
```
zip不固定长度

```haskell
zipValue3 = zip[1..]["a","n","c"] (1,"a")(2,"n")(3,"c")
```

list 和 tuple

```haskell
tupleList = [(a,b,c) | a<-[1..10],b<-[2..9],c<-[3..8],a^2+b^2==c^2]
```


这便是函数式编程语言的一般思路：
先取一个初始的集合（入参）并将其变形（map），执行过滤条件（filter），最终取得正确的结果。

##Chap2:类型

- 变量类型

命令：:t用来查看变量的type

```haskell

ghci> :t 'a'   
'a' :: Char   

ghci> :t True   
True :: Bool   

ghci> :t "HELLO!"   
"HELLO!" :: [Char]   

ghci> :t (True, 'a')   
(True, 'a') :: (Bool, Char)   

ghci> :t 4 == 5   
4 == 5 :: Bool

```

- 函数类型

函数类型声明：入参->出参

```haskell

removeNonUppercase :: [Char] -> [Char]   
removeNonUppercase st = [ c | c <- st, c `elem` ['A'..'Z']]

```

入参->入参->入参->出参

```haskell

addThree :: Int -> Int -> Int -> Int   
addThree x y z = x + y + z

```

Types : Integer, Float, Double, Bool, Char
类型首字母必须大写


- Type variables

主要讨论函数的出参，和入参类型

```haskell

ghci> :t head   
head :: [a] -> a

```

意思是head这个function，入参是泛型的[a]，出参是泛型的a，这种都是泛型的出入参函数叫做“多态函数”

- Typeclasses

还是讨论函数的参数问题:有些函数的入参，出参类型是被约束的，它要服从其“父类”的类型约束条件
例如：

```haskell
ghci> :t (==)   
(==) :: (Eq a) => a -> a -> Bool

```

有意思。在这里我们见到个新东西：=> 符号。
它左边的部分叫做类型约束
我们可以这样阅读这段类型声明："相等函数取两个相同类型的值作为参数并回传一个布林值，而这两个参数的类型同在 Eq 类之中(即类型约束)"

怎么理解呢？
 “==” 这个函数的入参和出参类型，取决于Eq，因为 “==” 的父类就是Eq，
 Eq的子类还有“/=”

相似的例子还有Ord：

```haskell
ghci> :t (>)   
(>) :: (Ord a) => a -> a -> Bool

```

 “>”的参数类型取决于父类Ord
同理，ord包含了<, >, <=, >= 这几个接口

```haskell
show1 = show 3  "3"

 Main> :t (show)
(show) :: Show a => a -> String

```

show的参数类型取决于show自己，而show这个typeclass包含的参数类型为所有
3 是Integer，在包含的范围之内，因此可以转换为string

read将字符串转为某成员类型,类型根据第二个参数确定

```haskell
read1 = read "True" || False  True
read2 = read "8.2" + 3.8  12.0

```

read若只有一个参数，则需要提供一个参数类型，帮助其转换

```haskell
read3 = read "3" :: Int
read4 = (read "4.0"::Float)*4
read5 = read "[1,2,3,4]" :: [Int]   
[1,2,3,4]   
read6 = read "(3, 'a')" :: (Int, Char)  

```

Enum包含：

```haskell
succ，pred，[1..3](range)

```
参数类型包括：

```haskell
Main> :t ([1..3])
([1..3]) :: (Enum t, Num t) => [t]

```

##chap3 : 函数


- 参数匹配（代数）

```haskell
lucky2 :: (Integral a) => a -> String
lucky2 x = (if x==7 then "LuckNumber" else "wrongNumber")

```

等价于这个:

```haskell
lucky :: (Integral a) => a -> String
lucky 7 = "Luck Number!"
lucky x = "wrong number!"

```

在调用 lucky 时，模式会从上至下进行检查，一旦有匹配，那对应的函数体就被应用了。
这个模式中的唯一匹配是参数为 7，如果不是 7，就转到下一个模式，它匹配一切数值并将其绑定为 x 。

```haskell
sayMe :: (Integral a) => a -> String   
sayMe 1 = "One!"   
sayMe 2 = "Two!"   
sayMe 3 = "Three!"   
sayMe 4 = "Four!"   
sayMe 5 = "Five!"   
sayMe x = "Not between 1 and 5"  

```

同样会对参数a从上向下匹配
如果把x放到最前面，那么后面的条件都不会执行
其实这就是haskell的switch-case

- 求阶乘:

```haskell
factorial :: (Integral a) => a -> a   
factorial 0 = 1   
factorial n = n * factorial (n - 1)  

```

- 两个向量相加:限制a类型为num

```haskell
addVectors :: (Num a) => (a,a) -> (a,a) ->(a,a)
addVectors a b = (fst a+fst b, snd a+snd b)

```
使用模式匹配:

```

addVectors2 :: (Num a) => (a, a) -> (a, a) -> (a, a)   
addVectors2 (x1, y1) (x2, y2) = (x1 + x2, y1 + y2) 

```

对 List 本身也可以使用模式匹配。
你可以用 [] 或 : 来匹配它。
因为 [1,2,3] 本质就是 1:2:3:[] 的语法糖。
你也可以使用前一种形式
像 x:xs 这样的模式可以将 List 的头部绑定为 x，尾部绑定为 xs。
如果这 List 只有一个元素，那么 xs 就是一个空 List,只有一个头部元素x。

- 自己实现list的head方法

```haskell
headVal :: [a] -> a   
headVal [] = error "Can't call head on an empty list, dummy!"   
headVal (x:_) = x  

```

怎么理解这个呢？
首先匹配list，需要用()
其次，"-"代表匹配任意list
例如用户输入：[1,2,3]实际上匹配的是1:[2,3]自然结果是1

弄个简单函数，让它用非标准的英语给我们展示 List 的前几项。

```haskell
tell :: (Show a) => [a] -> String   
tell [] = "The list is empty"   
tell (x:[]) = "The list has one element: " ++ show x   
tell (x:y:[]) = "The list has two elements: " ++ show x ++ " and " ++ show y   
tell (x:y:_) = "This list is long. The first two elements are: " ++ show x ++ " and " ++ show y  

```

同上，如果输入为：[]匹配的是第一条
输入为：[1]匹配的是第二条：1:[]
输入为：[1,2]匹配的是第三条：1:2:[]
输入为:[1,2,3]匹配的是第三条：1:2:[3]

- list长度:用递归求数组长度 

```haskell
length'::(Num b) => [a] -> b
length' [] = 0
lenght' (_:xs) = length'(xs)+1

```

例如，输入为:[1,2,3]，匹配的是1:[2,3]依此递归下去,list求和，同样道理

```haskell
sum' :: (Num a) => [a] -> a   
sum' [] = 0   
sum' (x:xs) = x + sum' xs  

```

- Guards

模式用来检查一个值是否合适并从中取值，而 guard 则用来检查一个值的某项属性是否为真。
咋一听有点像是 if 语句，实际上也正是如此。
不过处理多个条件分支时 guard 的可读性要高些，并且与模式匹配契合的很好。

就可以把guard简单理解为if

```haskell
bmiTell :: (RealFloat a) => a -> String   
bmiTell x   
    | x <= 18.5 = "You're underweight, you emo, you!"   
    | x <= 25.0 = "You're supposedly normal. Pffft, I bet you're ugly!"   
    | x <= 30.0 = "You're fat! Lose some weight, fatty!"   
    | otherwise   = "You're a whale, congratulations!"  

```
	
 guard 由跟在函数名及参数后面的竖线标志，通常他们都是靠右一个缩进排成一列。
 一个 guard 就是一个布尔表达式，如果为真，就使用其对应的函数体。如果为假，就送去见下一个 guard，如之继续
注：bmiTell x和下一句间不能有空行

重写max函数

```haskell
max' :: (Ord a) => a -> a -> a   
max' a b    
    | a > b     = a   
    | otherwise = b  


```


- where

```haskell
bmiTell2 :: (RealFloat a) => a -> a -> String   
bmiTell2 weight height   
    | bmi <= skinny = "You're underweight, you emo, you!"   
    | bmi <= normal = "You're supposedly normal. Pffft, I bet you're ugly!"   
    | bmi <= fat    = "You're fat! Lose some weight, fatty!"   
    | otherwise     = "You're a whale, congratulations!"   
    where bmi = weight / height ^ 2   
          skinny = 18.5   
          normal = 25.0   
          fat = 30.0
```	
where必须定义在最后面，而且变量的名字必须排成竖行


- let-in

```haskell
let-in允许在函数中的任意位置定义局部变量

calcBmis :: (RealFloat a) => [(a, a)] -> [a]   
calcBmis xs = [bmi | (w, h) <- xs, let bmi = w / h ^ 2]

```

- case

	
用它可以对变量的不同情况分别求值，还可以使用模式匹配。
Hmm，取一个变量，对它模式匹配，执行对应的代码块。
好像在哪儿听过？啊，就是函数定义时参数的模式匹配！
{好吧，模式匹配本质上不过就是 case 语句的语法糖而已}

case表达式：

```haskell
case expression of pattern -> result   
                   pattern -> result   
                   pattern -> result   
                   ...  



head2 :: [a] -> a   
head2 xs = case xs of [] -> error "No head for empty lists!"   
                      (x:_) -> x  
                   
	
describeList :: [a] -> String   
describeList xs = "The list is " ++ case xs of [] -> "empty."   
                                               [x] -> "a singleton list."    
                                               xs -> "a longer list." 


```

##chap4:递归

haskell用递归代替for/while循环

- 递归求最大值

```haskell
maximumVal :: (Ord a)=>[a]->a
maximumVal [] = error "error"
maximumVal [x] = x
maximumVal (x:xs) 
	| x > maxTail = x 
	| otherwise = maxTail
	where maxTail = maximumVal xs
```
	
我们取个 List [2,5,1] 做例子来看看它的工作原理。
当调用 maximum' 处理它时，前两个模式不会被匹配，而第三个模式匹配了它并将其分为 2 与 [5,1]。
where 子句再取 [5,1] 的最大值。
于是再次与第三个模式匹配，并将 [5,1] 分割为 5 和 [1]。
继续，where 子句取 [1] 的最大值，这时终于到了边缘条件！回传 1。
进一步，将 5 与 [1] 中的最大值做比较，易得 5，现在我们就得到了 [5,1] 的最大值。
再进一步，将 2 与 [5,1] 中的最大值相比较，可得 5 更大，最终得 5。

改用 max 函数会使代码更加清晰。
如果你还记得，max 函数取两个值做参数并回传其中较大的值。
如下便是用 max 函数重写的 maximun'


```haskell
maximum' :: (Ord a) => [a] -> a   
maximum' [] = error "maximum of empty list"   
maximum' [x] = x   
maximum' (x:xs) = max x (maximum' xs)
```

- 实现take

```haskell
take' :: (Num i, Ord i) => i -> [a] -> [a]
take' n _
	 | n <= 0 = []
take' _ [] = []
take' n (x:xs) = x : take' (n-1) xs
```

- 实现reverse

```haskell
reverse' :: [a] -> [a]
reverse' [] = []
reverse' (x:xs) = reverse' xs ++ [x]
```

- 实现zip

```haskell
zip' :: [a] -> [b] -> [(a,b)]   
zip' _ [] = []   
zip' [] _ = []   
zip' (x:xs) (y:ys) = (x,y):zip' xs ys
```

- 快速排序

```haskell
quicksort :: (Ord a) => [a] -> [a]   
quicksort [] = []   
quicksort (x:xs) =   
  let smallerSorted = quicksort [a | a <- xs, a <= x]  
      biggerSorted = quicksort [a | a <- xs, a > x]   
  in smallerSorted ++ [x] ++ biggerSorted
```

##chap5 : 高阶函数


其实理解高阶函数从数学的角度比较好理解

```
f(x) = x^2
g(x) = 2x+1
f(g(x)) = (2x+1)^2 = 4x^2+4x+1
```

这种就是高阶函数

- currying:

把一个函数的多个参数分解成多个函数
然后把函数多层封装起来，每层函数都返回一个函数去接收下一个参数这样，可以简化函数的多个参数。

以max为例

```haskell
getMaxValue x  = max 4 5
```

把空格放到两个东西之间，称作函数调用。它有点像个运算符，并拥有最高的优先级。

这个等价于：

```haskell
getMaxValue2 x = (max 4) 5
```

理解:

根据currying，(max 4)会拿4和内部的一个变量比较，这个变量值可能是0，比较后这个变量的值为4
然后它返回一个函数，这个函数接受一个参数，并且内部已经有了一个值了，这个值可能为4
最后将参数5传入这个函数，这个函数拿5和内部的变量4比较，则返回5

再看一个乘法：

```haskell
multThree :: (Num a) => a -> a -> a -> a 
multThree x y z = x * y * z
```

假设输入为multThree 3 5 9，根据空格，它的执行过程为：

```haskell
(((multThree 3)5)9)
```

比较大小的函数：

```haskell
compareWithHundred :: (Num a,Ord a) => a -> Ordering 
compareWithHundred x = compare 100 x 
```

这个函数也可以忽略掉参数

```haskell
compareWithHundredWithoutX :: (Num a,Ord a) => a -> Ordering 
compareWithHundredWithoutX = compare 100 

applyTwice :: (a -> a) -> a -> a   
applyTwice f x = f (f x)

```

首先注意这类型声明。 
在此之前我们很少用到括号，因为 (->) 是自然的右结合，不过在这里括号是必须的。
它标明了首个参数是个参数与回传值类型都是a的函数，第二个参数与回传值的类型也都是a
我们姑且直接把它看作是取两个参数回传一个值，其首个参数是个类型为 (a->a) 的函数,第二个参数是个 a

例如：

```haskell
ghci> applyTwice (+3) 10   
```

这里第一个参数为fx = x+3，x省略了，第二个参数为10
计算时是： f(10)= 13 13+3 = 16
数学模型为：f(f(x)) = (x+3)+3 = x+6

实现zipwith,求两个数组的 f 操作

```haskell
zipWith' :: (a -> b -> c) -> [a] -> [b] -> [c]   
zipWith' _ [] _ = []   
zipWith' _ _ [] = []   
zipWith' f (x:xs) (y:ys) = f x y : zipWith' f xs ys

```

例如求和

```haskell
ghci> zipWith' (+) [4,2,5,6] [2,6,2,3]   
[6,8,7,9]

```
首先f是一个接受两个参数，返回一个参数的函数
f x y 返回的就是 x+y 

各种花样

```haskell
ghci> zipWith' max [6,3,2,1] [7,3,1,5]   
[7,3,2,5]   
ghci> zipWith' (++) ["foo "，"bar "，"baz "] ["fighters"，"hoppers"，"aldrin"]   
["foo fighters","bar hoppers","baz aldrin"]   
ghci> zipWith' (*) (replicate 5 2) [1..]   
[2,4,6,8,10]   
ghci> zipWith' (zipWith' (*)) [[1,2,3],[3,5,6],[2,3,4]] [[3,2,2],[3,4,5],[5,4,3]]   
[[3,4,6],[9,20,30],[10,12,12]]

```
- map：取一个函数和 List 做参数，遍历该 List 的每个元素来调用该函数产生一个新的 List。 看下它的类型声明和实现:

```haskell
map :: (a -> b) -> [a] -> [b]   
map _ [] = []   
map f (x:xs) = f x : map f xs

```

从这类型声明中可以看出，它取一个取 a 回传 b 的函数和一组 a 的 List，并回传一组 b。 
这就是 Haskell 的有趣之处：有时只看类型声明就能对函数的行为猜个大致。
map 函数多才多艺，有一百万种用法。如下是其中一小部分:

```haskell
ghci> map (+3) [1,5,3,1,6]   
[4,8,6,4,9]

```

这个和list comprehension相同[x+3 | x <- [1,3,2,3,5]] 
map多个条件也仅仅遍历一遍数组


filter：函数取一个限制条件和一个 List，回传该 List 中所有符合该条件的元素

```haskell
filter :: (a -> Bool) -> [a] -> [a]   
filter _ [] = []   
filter p (x:xs)    
    | p x       = x : filter p xs   
    | otherwise = filter p xs


ghci> filter (>3) [1,5,3,2,1,6,4,3,2,1]   
[5,6,4]   

```

takeWhile 函数，它取一个限制条件和 List 作参数，然后从头开始遍历这一 List，并回传符合限制条件的元素

所有小于 10000 的奇数的平方和
ghci> sum (takeWhile (<10000) (filter odd (map (^2) [1..])))   


- lambda:就是匿名函数。有些时候我们需要传给高阶函数一个函数，而这函数我们只会用这一次，这就弄个特定功能的 lambda

编写 lambda，就写个 \ (因为它看起来像是希腊字母的 lambda  如果你斜视的厉害)
后面是用空格分隔的参数，-> 后面就是函数体。
通常我们都是用括号将其括起，要不然它就会占据整个右边部分。

例如，表达式:

```haskell
map (+3) [1,6,3,2] 与 map (\x -> x+3) [1,6,3,2]

```

对于多个参数：

```haskell
ghci> zipWith (\a b -> (a * 30 + 3) / b) [5,4,3,2,1] [1,2,3,4,5] 

```


- fold : fold 取一个二元函数，一个初始值(我喜欢管它叫累加值)和一个需要折叠的 List。
这个二元函数有两个参数，即累加值和 List 的首项(或尾项)，回传值是新的累加值。
然后，以新的累加值和新的 List 首项调用该函数，如是继续。
到 List 遍历完毕时，只剩下一个累加值，也就是最终的结果。


首先看下 foldl 函数，也叫做左折叠。
它从 List 的左端开始折叠，用初始值和 List 的头部调用这二元函数，得一新的累加值，并用新的累加值与 List 的下一个元素调用二元函数。
如是继续。

我们再实现下 sum，这次用 fold 替代那复杂的递归：

```haskell
sum' :: (Num a) => [a] -> a   
sum' xs = foldl (\acc x -> acc + x) 0 xs

```

acc是累加值
由于使用foldl，那么x是数组的第一个元素
0 是起始值
计算过程是拿数组的x和acc相加，作为新的x，然后递归

这条语句等于foldl (+) [1,2,3]

实现'elem'

```haskell
elem' :: (Eq a) => a -> [a] -> Bool   
elem' y ys = foldl (\acc x -> if x == y then True else acc) False ys

```

所有遍历 List 中元素并据此回传一个值的操作都可以交给 fold 实现。
{!无论何时需要遍历 List 并回传某值，都可以尝试下 fold!}。
因此，fold的地位可以说与 map和 filter并驾齐驱，同为函数式编程中最常用的函数之一。

foldl1是fold的简化版，初始值默认为数组的第一个元素
上面求和的函数用foldl1实现如下：

```haskell
sum1 xs = foldl1 (\acc x -> acc + x) xs

```

这条语句等于:

```haskell
foldl1 (+) [1,2,3]

```
scanl 和 scanr 与 foldl 和 foldr 相似，只是它们会记录下累加值的所有状态到一个 List。
也有 scanl1 和 scanr1。

```haskell
ghci> scanl (+) 0 [3,5,2,1]   
[0,3,8,10,11]   
ghci> scanr (+) 0 [3,5,2,1]   
[11,8,3,1,0]  

``` 


- Function composition:函数的迭代：f(g(x)) 在haskell中的表示

先看一个map函数

```haskell
val = map (\x -> negate (abs x)) [5,-3,-6,7,-3,2,-19,24]   
[-5,-3,-6,-7,-3,-2,-19,-24]

```

先求绝对值再求反
也可以合并执行：

```haskell
val2 = map (negate . abs) [5,-3,-6,7,-3,2,-19,24]   

oddSquareSum :: Integer   
oddSquareSum = sum (takeWhile (<10000) (filter odd (map (^2) [1..])))

oddSquareSum :: Integer   
oddSquareSum = sum . takeWhile (<10000) . filter odd . map (^2) $ [1..]

oddSquareSum :: Integer   
oddSquareSum =    
    let oddSquares = filter odd $ map (^2) [1..]   
        belowLimit = takeWhile (<10000) oddSquares   
    in  sum belowLimit

```

