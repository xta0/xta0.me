---
list_title:  学习 Haskell (二) | Learn You a Haskell for Great Good | Recursiion & HOF
title: Learn You a Haskell for Great Good
layout: post
tag: Haskell
categories: PL
---

>本文为"Learn You a Haskell for Great Good"读书笔记

## 递归

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

## 高阶函数


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

## Resources

- [Learn You a Haskell for Great Good!](http://learnyouahaskell.com/)
- [Haskell-Book]( http://www.cs.nott.ac.uk/~gmh/book.html)