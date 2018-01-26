---
layout: post
title: PL-C
categories: PL
tag: C

---

> 所有文章均为作者原创，转载请注明出处

# C语言

### C语言版本

- K&R C
	- 1978年，Kernighan和Ritchie的《The C Programmign Language》第一版出版，简称K&R C
- ANSI C 和 ISO C
	- 1989年，C语言被标准化，对K&R C进行了扩展，包括了一些新特性，规定了一套标准函数库
	- ISO成立WG14工作小组规定国际标准的C语言。
- C99	
	- ANSI标准化后，WG14小组继续改进C语言，1999年出版ISO9899:1999，即C99
- C11	
	- 2011年12月8日，ISO正式公布C语言新标准草案：ISO/IEC9899:2011, 即C11

- 标准的问题
	- C语言规范定义得非常宽泛
		- long型数据长度不短于int型
		- short型不长于int型
	- 导致：
		- 不同编译器有不同的解释
		- 相同的程序在不同的平台上运行结果不同
			- `int`在turboC上16位，在VC上32位
			- 对`++`	,`--`的解释不一致
			- 对浮点计算的精度不同等

### 变量命名

- **匈牙利命名法**
	- 以一个或多个小写字母开头，来指定数据类型
	- 其后是一个或多个第一个字母大写的单词，指出变量用途,如：
	- `chGrade, nLength, bOnOff, strStudentName`
	
- **驼峰命名**
	- 一个变量的名字由一个或多个单词连接
	- 第一个单词以小写字母开始
	- 后面单词的首字母大写
	- `myFirstName`

## 数据部分

### 整型

- **Signed vs Unsigned**:
	- `unsigned int i = 123`对应内存布局为: `00000000 00000000 00000000 01111011`
	- `signed int i = -123`对应的内存布局为: `11111111 11111111 11111111 10000101` 
	- 对于有符号数，第一个bit用来表示符号位,负数用1表示，正数用0
	- 无符号数在内存中以原码的形式存储，有符号数以补码的形式存储，计算方式为：无符号数取反+1，例如：`123原码 <-- 取反+1 --> -123` ,具体计算方式如下：
		1. 先确定符号位为1
		2. 求出123的原码：`10000000 00000000 00000000 01111011`
		3. 对原码部分各位取反: `11111111 11111111 11111111 10000100`
		4. +1: `11111111 11111111 11111111 10000101` 
	- 代码表示
		- 十六进制0x开头：`int a = 0xffffff12`
		- 八进制0开头：`int a = 037777777605`
	- 最大数
		- 无符号`unsigned int`: 
			- 十六进制： `0xffffffff`
			- 十进制：`4294967295` ， 大约42亿
		- 有符号`int`:
			- 十六进制 `0x7fffffff`
			- 十进制：`2147483647`，大约21亿
			- 二进制：`01111111 11111111 11111111 11111111` 
	- 最小数
		- 无符号: `0x000000`
		- 有符号: 
			-`0x7fffff`(最大有符号数) + `1` 
				- 二进制： `10000000 00000000 00000000 00000000`（最小有符号数）  
				- 十进制：`-2147483648`
				- C语言规定，当最高位是1，其它位是0时，最高位既表示负号，也表示正数最高位1			

- **浮点型**
	- float : 32bit 精度7位
	- double: 64bit 精度15位
	- long double: 64bit 精度15位

### 内存布局

- 以浮点型为例：
	- 第一位为符号位
	- 7位指数位：最多表示2的127次方，最多表示10的38幂
	- 24位二进制小数位：最多表示2的24次幂

	![](/assets/images/2007/05/float.png)

- 使用须知
	- 避免将一个很大数和和一个很小数相加或相减，否则会“丢失”小的数

	```c
	float a = 123456.789e5
	b = a + 20
	cout <<b<<endl
	//由于a是float型，它可写为：`1.23456789e10`,由于它的精度只有7位，因此实际上只能精确到`1.2345678e10`,所以b的值会计算错误
	```

### Const指针

- 不可以通过常量指针修改其指向的内容
- 不可以将常量指针赋值给非常量指针

```c
int n = 100;
int m = 101;
const int* p1 = &n;
int *p2 = &m;

//*p1 = 100; //compile error!
p1 = p2; //ok

printf("%d\n",*p1 ); //101
p2 = p1; //compile error!
p2 = (int* )p1; //ok
	
```
 
- 行数参数为常量指针，可以避免函数内部改变参数指针指向的值

```c
void myPrint(const char* p)
{
	//*p = "this";//compile error!
}
```

### 引用
	
- 定义引用时就初始化其值
- 初始化后，它就一直引用该变量，不会再引用别的变量了。
- 引用只能引用变量，不能引用常量和表达式

```c

int n = 7;
int &r = n;
r = 4;
cout << r; //4
cout << n; //4
n = 5;
cout << r; //5

```
	
- 引用做参数：`swap(int &a, int &b)`
- 引用作为函数的返回值：

```c

int n=40;
int& setValue()

int main()
{
	setValue()=40;
	cout<<n; //40
	return 0;
}

```

- const引用：

```c
int n=100;
const int& r = n;
r = 200; //compiler error
n = 300; //fine 

```

## 运算部分

### 赋值运算

- 将长数赋值给短数
	- 例如，将long型赋值给short型：

	```c
	int main()
	{
		long int long_i = 0x2AAAAAAA;
		short short_j = long_i;

		//long_i=00101010 10101010 10101010 10101010,会将低16bit赋值给short_j
		//即10101010 10101010
		//由于short_j是有符号数，那么第一位为1时为负数，即-21846
	}
	```

- 将短数赋值给长数
	- 将小数赋值给大数对于有符号数的规则为：
		- 若小数的高位为1，则大数的高位补1
		- 若小数的高位为0，则大数的高位补0

- 有符号数和无符号数互相赋值
	- 不考虑符号位

### 运算符

- 运算符优先级：

逻辑非`!` > 算术运算 >  关系运算 > `&&`和`||` > 赋值运算

-  逗号运算符
	- 运算符优先级别最低
	- 将两个表达式连接起来
		- `exp1`,`exp2`,`exp3`,...`expn` 
		- 先求`exp1`再求`exp2`,...,再求`expn`，整个表达式的值为表达式`n`的值,例如:`a = 3*5, a*4;`展开为`a=15,a*4`，结果为`60`
	- 考虑下面两个式子，x的值分别为多少？
		- `x=(a=3,6*3)` x=18
		- `x=a=3,6*3` x=3

### 位运算

- C语言中的位运算有
	- `&`
	- `|`
	- `^` 异或，双目运算(需要两个bit参与运算)
	- `~`
	- `<<`，左移
		- 高位左移后溢出，舍弃不起作用
		- 例如`a=15`，即`00001111`,左移2位得`00111100`,即十进制数60 ： `a = a<<2`
		- 左移1位相当于该数乘以2，左移两位相当于该数乘以2的平方
			- 只适用于高位溢出的舍弃bit不包含1的情况  
	- `>>` ，右移
		- 无符号数，低位移除舍弃，高位补0
		- 有符号数
			- 若原来的符号位为0，则左边移入0
			- 若原来的符号位位1，则左移移入0还是1，由操作系统决定
				- 若移入0，称为逻辑右移，或简单右移
				- 若移入1，称为算术右移 

- 常用的位运算
	- 使特定位翻转
		- 例如使`01111010`低4位翻转，可将其与`00001111`进行`^`运算，得到`01110101`

	- 使特定位保持不变
		- 与`0`进行`^`

	- 互换两个数的值，而不必使用临时变量
		- 例如`a=3,b=4`，交换`a,b`可用:

		```c
		a = a^b;
		b = b^a;
		a = a^b;
		```    
		
## 控制语句

### For

- for语句的定义

```

for(expr1; expr2; expr3)
{
	//语句
}

```

执行顺序:
- 先执行`expr1`
- 判断`expr2`是否为true，如果是true执行语句，如果是false则跳出
- 当语句执行完后执行`expr3`。
- 执行完`expr3`后重新执行`expr1`


### goto

- 无条件转向语句
- 它的一般形式为:
	- `goto 标识符`
	 


## 数组

### 一维数组

- 定义数组:

```c
float sheep[10];
int a2001[1000];

```

**数组大小不能为变量,可以为符号常量

- 数组初始化

	- `a[4] = {1,2,3,4};`
	- `a[ ] = {1,2,3,4};`
	- `a[4] = {1,2};`剩下的元素自动补0
	- `a[4] = {0};`初始化一个全0数组
	

### 二维数组

- 定义二维数组
	- `a[3][4]`：
		- 定义一个3行4列的数组
		- 相当于定义3个一维数组:`a[0]`,`a[1]`,`a[2]`

- 内存布局

```
a[0][0]
...
a[0][3]
a[1][0]
...
a[1][3]
a[2][0]
...
a[2][3]

```  

![](/assets/images/2007/05/2-dimension-array.png)



- 初始化
	- `a[3][4] = { {1,2,3,4}, {5,6,7,8}, {9,10,11,12} };`
	- `a[3][4] = { 1,2,3,4, 5,6,7,8, 9,10,11,12}`省略里面的括号
	- `a[][4] = { 1,2,3,4, 5,6,7,8, 9,10,11,12 }`
	- `a[][4] = { {1},{0,6},{0,0,11} }`缺的元素补0
	- `a[3][4] = { 0 }`

- 访问

```c

for (int i=0; i<3; i++)
{
	for(int j=0; j<4; j++)
	{
		cout << setw(3) << a[i][j]
	}
	cout << endl;
}

``` 

### 数组的应用

- 桶排序
- 使用下标做统计
	- 一维二维数组
- 寻找素数

```c
#include<iostream>
#include<cmath>

using namespace std;
int main(){

	int sum=0, a[100]={0};
	for(int i=2; i<sqrt(100.0); i++){
		
		sum = i;
		
		if(a[sum] == 0 ){
			
			while(sum < 100){
				sum +=i ;
				if(sum < 100){
					a[sum] = 1; //数组标记出能被i以及i的倍数整除的元素
				}
			}
		}
	}
	for(int i=2; i<100; i++){
		if(a[i] == 0){
			cout<<i<<"";//输出所有未被标记的即为素数
		}
	}
	
	return 0;
}

``` 

## 字符串

### 字符数组

- 定义
	- `char a[4] = {'a','b','c','d'}`
	- `char a[4] = {'a','b'}` 剩下的元素被初始化为`\0`
	- `char a[ ] = {'C','h','i','n','a'}`
	- `char a[ ] = "China"`使用这种方式初始化，数组末尾会自动多一个`\0`
		- `char a[5] = "China"`这种初始化方法是错误的
- 赋值
	- 只可以： 在数组定义并且初始化时:`char c[6] = "China"`
	- 不可以：不能用赋值语句将一个字符串常量或字符数组直接赋值给另一个字符数组
		- `str1[] = "China"`  错误
		- `str1 = "China"` 错误
		- `str2=str1` 错误

	- 正确的赋值方式:

	```c
	char str1 = "abc", str2[4];
	while(str1[i] != '\0')
	{
		str2[i] = str1[i];
		i++;
	}
	str2[3] = '\0';
	```
- 输出/输出
	- 使用`cout`输出字符数组，要确保数组以`\0`结尾
	- 使用`cin`输入字符数组时，默认空格和回车作为字符串间断
- 二位数组

```c
char weekday[7][11]={"Sunday","Monday","Tuesday","Wednesday","Thursday","Firday",""Saturday}.
for(int i=0;i<7;i++){
	cout<<weekday[i]<<endl;
}
``` 
### 常用字符串函数

- 字符串长度`strlen`
	- `int i = strlen(len)`
- 比较字符串`strcmp`
	- `bool v = strcmp(str1,str2)`
- 切分字符串 `strtok`

```c
 char input[] = "A bird came down the walk";
 char *token = strtok(input, " ");
 while(token) {
 	puts(token);
 	token = strtok(NULL, " ");//如果第一个参数为null，表示在前一次的位置继续向后查找
}
```


## 函数

### 函数的声明

- 函数原型：由函数的返回类型，函数名，以及参数表构成的一个符号串，其中参数可以不写名字，例如

```c
bool checkPrime(int)
``` 
- 函数的原型又称为函数的signature
- C语言中函数的声明就是使用函数的原型

### 函数的执行过程

例如：

```c

float max(float a, float b)
{
	return a>b?a:b;
}

int main()
{
	int m=3, n=4;
	float result = 0;
	
	result = max(m,n);
	
	cout << result;
	
	return 0;
	
}

```

1. 调用max函数时，开辟新的stack
2. 将参数m,n传递过去，max函数接收到的参数a,b是m,n的值，但是m,n有各自的存储空间
3. max执行完成后释放stack
4. 接收函数的返回值
5. 恢复现场，从断点处执行  

### 参数传递

- 实参与形参具有不同的存储单元，实参与形参变量的数据传递是“值传递”
- 函数调用时，系统给形参在函数的stack上分配空间，并将实参的值传递给形参
- 数组名做函数参数

```c
void change(int a[]){
	//...
}
```

### 递归

- 递归调用跟函数的嵌套调用没有区别，开辟新的空间
- 用递归解决具有递推关系的问题
	- 关注点放在求解目标上
	- 找到第`n`次和`n-1`次之间的关系
	- 确定第一次的返回结果
- 递归用来描述重复性的动作，代替循环
	- 连续发生的动作是什么 -> 确定递归函数，入参
	- 和前一次动作之间的关系 -> 通项公式
	- 边界条件是什么 -> 递归终止的边界
- 进行”自动分析“
	- 先假设有一个函数能给出答案
	- 在利用这个函数的前提下，分析如何解决问题
	- 搞清楚最简单的情况下答案是什么 	


- 常见的递归问题
	- 打印二进制
	
	```c
	void convert(int x){
		if((x/2)!=0){
			convert(x/2);
			cout<<x%2;
		}else{
		cout<<x;
		}
	}
	```
	
	- 汉诺塔问题

	```c
	/*两种解法：
	1. 可以先枚举—>递推->得到通项公式
	2. 简化问题:
	(1)移动2个 = 两次移动1个的次数 + 移动一次底座
	(2)移动3个 = 两次移动2个的次数 + 移动一次底座
	(3)移动n个 = 两次移动(n-1)个的次数 + 移动一次底座
	*/
	int  hanno(int n)
	{
		if (n == 1){
			return 1;
		}
		return 2*hanno(n-1)+1;	
	}
	```
	
	- 逆波兰表达式
	
	```c
	/*
	逆波兰表达式是一种把运算符前置的算术表达式
	如：2+3 的逆波兰表示法为 + 2 3 
	如：(2+3)*4 的逆波兰表示法为 x + 2 3 4
	输入：x + 11.0 12.0 + 24.0 35.0
	输出：1357.0
	*/
	
	//伪代码
	void reverse (deque<string> s){
	    string token = s.front();
	    s.pop_front();
	    if (token == "+") 
	    {
	        return notation(s) + notation(s);
	    }
	    else if (token == "-")
	    {
	        return notation(s) - notation(s);
	    }
	    else if (token == "x")
	    {
	        return notation(s) * notation(s);
	    }
	    else if (token == "/")
	    {
	        return notation(s) / notation(s);
	    }
	    else
	    {
	        return stof(token);
	    }
	}
	```
	
	

## 指针

### 指针与指针变量

- 某个变量的地址称为"指向该变量的指针"，注意:**"地址" == "指针"**
	- `0x0012ff78`这个地址就是它指向变量的指针
	- 例如，`http://www.nasa.gov/assets/images/content/166502.jpg`是一幅图片的指针 

- 存放地址的变量称为**指针变量**
	
	- 指针变量也有自己的地址
	
	- 定义:`int *pointer`
		- `pointer`是变量名
		- `*`代表变量的类型是指针
		- `int`表示指针变量的基类型，即指针变量指向变量的类型
	
	- 赋值，表达式:`&E`，取变量`E`的地址
	
	```c
	int *pointer;
	int c=100;
	pointer = &c;
	```
	- 访问，表达式：`*E`，如果`E`是指针，返回`E`所指向的内容
	
	```c
	int d = *pointer;
	*pointer = 49;
	```
	
- 例子

```c
#include <iostream>
using namespace std;

int main(){

    int count = 18;
    int *pointer = &count;
    *pointer = 58;
    cout<<count<<endl; //58
    cout<<pointer<<endl; // 0x7ffee67631d8
    cout<<&count<<endl; // 0x7ffee67631d8
    cout<<*pointer<<endl; // 58
    cout<<&pointer<<endl; //0x7ffee67631d0

    return 0;
}
```

- `&`和`*`的优先级
	- `*&a = *(&)a`
	- `&*a = &(*)a`
	- `(*a)++ != *(a++)`

![](/assets/images/2007/05/priority.png)

### 数组与指针

- 数组名代表数组元素的首地址
	- 数组名相当于指向数组第一个元素的指针
	- 对数组名取地址`&a`的值等同于数组第一个元素的地址`a`，但是含义不同
		- 返回基类型为数组的指针，意思是，当`&a+1`是，指针跨过的是整个数组长度和`a+1`不同
	
	```c
	int a[4] = {1,3,4,6};
	cout<<a<<end; 			//0x0028f7c4
	cout<<&a<<endl;			//0x0028f7c4
	cout<<a+1<<endl;			//0x0028f7c4 + 4  = 0x0028f7c8
	cout<<&a+1<<endl;		//0x0028f7c4 + 16 = 0x0028f7d4 
	cout<<*(&a)<<endl;		//0x0028f7c4
	cout<<*(&a)+1<<endl;	//0x0028f7c8
	```
	
- 指向数组的指针：
	- `int a[10]; int *p; p=a;`p为指向数组的指针


- 二维数组
	- 理解二维数组

	![](/assets/images/2007/05/2-dimension-array-1.png)
	
	- 索引二维数组	 
	
	```c
	int a[3][4] = {{1,3,5,7},{9,11,13,15},{17,19,21,23}};
	int (*p)[4] = a;
	for(int i=0;i<3;i++){
	    for(int j=0;j<4;j++){
	        cout<<*(*(p+i)+j)<<endl; 
	        // cout<<p[i][j]<<endl;
	    }
	} 
	```
	
	假设有数组`a`，是一个三行四列的数组，如何使用指针来索引？
	
	1. 从`p=a`开始
		- `a`相当于指向`a[3][4]`第一个元素的指针，所谓**第一个元素**即是第一个子数组`{1,3,5,7}`，所以`a`是一个第一个组数组的首地址
		- 指针`p`的基类型应该与`a`相同，即“包含四个整形元素的一维数组”
			- 定义：`int (*p)[4]`，`p`指向`a`的第一个子数组
	
	2. 分析`*(*(p+i)+j)`
		- `p+i`是第`i`个子数组的地址，等价于`&a[i]`
		- `*(p+i)`等价于`a[i]`
		- `*(p+i)+j`等价于`a[i]+j`等价于`&a[i][j]`
		- `*(*(p+i)+j)`等价于`a[i][j]`等价于`p[i][j]`

- 三条规律
	- 数组名相当于指向数组第一个元素的指针
	- `&E`相当于把`E`的管辖范围升了一级
	- `*E`相当于把`E`的管辖范围降了一级

### 字符串与指针

- 指向字符串的指针：
	- `char a[10]; char* p; p = a;`p为指向字符串的指针

### const

- `const`的意思是指向**符号常量**的指针，即指针所指向的内容为常量
- 定义`const`指针时，需要直接初始化其值
- 例1：

```c
{
	const int a = 78; const int b = 28; int c = 18;
	const int *pi = &a;
	*pi = 100; //error
	pi = &b; //可以
	pi = &c; *pi = 100; //error

}
```

### 函数指针

- `int (*POINTER_NAME)(int a, int b)`

- 将函数指针做参数传递，需要`typedef`一个类型

- 函数指针可以实现多态


## Struct

### 定义

- 语法定义: 
	- `struct STUDENT {...};` 
		- `STUDENT`表示数据类型，类似`int,char`等

- 定义结构体:
	- 直接使用已声明的结构体(struct + 结构体类型名 + 变量名):
		- `struct STUDENT stu1, stu2;`，例如：
		
		```c
		struct Person {
		    char *name;
		    int age;
		    int height;
		    int weight;
		};
		struct Person *Person_create(char *name, int age, int height, int weight)
		{
		    struct Person *who = malloc(sizeof(struct Person));
		    assert(who != NULL);
		    who->name = strdup(name);
		    who->age = age;
		    who->height = height;
		    who->weight = weight;
		    return who;
		}
		```

	- 使用`typedef`:
		- `typedef struct _STUDENT{...}STUDENT;`
			- 其中`_STUDENT`叫做struct tag，可以省略
			- 使用:`STUDENT stu1,stu2`  
	
	- 在声明类型的同时定义变量:

	```c
	struct Person {
	    char *name;
	    int age;
	    int height;
	    int weight;
	}xt1,xt2;
	```
	

### 赋值

- 结构体赋值，传参，做返回值

```c
struct student x1 = {1,2};
struct student x2;
x2 = x1;

```
x2中的值相当于x1中的值的copy，同理，结构体变量做函数参数和返回值也是copy的

### 应用

- 链表：一种非常常常用的数据结构
	- 链表头：指向第一个链表节点的指针
	- 链表节点：链表中的每一个元素，包括：
		- 当前节点的数据
		- 下一个节点的地址

	- 链表尾部：不在指向其他节点，其下一个节点的指针为NULL  

## 其他

### Std

- `strdup`: 会malloc一个新的string和原string一样
	
- `man 3 strdup`查看API说明


### MakeFile

当一个工程很大，有很多文件时，使用gcc去编译就局限了。这个时候通常使用makefile，makefile中，需要把这些文件组织到一起。
makefile是一个纯文本文件，实际上它就是一个shell脚本，并且对大小写敏感，里面定义一些变量。重要的变量有三个：

- CC ： 编译器名称

- CFLAGS ： 编译参数，也就是上面提到的gcc的编译选项。这个变量通常用来指定头文件的位置，常用的是-I, -g。

- LDFLAGS ：链接参数，告诉链接器lib的位置，常用的有-I,-L，

引用变量的形式是 : $(CC),$(CFLAGS),$(LDFLAGS)

例如：

<code> % gcc  main.c </code> 这样一条command在makefile中表达如下：

<code> $(CC) main.c </code>

例如：

<code> %gcc -g -I./ext -c main.c </code> 这样一条command在makefile中表达如下：

<code> CFLAGS = -g -I./ext </code>

<code> %(CC) $(CFLAGS) -c main.c </code>

一个简单的makefile：

```
CC = gcc
main:main.c main.h
        $(CC) main.c
```

必须要包含这三部分

main.o: main.c main.h

这句话的意思是main.o必须由main.c，main.h来生成

`$(CC)main.c`

是shell命令，前面必须加tab

针对上面的例子，我们可以写一个makefile 文件


```
C = gcc
CFLAGS = -g -I./ext/

PROG = p
HDRS = main.h module_1.h ./ext/module_2.h
SRCS = main.c module_1.c ./ext/module_2.c

$(PROG) : main.h main.c module_1.h ./ext/module_2.h
	$(CC) -o $(PROG) $(SRCS) $(CFLAGS)

```

## 参考文献


- <a href="https://www.google.com.hk/url?sa=t&rct=j&q=&esrc=s&source=web&cd=2&ved=0CC4QFjAB&url=%68%74%74%70%3a%2f%2f%66%6c%79%66%65%65%6c%2e%67%6f%6f%67%6c%65%63%6f%64%65%2e%63%6f%6d%2f%66%69%6c%65%73%2f%48%6f%77%25%32%30%74%6f%25%32%30%57%72%69%74%65%25%32%30%6d%61%6b%65%66%69%6c%65%2e%70%64%66&ei=DycDU9blG-TUigeM04GwAg&usg=AFQjCNGR312P8_ZhYaJaGVLK_7R6pgkSRA">陈皓：跟我一起学makefile</a>
- [Coursera](https://www.coursera.org/learn/c-chengxu-sheji/home/welcome)
- [Learn C The Hard Way](http://c.learncodethehardway.org/book/)