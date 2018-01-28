---
layout: post
title: PL-C++
categories: PL
tag: C++

---

> 所有文章均为作者原创，转载请注明出处

### C/C++语言历史

- 1954-1956 IBM的John Backus和他的研究小组研发了FORTRAN
- 1960年1月 Alan J.Perlis发明了`Algol 60`<Report on the Algorithmic Language ALGOL 60>，A语言诞生
- 1963年 剑桥大学发明CPL, 1967年剑桥大学Matin Richards简化了CPL，推出了BCPL语言
- 1970年贝尔实验室，Ken Thompson在一台废弃的PDP-7机器上简化了BCPL，发明了B语言，并用B语言写了一个操作系统叫UNIX
- 1972-1973 Dennis Ritchie在B语言的基础上，发明C语言，并重写了UNIX 
- 1978: K&RC《The C Programming Language》
- 1979 贝尔实验室Bjarne Stroustrup开发了C++
- 1985年10月 Bajarne完成了《C++ Programming Language》第一版
- 1989: ANSI C 和 ISO C
- 1998年，C++ 98标准诞生
- 1999: C99
- 2011年，C++ 11 

## 内存分配

### new

- 创建一个T类型的指针：`T* p = new T;`

```c++

int *pn = NULL;
pn = new int(5);

```
- 创建一个T类型的数组：`T* p = new T[N];`

### delete

- delete两次会报错

```c++

int* p = new int(5);
delete p;
delete p; //error!

```

- delete 数组：

```c++

int *p = new int[20];
p[0] = 1;
delete[] p;

```

## 函数

### Method Overload

- C++允许函数重载:

```c++

int max(int a,int b, int c);

int max(double f1, double f2);

```

上面两条函数声明在C++中不会报错，但是在C中因为函数的符号相同会报错

```c++

int max(double a,double b);

void max(double f1, double f2); //error

```

上面两条函数声明会报错，因为仅返回值类型不同不属于重载，属于方法的重复定义。

### 缺省函数

```c
void func(int x, int y=1, int z=2){}

func(10); 
func(10,2);
func(10,,9);//error

```

## 类


### 构造函数

- 成员函数的一种
	- 名字与类名相同，可以有参数，不能有返回值
 	- 作用是对对象进行初始化，给成员变量赋值
 	- 如果没定义构造函数，编译器生成一个默认的无参数的构造函数
- 对象不论以什么样的形式创建都会调用构造函数
 
- 拷贝构造函数：
	- `X::X(X&)`
	- `X::X(const X &)` 
	- 如果没有定义拷贝构造函数，则系统默认生成一个
	- 拷贝构造函数，如果涉及成员变量指向一片内存空间的，需要使用深拷贝，赋值被拷贝对象的内存空间

```c++

class A
{
	private: 
		double real,image;
		
	A(const A &c)
	{
		real = c.real;
		image = c.image;
	}
};

void func(A a){ ... } 

A a1;
A a2(a1);
A a3 = a1; //初始化语句，非赋值语句
func(a1); //这种情况也会调用，在func的stack中，创建的a是通过拷贝构造函数创建的


```

- 类型转换构造函数
	- 只有一个参数
	- 不是拷贝构造函数

```c++

class B
{
	public:
		double real, image;
		B(int i)
		{
			real = i;
			image = 0;
		}
};


B  b1 = 12; //触发类型转换构造函数
b1 = 20; //20会被自动转换成一个临时B对象，同样会触发类型转换构造函数

```

### 析构函数

### 静态成员函数&成员变量

- 静态成员函数：相当于类方法，不作用于某个对象，本质上是全局函数
- 静态成员变量：该类的所有对象共享这个变量,使全局变量
- sizeof运算符不会计算静态成员变量
- 访问:
	- `类名::成员名`: `CRectangle::PrintTotal();`
	- `对象名.成员名`: `CRectangle r; r.PrintTotal();`
	- `指针->成员名`:	`CRectangle* p = &r; p -> PrintTotal();`
	- `引用.成员名`: `CRectangle &ref = r; int n = ref.nTotalNumber`

- 必须在类定义的文件中对静态成员变量做一次说明或初始化。

	
### 常量成员函数&常量对象&常引用


- `const`成员函数不能修改成员变量，不能访问成员函数，本质上是看这个函数会不会有修改对象状态的可能性

- `const`成员函数也可作为构造函数，算重载
	 
- 常量对象不能修改成员成员变量，不能访问成员函数

```c++

class Hello
{
private:
	int value;
public:
		void getValue() const;
		void foo(){}
};

void Hello::getValue() const
{
	value = 0;//wrong;
	foo(); //error
}

int main()
{
	const Hello o;
	
	o.value = 100; //wrong!
	o.func(); //wrong!
	o.getValue(); //ok
	
	return 0;
} 

```
- 对象作为函数的参数时，生成该参数需要调用复制构造函数，效率较低。用指针做参数，代码不好看，如何解决？可以用对象的引用作为参数：

```c++

class Sample{ ... };

void printSample(Sample& o)
{
	...
}

```

对象引用作为函数参数有一定风险，如果函数中不小心修改了o，是我们不想看到的。解决方法是将函数参数声明为const，这样就确保了o的值不会被修改

```c++

void printSample(const Sample& o)

```


### 封闭类和初始化列表

- 一个类的成员变量是另一个类对象，包含成员对象的类叫封闭类

```c++

class Car
{
private:
	int price;
	Engine engine;
public:
	Car(int a, int b, int c);

};

//初始化列表
Car::Car(int a, int b, int c):price(a),engine(b,c)
{

};

```

- 这种情况，Car类必须要定义构造函数来初始化engine，如果使用默认构造函数，编译器无法知道Engine类对象该如何初始化。

### 友元

允许类的私有成员被外部访问，friend属性不可传递和继承：

- 友元函数 
- 友元类

```c++

class Car
{
private:
	int price;

public:
	Car(int p):price(p){}

//友元函数
friend int mostExpensiveCar(Car* pCar);
//友元类
friend class Driver;

};

int mostExpensiveCar(Car* pCar)
{
	//访问car的私有成员
	printf("Car.price:%d\n",pCar->price);
};

class Driver
{
public:
	void getCarPrice(Car* pCar){printf("%s_Car.price:%d\n",__FUNCTION__,pCar->price);};
};


Car car(100);//赋值构造函数
mostExpensiveCar(&car);  //友元函数
Driver driver; //友元类
driver.getCarPrice(&car);

```

### 理解this指针

在早期c++刚出来时，没有编译器支持，因此需要将c++翻译成c执行，例如人下面一段程序：

```c++

class Car
{
public:
	int price;
	void setPrice(int p); 
};

void Car::setPrice(int p)
{
	price = p;
}

int main()
{
	Car car;
	car.setPrice(100);
	return 0;
}

```

被翻译成：

```c

struct Car
{
	int price;
};

void setPrice(struct Car* this, int p)
{
	this -> price = p;
}

int main()
{
	Car car;
	setPrice(&car, 100);
	return 0;
}

```

- class对应struct
- 成员函数对应全局函数，参数多一个`this`
- 所有的C++代码都可以看做先翻译成C后在编译

理解下面一段代码：

```c++

class Hello
{
	public:
		void hello(){printf("hello!\n");};
};

int main(int argc, char** argv)
{
	Hello* p = NULL;
	p -> hello();
}

```

程序会正常输出hello，原因是成员函数`void hello()`会被编译器处理为：`void hello(Hello* this){printf("hello!\n");}`与this是否为NULL没关系。

- 静态成员函数中不能使用`this`指针，它不作用于某个对象，因此静态成员函数就是c语言的全局函数，没有多余的参数。

## 继承

- 继承：继承类拥有基类全部的成员函数和成员变量，不论是private,protected,public
 	- 在子类的各个成员函数中，不能访问父类的private成员

```
class A : public B
{

};

``` 	
- 子类对象的体积，等于父类对象的体积 + 子类对象自己的成员变量的体积。在子类对象中，包含着父类对象，而且父类对象的存储位置位于派生类对象新增的成员变量之前

### 子类的构造函数

- 执行子类的构造函数之前
	- 先执行基类的构造函数：
		- 初始化父类的成员变量
	- 调用成员对象类的构造函数
		- 初始化成员对象

- 执行完子类的析构函数后：
	- 调用成员对象类的析构函数
	- 调用父类的析构函数  
	
- 子类交代父类初始化，具体形式:

```
构造函数名:基类名(积累构造函数实参表)
{

};

```

例子：

```c

class Bug
{	
private:
	int nlegs; int nColor;
public:
	Skill s1, s2;
	int ntype;
	Bug(int legs, int color);
	void printBug(){}; 

};


class FlyBug : public Bug
{
	int nWings;
public:
	FlyBug(int legs, int color, int wings):Bug(legs,color),s1(10),s2(100)
	{
		nWings = wings;
	}; 

};

```

- 调用基类构造函数的两种方式
	- 显式方式：派生类构造函数中 -> 基类的构造函数提供参数`derived::derived(arg_derived-list):base(arg_base-list)`

	- 隐式方式：默认调用父类的默认构造函数

	
### 访问控制

- 如果子类定义了一个和父类同名的成员变量，访问父类的成员变量时需要使用`base::i = 5`;
- 成员函数同理：`base::func();`

```c

derived obj;
obj.i = 1;
obj.base::i = 1;

```

- 访问范围说明符
	- 基类的private成员：可以被下列函数访问：
		- 基类的成员函数
		- 基类的友员函数

	- 基类的public成员：可以被下列函数范根：
		- 基类的成员函数
		- 基类的友员函数
		- 派生类的成员函数
		- 派生类的友员函数
		- 其他函数
	
	- 基类的protected成员：可以被下列函数访问：
		- 基类的成员函数
		- 基类的友员函数
		- 派生类的成员函数可以访问当前类的protected成员   

### public继承的赋值兼容规则

```c
class base{};
class derived:public base{};
base b;
derived d;

```

- 派生类的对象可以赋值给基类对象:
	- `b=d;`

- 派生类对象可以初始化基类引用:
	- `base &br = d;`

- 派生类对象的地址可以赋值给基类指针:
	- `base* pb = &d;` 

## 多态

### 多态的表现形式

- 派生类的指针可以赋值给基类指针
- 通过基类指针调用基类和派生类中的同名虚函数时：
	- 若该指针指向一个基类对象，那么被调用的函数是基类的虚函数
	- 若该指针指向一个派生类对象，那么被调用的函数是派生类的虚函数 

### 虚函数

- 在类定义中，前面有virtual关键字的成员函数就是虚函数

```c

class base{
	
	virtual int get();

};
int base::get(){ return 0; };

```

- virtual关键字只用在类定义里的函数声明中，写函数体时不用。
- 构造函数和静态成员函数不能是虚函数


### 多态的原理

多态的关键在于通过基类指针或引用调用一个虚函数时，编译时不确定到底调用的是基类还是派生类的函数，运行时才决定。

- 虚表:

每一个有虚函数的类（或有虚函数的类的派生类）都有一个虚函数表，该类的任何对象中都放着虚函数表的指针。虚函数表中列出了该类的虚函数地址。多出来的4个字节就是用来存放虚函数表的地址的

- 开销：

多态的函数调用语句被编译成一系列根据基类指针所指向的对象中存放的虚函数表的地址，在虚表中查找虚函数的地址，并调用虚函数指令。

这么做有两点开销，一是每个有虚函数的类都会多出4字节大小，二是需要查虚函数表，会有时间消耗

### 虚析构函数

- 问题：

```c

class CSon{
	public: ~CSon(){};
};

class CGrandson: CSon{
	public: ~CGrandson(){};

}; 

int main()
{
	CSon* p = new CGrandson();
	delete p;
	return 0;
}

```

由于new出来的是`CGrandson`对象，当`delete p`的时候调用了`CSon`的析构函数，没有调用`CGrandson`的析构函数。也就是说通过基类的指针删除派生类的对象时，只调用基类的析构函数。因此我们希望析构函数也能多态，解决办法：

- 把基类的析构函数声明为`virtual`
	- 派生类的析构函数`virtual`可以不进行声明
	- 通过基类指针删除派生类对象时
		- 首先调用派生类的析构函数
		- 然后调用基类的析构函数

- 类如果定义了虚函数，最好将析构函数也定义为虚函数

### 纯虚函数

- 没有函数体的虚函数

```

class A{

	private int a;
	public:
		virtual void print() = 0;
		void fun(){ cout << "fun"; }
};

```

- 抽象类：包含纯虚函数的类
	- 只能作为基类来派生新类使用
	- 不能创建抽象类的对象
	- 抽象类的指针和引用 -> 由抽象类派生出来的类的对象
	- 在抽象类的成员函数内可以调用纯虚函数
	- 在构造函数/析构函数内不能调用纯虚函数

```c

A a; //错，A是抽象类，不能创建对象
A* pa; //ok，可以定义抽象类的指针和引用
pa = new A; //错误，A是抽象类，不能创建对象

```   

- 如果一个类从抽象类派生而来
	- 它实现了积累中所有的纯虚函数，才能成为非抽象类 

## 运算符重载

### 算术运算符重载

- 运算符重载的实质是函数重载

```

返回值类型 operator 运算符（形参表）{


}

```

- 在程序编译时：
	- 把运算符的表达式 -> 对运算符函数的调用
	- 把运算符的操作数 -> 运算符函数的参数
	- 运算符多次被重载时，根据实参类型决定调用哪个运算符函数
	
- 运算符可以被重载成普通函数
	- 参数个数为运算符数目
	
```c++

class Complex
{
	public:
		Complex(double r = 0.0, double i = 0.0)
		{
			real = r;
			image = i;
		}
		
	double real;
	double image;
};

Complex operator+ (const Complex& a, const Complex& b){
	return Complex(a.real+b.real, a.image+b.image);
}

```

- 也可以被重载成类的成员函数 
	- 参数个数为运算符目数减一 

```c++

class Complex
{
	public:
		Complex(double r = 0.0, double i = 0.0)
		{
			real = r;
			image = i;
		}
		
		Complex operator+ (const Complex& );
		Complex operator- (const Complex& );
		
	private:
		double real;
		double image;
};

Complex Complex::operator+(const Complex& op)
{
	return Complex(real+op.real,image+op.image);
}

int main()
{
	Complex x, y(4.3,2.6), z(3.3,1.1);
	x = y+z; //=> x = y.operator+(z)

	return 0;
}

```
### 重载`<<`

C++中的`cout<<`使用的也是运算符重载，`cout`是`ostream`类的对象，`ostream`重载了`<<`:

```c++

ostream& ostream::operator<<(int n)
{
	///...
	
	
	return *this;
}

```

而`cout<<3<<"this";`等价于:

```c

cout.operator<<(3).operator<<("this");;

```

### 赋值运算符重载

- 赋值运算符两边类型可以不匹配
- 赋值运算符“=”只能重载为“成员函数”
- 返回值不能设计成void，会有`a=b=c`的情况
- 返回值要设计成引用类型

### 运算符重载为友元函数

- 成员函数不能满足使用要求
- 普通函数，又不能访问类的私有成员


### 自加/自减运算符重载

- 自加`++`, 自减`--`运算符有前置/后置之分
- 前置运算符为一元运算符重载

- 重载为成员函数：
	
```c
	
T operator++();
T operator--();
	
```  
- 重载为全局函数:

```c
	
T operator++(T);
T operator--(T);
	
``` 

- 后置运算符作为二元运算符重载
	- 多写一个参数，具体无意义
- 重载为成员函数:

```c

T operator++(int);
T operator--(int);

```

- 重载为全局函数:

```c

T operator++(T, int);
T operator--(T, int);

```   

## 泛型

- 泛型
	- 算法实现不依赖于具体的数据类型
	- 大量编写模板，使用模板的程序设计
		- 函数模板
		- 类模板

### 函数模板

- 函数模板:

```
template<class 参数1, class 参数2,...>
返回值类型 模板名(形参表)
{
	函数体
}

``` 

- 函数模板可以重载，只要它们形参表不同即可
	- 例如，下面两个模板可以同时存在:

```cpp

template<class T1, class T2>
void print(T1 arg1, T2 arg2)
{
	cout << arg1 << "" << arg2<<endl;
}

template<class T>
void print(T arg1, T arg2)
{
	cout << arg1 << "" <<arg2 <<endl;
}

```  

- C++编译器遵循以下优先顺序

	- 先找参数完全匹配的普通函数（非由模板实例化而得的函数）
	- 再找参数完全匹配的模板函数
	- 再找实参经过自动类型转换后能够匹配的普通函数
	- 上面的都找不到，则报错

### 类模板

- 定义类的时候给它一个/多个参数
- 这些参数表示不同的数据类型

```c++

template<类型参数表>
class 类模板名
{
	

};

```

- 类型参数表的写法就是:`class 类型参数1, class 类型参数2,...`

- 类模板里的成员函数，如在类模板外面定义时,

```c++

template<形参表>
返回值类型 类模板名<类型参数名列表>::成员函数名(参数表)
{

}

```

- 用类模板定义对象的写法:

`类模板名<真实类型参数表> 对象名(构造函数实际参数表)`

- 如果类模板有无参数构造函数，也可以直接写:

`类模板名<真实类型参数表> 对象名`

- 编译器由类模板生成类的过程叫类模板实例化
	- 编译器自动用具体的数据类型替换模板中的类型参数，生成模板类的代码

- 由类模板实例化得到的类叫模板类
	- 为类型参数指定的数据类型不同，得到的模板类不同

- 同一个类模板的两个模板类是不兼容的

```cpp

Pair<string, int>* p;
Pair<string, double> q;
p = &q; //wrong!

```

- 函数模板作为类模板成员

```cpp

template<class T>
class A
{
	pubic:
		template<class T2>
		void Func(T2 t){cout<<t;}; //成员函数模板
};

```

- 类模板的参数声明中可以包括非类型参数

```cpp

template<class T, int ele>

```

- 非类型参数：用来说明类模板中的属性
- 类型参数：用来说明类模板中的属性类型，成员操作的参数类型和返回值类型

```cpp

CArray<double,40> a2;
CArray<int,50> a3;

```

- 注意：`CArray<double,40>`和`CArray<int,50>`完全是两个类，这两个类对象之前不能相互赋值 


### string类

- string类是一个模板类，它的定义如下：

```cpp

typedef basic_string<char> string;

```

- 使用string类要包含头文件`<string>`

- 初始化:

```cpp

string s1("hello");
string s2(8,'x');
string s3 = "world";

```

- 常用API：
	- 长度：`length()`
	- 赋值: `assign()`: `string s1("abc"),s3; s3.assign(s1,1,3)`
	- 访问每个字符：`[]` 或者 `at(index)`
	- 拼接: `+`,`append()`
	- 字串:`substr(4,5); //下标4开始，长度5`
	- 查找:`find("abc");`,`rfind("adbc"); //重后向前找`
	- 替换:`replace(2,3,"haha"); //从下标开始的3个字符被替换为haha`
	- 插入:`insert(5,s2); 将s2插入到下标为5的位置`
	- 转换为char* : `s1.c_str(); //将String转化为const char* `

	
## STL标准库

- 容器：可容纳各种类型的通用数据结构，是类模板
- 迭代器：可以用于依次存取容器中的元素，类似指针
- 算法：用来操作容器中的元素的函数模板
	- sort()来对一个vector中的数据进行排序
	- find()来搜索一个list中的对象

算法本身与他们操作的数据类型无关，因此可以用在简单的数组到高级的数据结构中使用

### 容器概述

可以用于存放各种类型的数据（基本类型的变量，对象等）的数据结构，都是类模板，分为三种：

- 顺序容器：vector，deque，list
- 关联容器：set，multiset，map，mutimap
- 容器适配器：stack，queue，priority_queue 

### 顺序容器

容器并非排序的，元素的插入位置通元素的值无关。有vector，deque，list三种。

- vector：头文件<vector>，动态数组。元素在内存中连续存放。随机存取任何元素都能在常数时间完成。在尾端增删元素具有较佳的性能（大部分情况下是常数时间）。[A0,A1,A2,A3,...,An] 


- deque: 头文件<deque>,双向队列，元素在内存内连续存放，随机存取任何元素都能在常数时间完成（但次于vector）。在两端增删元素具有较佳的性能

- list：头文件<list>，双向链表，元素在内存中不连续存放。在任何位置增删元素都能在常数时间完成，不支持随机存取。


### 关联容器

- 元素是排序的
- 插入任何元素，都按相应的排序规则来确定其位置
- 在查找时具有很好的性能
- 通常以平衡二叉树方式实现，插入和检索的时间都是O(log(N))
- set/multiset 头文件<set>，即集合。不允许有相同的元素，multiset中允许存在相同的元素 
- map/multimap 头文件<map>，map与set不同在于map中的元素有且皆有两个成员变量，一个名为first，一个名为second，map根据first值进行大小排序，并可以快速的根据first来检索。map同multimap的区别在于是否允许相同的first值。


### 容器适配器

- stack：头文件<stack>，栈，是项的有限序列，并满足序列中被删除，检索和修改的项只能是最近插入序列的项（栈顶的项）

- queue：头文件<queue>，队列，插入只可以在尾部进行，删除，检索和修改只允许从头部进行，先进先出

- priority_queue：头文件<queue>，优先级队列


### 顺序容器和关联容器中都有的成员函数

- begin: 返回指向容器中第一个元素的迭代器
- end: 返回指向容器中最后一个元素的迭代器
- rbegin:返回指向容器中最后一个元素的迭代器
- rend:返回指向容器中第一个元素的迭代器
- erase: 从容器中删除一个或几个元素
- clear: 从容器中删除所有元素

### 顺序容器的常用成员函数

- front:返回容器中第一个元素的引用
- end:返回容器中最后一个元素的引用
- push_back:在容器末尾增加新元素
- pop_back:删除容器末尾的元素
- erase:删除迭代器指向的元素

### 迭代器

- 用于指向顺序容器和关联容器中的元素
- 用法和指针类似
- 有const和非const两种
- 通过迭代器可以读取它指向的元素
- 定义:

```
容器类名::iterator 变量名;
容器类名::const_iterator 变量名;
```

- 访问迭代器指向的元素:

```
* 迭代器变量名

```

- 使用迭代器遍历vector

```cpp

//方法1:
vector<int> v(100);
for(int i=0; i<v.size(); i++)
{
	cout<<v[i];
}

//方法2:
vector<int>::const_iterator i;
for(i = v.begin();i != v.end(); i++)
{
	cout<<*i;
}


```

- 使用迭代器遍历list

```cpp

//双向迭代器不支持<，list也不支持[]访问



```

### 算法

- 算法就是一个个函数模板，大多数在<algorithm>中定义
- STL中提供能在各种容器中通用的算法，比如查找，排序等
- 算法通过迭代器来操作容器中的元素，许多算法可以对容器中的一个局部区间进行操作，因此需要两个参数，一个是起始元素的迭代器，一个是终止元素的后面一个元素的迭代器。比如，排序和查找

- 有的算法返回一个迭代器，比如find()算法，在容器中查找一个元素，并返回一个指向该元素的迭代器

- 算法可以处理容器，也可以处理普通数组

## 顺序容器

### vector

- 可变长的动态数组
- 必须包含头文件`#include`
- 支持随机访问迭代器
	- 根据下标随机访问某个元素，时间为常数
	- 在尾部添加速度很快
	- 在中间插入很慢 

- 构造函数初始化:

```
vector();//无参数构造函数，将容器初始化为空
vector(int n); //将容器初始化为n个成员的容器
vector(int n, const T& val); //初始化n个成员为val的容器
vector(itor first, itor last); //通过迭代器初始化vector

```

- 常用API：

```
void pop_back(); //删除末尾元素
void push_back(const T& val); //末尾添加元素



```

## 关联容器

- 内部元素有序排列，新元素插入的位置取决于它的值，查找速度快
- 除了各容器都有的函数外，还支持以下成员函数
	- `find`：查找等于某个值的元素(x小于y和y小于x同时不成立即为相等)
	- `lower_bound`：查找某个下界
	- `upper_bound`：查找某个上界
	- `equal_range`：同时查找上界和下界

### multiset

- 预备知识:pair模板:

```cpp

template<class T1, class T2>
struct pair{
	typedef T1 first_type;
	typedef T2 second_type;
	T1 first;
	T2 second;
	pair():first(),second(){}
	pair(const T1& _a, const T2& _b):first(_a),second(_b)	{}
	template<class U1, class U2>
	pair(const pair<_U1,_U2>& _p):first(_p.first),second:(_p.second){
	
	};

}


```

map,multimap容器里存放着的都是pair模板类对象，且first从小到大排序，第三个构造函数用法实例：

```cpp

pair<int, int> p(pair<double,double>(5.5,4.6))

//p.first = 5,
//p.second=  4

```

- multiset

```cpp

template<class key, class Pred=less<key>,class A = allocator<key>>
class multiset{...}

```

- pred类型的变量决定了multiset中的元素顺序是怎么定义的。multiset运行过程中，比较两个元素x，y大小的做法是通过Pred类型的变量，例如`op`，若表达式`op(x,y)`返回true则 x比y 小，`Pred`的缺省类型为`less<key>`，其中less模板为一个functor:

```cpp

template<class T>
struct less:publi binary_function<T,T,bool>
{
	bool operator()(const T& x, const T& y) const{
		return x<y;
	}
}
//less模板是靠<比较大小的

```
- 成员函数:
	- `iterator find(const T&val);`：在容器中查找值为val的元素，返回其迭代器。如果找不到，返回end()。
	- `iterator insert(const T& vale);`：将val插入到容器中并返回其迭代器
	- `void insert(iterator first, iterator last);`将区间[first,last)插入容器
	- `int count(const T& val);`统计多少个元素的值和val相等
	- `iterator lower_bound(const T& val);`查找一个最大位置it，使得[begin(),it)中所有元素都比val小。
	- `iterator upper_bound(const T& val);`查找一个最大位置it，使得[it,end())中所有元素都比val小。

- `multiset`的用法

```cpp

class A{};
int main(){
	
	std::multiset<A> a;
	a.insert(A()); //error,由于A没有重载<无法比大小，因此insert后编译器无法知道插入的位置
}

//multiset<A> a;
//等价于
//multiset<A,less<A>> a;
//插入元素时，multiset会将被插入的元素和已有的元素进行比较。由于less模板使用<进行比较，所以，这都要求A对象能用<比较，即适当重载了<

```

- `multiset`用法2

```cpp

class A
{
	private: 
		int n;
	
	public:
		A(int n_):n(n_){}
		
	friend bool operator<(const A& a1, const A& a2){
		return a1.n < a2.n;
	}	
	friend class Myless;
}


struct Myless{
	
	
	bool operator()(const A& a1, const A& a2){
		return (a1.n % 10)< (a2.n % 10);
	}
}

int main(){
	const int SIZE = 6;
	A a[SIZE] = {4,22,3,9,12};
	std::multiset<A> set;
	set.insert(a,a+SIZE);
	set.insert(22);
	set.count(22); //2
	
	//查找:
	std::multiset<A>::iterator pp = set.find(9);
	if(pp != set.end()){
		//说明找到
	}
	
	
}

```

### set

set中不允许有重复元素,插入set中已有元素时，忽略：

```
tempate<class key, class pred = less<key>>
class set{...}


```

- 用法实例:

```cpp

int main(){

	std::set<int> ::iterator IT;
	int a[5] = {3,4,5,1,2};
	set<int> st(a,a+5);
	pair<IT,bool> result;
	result = st.insert(6);
	if(result.second) //插入成功，则输出被插入的元素
	{
	}
	if(st.insert(5).second){
	}
	else{
		//这时候表示插入失败
	}
}

```

### multimap

```cpp

template<class key, class T, class Pred = less<key>, class A = allocator<T>>

class multimap{

...
typedef pair<const key, T> value_type;
...

}; //key代表关键字的类型

```

- multimap中的元素由<key,value>组成，每个元素是一个pair对象，关键字就是first成员变量，类型为key

- multimap中允许多个元素的关键字相同，元素按照first成员变量从小到大排列，缺省用`less<key>`定义关键字的"小于"关系

- multimap实例：

```cpp

#include<map>
using namespace std;
int main(){
	
	typedef multimap<int,double,less<int>> mmid;
	mmid pairs;
	
	//typedef pair<const key, T> value_type;
	pairs.insert(mmid::value_type(15,2.7)); 
	pairs.insert(mmid::value_type(15,99.3)); 
	pairs.count(15); //2
	
	for(mmid::const_iterator i = pairs.begin();
	i != paris.end(); i++){
		
		i->first;
		i->second;
	
	}
	
}

```

### map

```cpp

template<class key,class T, class Pred = less<key>,class A = allocator<T>>
class map{
	
	///
	typedef pair<const key, T> value_type;

};

```

- map中元素都是pair模板类对象。关键字(first成员变量)各不相同。元素按照关键字从小到大排列，缺省情况下用`less<key>`即"<"定义

- map的[]成员函数

若pairs为map模板类对象

```
pairs[key]

```

返回对关键字等于key的元素的值(second成员变量)的引用。若没有关键字key的元素，则会往pairs里插入一个关键字为key的元素，其值用无参构造函数初始化，并返回其值的引用

如：

```cpp

map<int, double> pairs
pairs[50] = 5;

```


## 函数对象

若一个类重载了运算符"()"则该类的对象就成为函数对象

```
class A
{
public:
	int operator()(int a1, int a2)
	{
		return a1+a2;
	}

};

A a; //函数对象
cout << a(1,2); //average.operator()(3,2,3)

```
### STL中的函数对象模板

以下模板可以用来生成函数对象：

- equal_to
- greater
- less

头文件:`<functional>`

greater函数对象模板:

```cpp

template<class T>
struct greater:public binary_function<T,T,bool>{
	
	bool operator()(const T& x, const T& y)const{
		return x>y;
	}

}


```


## 算法

- 大多重载的算法都有两个版本：


## 强制类型转换


- 传统类型转换的风险

传统的类型转换如(int)x;这种编译器无法识别类型转换带来的风险，指针类型转换同样有这个问题，因此需要定义一套标准的类型转换运算符

### static_cast

static_cast用来进行比较“自然”和低风险的转换，比如整形和实数型，字符型之间的转换

static_cast不能在不同类型指针之间互相转换，也不能用于整形和指针之间的相互转换，也不能用于不类型的引用之间转换

```cpp

class A
{
public:
	operator int() { return 1;}
	operator char* () {return NULL;}
};

int main()
{
	A a;
	int n; char* p = "New Dragon Inn";
	n = static_cast<int>(3.14); //n的值变为3
	n = static_cast<int>(a); //调用a.operator int， n=1

	p = static_cast<char* >(a); //调用a.operator char* , p的值为NULL
	n = static_cast<int>(p); //编译错误，static_cast不能将指针转换成整形
	p = static_cast<char*>(n); //编译错误，static_cast不能将整形转为指针
	
	return 0;
}


```

### interpret_cast

用来进行各种不同类型的指针之间的转换，不同类型的引用之间的转换，以及指针和能容纳的下指针的整数类型之间的转换。转换的时候，执行的是逐个比特的拷贝操作

### const_cast

### dynamic_cast

## C++ 11新特性

### 统一初始化方法

```cpp

int arr[3]{1,2,3};
vector<int> iv{1,2,3};
map<int,string> mp{ {1,"a"}, {2,"b"} };
string str{"hello!"};
int *p = new int[20]{1,2,3};

struct A{
	int i,j;
	A(int m, int n):i(m),j(n){
	
	}
};

A func(int m, int n){

	return A{m,n};
}

int main(){
	
	A* pa = new A{3,6};

}

```

### 成员变量默认初始值

```cpp

class B{
	
	public:
		int m=1234;
		int n;
};

```


### auto关键字

用于定义变量，编译器可以自动判断变量类型,变量必须初始化

```cpp

auto i = 100;
auto p = new A();
auto k = 343;

auto add(T1 x, T2 y) -> decltype(x+y){
	return x+y;
}

```

### decltype关键字

求表达式类型:

```
int i;
double t;
struct A {double x;};
const A* a = new A();

decltype(a) x1; //x1 is A*

```

### shared_ptr

- 头文件:<memory>，类模板
- 通过shared_ptr的构造函数，可以让shared_ptr对象托管一个new运算符返回的指针,写法如下：

```
shared_ptr<T> ptr(new T);

```

此后ptr就可以像T* 类型的指针一样使用， *ptr就是用new分配的那个对象，而且不必操心内存释放的事情。

- 多个shared_ptr对象可以同时托管一个指针，系统会维护一个托管计数。当无shared_ptr托管该指针时，delete该指针

- shared_ptr对象不能托管指向动态分配的数组的指针，否则程序会出错


```
#include <memory>

struct A{
	
	int n;
	A(int v = 0) : n(v){}
	~A(){}

};

shared_ptr<A> sp1(new A(10));
A* p = sp1.get(); //获取原始指针

shared_ptr<A> sp2(sp1);
shared_ptr<A> sp3 = sp1;

//此时A(10)对象被三个指针托管

sp1.reset(); //sp1放弃对A的托管

if(!sp1){
	cout << sp1 is null; //sp1放弃对A的托管后，自己也为null
}

A *q = new A(11);
sp1.reset(q); //sp1托管q

shared_ptr<A> sp4(sp1) //sp4托管q
shared_ptr<A> sp5; 
sp5.reset(q); //报错
sp1.reset(); //sp1放弃托管q
sp4.reset(); //sp4放弃托管q

```


### 空指针nullptr

```cpp

int *p1 = NULL;
int *p2 = nullptr;
shared_ptr<int> p3 = nullptr;

//p1 == p2 //yes
//p3 == nullptr //yes
//p3 == p2 //error!
//p3 == NULL //yes

bool b = nullptr; //b = false
int i = nullptr; //error

```

### 基于范围的for循环

```cpp

struct A { int n; A(int i):n(i){}};

int main(){
	
	int arr[] = {1,2,3,4};
	for(int/auto &e : arr)
	{
		e *= 10; 
	}
	
	for(int e : arr)
	{
		cout << e;
	}

}

```

### 右值引用和move语义

右值：一般来说，不能取地址的表达式就是右值，能取地址的，就是左值

```

class A{};

A &r = A(); //error, A()是无名变量，是右值
A &&r = A(); //ok, r是右值引用

```

C++ 11之前引用都是左值引用，右值引用的主要目的是提高程序的运行效率，减少需要进行深拷贝对象的拷贝次数

- 例子：老的写法:

```cpp

struct String{
	char* p;
	
	//默认构造函数
	String():p(new char[1]){ p[0] = '\0'};
	
	//赋值构造函数
	String(const char* s){
		p = new char[strlen(s)+1];
		strcpy(p,s);
	}
	
	//拷贝（复制）构造函数
	String(const String &s){
		p = new char[strlen(s.p)+1]	;
		strcpy(p,s.p);
	}
	
	//重载赋值
	String& operator=(const String &str)
	{
		if(p != str.p)
		{
			delete []p;
			p = new char[strlen(str.p)+1];
			strcpy(p,str.p);
		}
		return *this;
	}
};

```

- 新写法：move constructor

```cpp

//move constructor
String(String && s):p(s.p){
	s.p = new char[1];
	s.p[0] = 0;
	//将参数s.p指向了另一片存储空间

}//此时当前对象的p直接指向了s.p，并不会触发拷贝的操作

//move assigment
String & operator=(String && s){
	if(p != s.p){
		delete[] p;
		p = s.p; //直接赋值
		s.p = new char[1];
		s.p[0] = 0; //将s.p指向另一片内存
	}
	return * this;
}

template<class T>
void MoveSwap(T& a, T& b){
	T tmp(move(a)); //std::move(a)为右值，会调用move constructor，std::move(x)将一个左值变成右值
	a = move(b); //std::move为右值,因此会调用move assignment
	b = move(tmp); //std::move为右值,因此会调用move assignment
}

int main()
{
	String &r = String("this"); //error
	
	String s;
	s = String("ok"); //String("ok")是右值
	String && r = String("this");
	String s1 = "hello", s2 = "world";
	MoveSwap(s1,s2);
	return 0;
}

```

### unorderd_map

```cpp

unorded_map<string, int> turingWinner;
turningWinner.insert(make_pair("Scott",1976));

//查询
unorded_map<string,int>::iterator p = turningWinner.find(name);

if(p != turingWinnder.end())
{
	//
}
else{

	//
}


```

### Lambda表达式

- 只用一次的函数对象，能否不要专门为其编写一个类
- 只调用一次的简单函数，能否在调用时才定义
- 形式：

```

[外部变量访问方式说明符](参数表) -> 返回值类型
{
	//函数体
}

[=] 以传值的形式使用外部所有变量，值不可以被修改
[] 不适用任何外部变量
[&] 以引用的形式使用所有外部变量，引用可以修改
[x,&y] x以传值形式引入， y以引用形式引入
[=,&x,&y] x,y以引用形式使用，其余变量以传值形式引入
[&,x,y] x,y以传值形式引入，其余变量以引用形式使用

"-> 返回值类型"，可以忽略，编译器可以自动推断

```

- 例子:

```cpp

int x=100, y = 200, z = 300;
cout << [](double a, double b){return a+b}(1.2,2.5)<<endl;

auto ff = [=,&y, &z](int n){
	cout <<x<<endl;
	y ++;
	z++ ;
	return n*n;

};

ff(15);

//操作集合
vector<int> a{1,2,3,4};
int total = 0;
for_each(a.begin(), a.end(), [&](int &x){total+=x; x*=2;});

for_each(a.begin(), a.end(), [](int x){cout << x <<endl} ); 

//实现递归
function<int(int)> fib = [&fib](int n){
	return n<2?1:fib(n-1) + fib(n-2);
}

//function<int(int)>表示输入为int，返回值为int的函数

```