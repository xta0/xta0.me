---
list_title: Java基础 | 基本语法 | Basic Syntax
title: Java基本语法
layout: post
categories: [Java]
---

## 数据类型，变量，常量

### Java的数据类型划分

- 基本的数据类型(primitive types):
    - 数值
        - 整数: byte(1), short(2), int(4) ,long(8)
            - 二进制数以`0b`或`0B`开头,`0b00010010`(Java 7以上)
            - Java语言中整形默认为 `int`
            - 声明`long`型：`long a = 3L`后面加`l`或`L`
            - Java中没有“无符号数”, C中的`unsigned int`在Java中用`long`表示
        - 浮点型: float(4), double(8)
            - 科学技术法:
                - `3.14e2`,`3.14E2`
            - Java中浮点型默认为`double`，如果想使用`float`需要:`float a = 3.45f`末尾加`f`
    - 字符型:
        - <mark>Java中的字符用Unicode编码，每个字符两个字节</mark>
            - 可用16进制表示
            - `char c1 = '\uff09 '`
    - 布尔型: boolean
        - 只能用`true`或`false`

- 引用数据类型(reference types):
    - 类: class
    - 接口: interface
    - 数组
- 差别:
	- 基本类型: 变量在栈
		- `double d = 3`
	- 引用类型: 变量引用到堆
		- `Person p = new Person()`
- 赋值:
	- `double d2 = d;` 复制的是值
	- `Person p2 = p;` 复制的是引用

## 运算符与表达式

### 运算符

- 位运算
	- `~`按位取反
	- `&`按位与
	- `|`按位或
	- `^`按位异或

- 移位运算
	- 左移`a<<b`将二进制形式的`a`逐位左移`b`位，空出的补0
	- 带符号右移：`a>>b`将二进制形式的`a`逐位右移`b`位，高位补符号位
	- 无符号右移：`a>>>b`将二进制形式的`a`逐位右移`b`位，高位补符号位
	- 适用于数据类型：byte, short, char, int, long
	- 对于低于int型的数据，系统将先自动换为int型，再移位
	- 对于int型整数移位 `a>>b`, 系统先将b对32取模，得到的结果才是真正的移位数
	- 对于long型整数移位 `a>>b`, 系统先将b对64取模，得到的结果才是真正的移位数

###表达式

- a++的副作用:

```java
public class Test{
    public static void main(String[] args){
        int a = 2;
        int b = a++ + ++a;
        System.out.println(b); 
   }
}
```

上面代码执行结果为6，为什么呢? 使用javap -c 反汇编查看:

```java
Compiled from "Test.java"
public class Test {
  public Test();
    Code:
       0: aload_0
       1: invokespecial #1                  // Method java/lang/Object."<init>":()V
       4: return

  public static void main(java.lang.String[]);
    Code:
       0: iconst_2
       1: istore_1
       2: iload_1
       3: iinc          1, 1
       6: iinc          1, 1
       9: iload_1
      10: iadd
      11: istore_2
      12: getstatic     #2                  // Field java/lang/System.out:Ljava/io/PrintStream;
      15: iload_2
      16: invokevirtual #3                  // Method java/io/PrintStream.println:(I)V
      19: return
}
```

上面是虚拟机的指令:

```
`iconst_2`:定义常量
`istore_1`:将常量存入寄存器1
`iload_1`:从寄存器中读取a = 2
`iinc`: 值+1 => a++ => a = 3
`iinc`: 值+1 => ++a => a = 4
`iload_1`:从寄存器中读取a = 2
`iadd`: 2+4 = 6
`istore_2`: 将6保存到寄存器2，也就赋值给b
```

## Control Flow

### 程序中3种基本流程

- 顺序，分支(if)，循环
- 证明所有程序都可由这三种过程组合来完成
- <mark>C语言中的问题</mark>
    - 没有expression的概念，全是statement，例如没有`x+y;`这种没意义的语句，也没有逗号表达式的概念

- 使用Switch
	- 变量后面可以是整数，字符，字符串
	- `case`后面是常量
	- 注意`break`

- 带标签的continue和break
    - 计算100以内的素数:

    ```java
    public class chap3{
        public static void main(String[] args){
            //计算1~100以内的素数
            out:
            for(int i=1; i<=100; i++){
                for(int j=2; j<i; j++){
                    if(i%j == 0)
                        continue out;
                }
                System.out.println(i);
            }
        }
    }
    ```

## 数组

### 一维数组:

- 一维数组的声明：
	- `int[] a;`
	- `double[] b;`
	- `Mydata[] c;` 

-  注意，方括号可以写到变量名前面，也可以写到后面
- 数组定义和分配空间分别执行:

```java
int []a = new int[3]
a[0] = 1;
a[1] = 2;
a[3] = 3;
```

- Java语言中声明数组时不能指定长度：

```c
int a[5]; //error
```

- 数组时引用类型，必须在heap上分配空间:
	- `int []a = new int[5];`
	- 这里a只是一个引用
- 初始化:
	- 静态初始化:定义数组时就分配空间并初始化

	```java	
	int[] a = {3,9,8};
	int[] b = new int[]{3,9,8};
	Mydata[] dates = { new Mydata(22,8), new Mydata(99,8)};
	```
	
	- 注意最后可以多一个逗号:如`{3,9,8,}`
	- 数组一经分配空间，其中的元素会被赋值为默认值0或`null`,如:

	```java
	int []a = new int[5]
	//a中的元素为0
	```
	
- 数组元素的引用:
	- index为下标:`a[3],b[i]...`

- 快速枚举:

```java
int[] ages = new int[10];
for(int age : ages){
	print(age);
}

```

### 多维数组

- 二维数组是数组的数组

```java
int [][]a = { {1,2}, {3,4}, {5,6} };
int [][]t = new int [3][]
t[0] = new int[2];
t[1] = new int[3];
t[2] = new int[4]; 
```


## 类

### 字段和方法

```java
class Person{
    String name;
    int age;
    void sayHello(){
        print("hello");
    }
}
```

- 字段：是类的属性，用变量表示的
- 字段又称为域，域变量，属性，成员变量等
- 方法：是类的功能和操作，是用函数表示的

### 构造方法

- 一种特殊的方法
- 用来初始化(new)该类的一个新对象
- 构造方法和类名同名，而且不写返回数据类型:

```java
Person(String n, int a){
this.name = n;
this.age = a;
}
```

- 一个类至少有多个构造方法
- 如果没有定义任何构造方法，系统会自动产生一个默认构造方法，没任何参数，方法体为空

### 方法重载

- overloading: 多个方法有相同的名字，编译时能识别出来。
- 这些方法的签名不同，或者参数的个数不同，或者类型不同
- 通过方法重载可以实现多态


###this的使用

- 在方法和构造方法中，使用this来访问字段及方法
- 使用this解决局部变量与字段同名
- 构造方法中用this调用另一个构造方法:

```java
Person(){
this(0,""); //这种情况，这条语句必须放在第一句:
...
}
```


## 继承

- Java支持单继承： 一个类只能有一个直接父类
- 关键字`extends`,如果没有`extends`，则默认继承`java.lang.Object`的子类
- 父类的非私有方法自动飞机城
- 方法的(Override)
- 子类可以重新定义与父类同名的方法，实现对父类方法的overrride：

```java 
@override //JDK 1.5后可以用这个标记来表示(不用也可以)
void sayHello(){
System.out.println("Hello");
} 
```

- 方法的Overload
    一个类中可以有几个同名的方法，这称为方法的重载，同时，开可以重载父类的同名方法。与override不同的是，重载不要求参数类型列表相同。重载方法实际上是新加的方法:
    
```java
void sayHello(Student stu){
    System.out.println(stu.name);
}
```

### 使用super

- 使用super访问父类的域和方法，如果父类和子类有同名字段或方法，使用super区分：

```java

void sayHello()
{
super.sayHello();
System.out.print("a");
}

```
上面例子中，子类override了父类的方法，也可以用super调用父类的方法

- 使用父类的构造方法
- 构造方法是不能继承的
    - 比如，父类Person有一个构造方法`Person(String,int)`,不能说子类Student也自动有一个构造方法`Student(String,int)`

- 子类的构造方法中，可以用`super`调用父类的构造方法

```java
Student(String name, int age, String school)
{
    super(name,age);
    this.school = school;
}
```
使用时，`super()`必须方法到第一句

- 类型转换

## Package

- 引入package是为了解决名字空间，名字冲突
- 它与类的继承没有关系。事实上，一个子类的父类可以位于不同的包里
- package有两方面含义：
- 一是名字空间，存储路径
- 一是可访问性（同一个包种的个各类，默认情况可以互相访问）
- `import`引入所需要的类 
- `import a.b.*` 使用*号只能表示本层次所有的类，不包括子层次下的类
- Java编译器会自动导入`java.lang.*`中的类
- 使用`javac`可以将`.class`文件放到指定目录，只需要使用一个命令`-d`来表明包得根目录
- 例如 `javac -d ~/test/ ~/test/chap1/test.java`意思是将编译好的文件放到`~/test`下面。
- 这时候运行时需要指定包名:`java chap1.test`
- 在编译和运行程序中，经常要用到多个包，怎么指定这些包得目录呢？包层次的根目录是由环境变量`CLASSPATH`来确定的。具体操作有两种方法:
- 一是，在`java`及`javac`的命令行中，用`-classpath`(或`-cp`)来指明: `java -classpath ~/chap04; ~/chap05; ~/chap06/class; .Hello`
- 二是设定`classpath`环境变量，用命令行设定一个`CLASSPATH`变量

## Modifiers（修饰符）

### 两类

- 访问修饰符(access modifiers)
    - 如`public/private`等

- 其它修饰符
    - 如`abstract`等

- 可以修饰类，也可以修饰成员的字段，方法

|       none    | 同一类中  | 同一个包中 | 不同包中的子类 | 不同包中的非子类 |
| ------------- |---------| ----------|-------------|-------------|
| private       | yes |
| 默认(包可访问)  | yes |           yes |       
| protected     | yes |           yes |   yes |        
| public        | yes |           yes |   yes |     yes|

### 类的访问控制符

- 类的访问控制符或者为`public`，或者默认
- 如果用public修饰，则类可以被其他类所访问
- 如果没有public修饰，则该类只能被同包中的类访问

### setter和getter

- 将字段用`private`修饰，从而更好的将信息进行封装和隐藏
- 用`setXXX`和`getXXX`方法对类的属性进行存取，分别称为`setter`和`getter`
- 属性用`private`更好的封装和隐藏，外部类不能随意存取和修改
- 提供方法来存取对象的属性，在方法中可以对给定的参数进行合法性校验
- 方法可以同来给出计算后的值
- 方法可以完成其他的必要工作(如清理资源，设定状态，等)
- 只提供`getter`不提供`setter`可以保证属性是只读的

```java

class Person2
{
private int age;

public void setAge(int age)
{
    if(age > 0 && age < 200)
        this.age = age; 
} 

public int getAge(){
        return this.age;
    }
}
```

## 非访问控制符 

|       none    | 基本含义  | 修饰类 | 修饰成员 | 修饰局部变量 |
| ------------- |---------| ----------|-------------|-------------|
| static        | 静态的，非实例的，类的 | 可以修饰内部类 | yes|
| final  		   | 最终的，不可改变的 | yes | yes | yes |        
| abstract      | 抽象的，不可实例化的 |  yes |   yes（修饰抽象的方法） |        

### static字段

- static最本质的特点是
- 他们是修饰类的，不属于任何一个对象实例。
- 它不保存在某个对象实例的内存区间，而是保存在类的内存区域的公共存储单元
- 类变量可以通过类名直接访问，也可以通过实例对象来访问，这两种方法的结果是相同的
    - 例如`System`类的

### 接口与抽象类的区别和联系:

- 接口：只能定义全局常量和公共抽象方法
- 组成：
- 接口必须有子类，并且子类要实现全部的抽象方法
- 一个子类可以实现多个接口，但是只能继承一个抽象类
- 一个接口可以继承多个接口
- 一个子类可以继承抽象类，同时实现接口 


## Resources

- [Core Java Volumn 1](http://ptgmedia.pearsoncmg.com/images/9780137081899/samplepages/0137081898.pdf)
- [Core Java Volumn 2](http://ptgmedia.pearsoncmg.com/images/9780137081608/samplepages/013708160X.pdf)